"""Presentation runtime: validated person detection and persistent identity tracking."""
from __future__ import annotations
import hashlib
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision.models import resnet50
from ultralytics import YOLO
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_yaml
from identity_tracker import IdentityTracker, Observation
from face_encoder import FaceEncoder
from partial_encoder import PART_GROUPS, PartialBodyEncoder
from person_verifier import PersonVerifier
from models import checkpoint, detector_path

COLORS = ((110, 214, 83), (245, 170, 55), (127, 115, 249), (88, 209, 240), (227, 123, 188))

def select_device(requested: str) -> torch.device:
    if requested == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    device = torch.device(requested)
    if device.type not in {'cuda', 'mps', 'cpu'}:
        raise ValueError('장치는 auto, mps, cuda:0, cpu 중에서 지정하세요.')
    if device.type == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('이 Python 환경에서 CUDA를 사용할 수 없습니다.')
    if device.type == 'mps' and (not torch.backends.mps.is_available()):
        raise RuntimeError('이 Python 환경에서 MPS를 사용할 수 없습니다. ARM64 Python을 확인하세요.')
    return device

class AIMBackbone(nn.Module):
    """AIM's released LTCC ResNet-50 backbone and max/average pooling."""

    def __init__(self):
        super().__init__()
        backbone = resnet50(weights=None)
        backbone.layer4[0].conv2.stride = (1, 1)
        backbone.layer4[0].downsample[0].stride = (1, 1)
        for name, module in list(backbone.named_children())[:-2]:
            self.add_module(name, module)
        self.bn = nn.BatchNorm1d(4096)

    def forward(self, image):
        feature = self.maxpool(self.relu(self.bn1(self.conv1(image))))
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            feature = layer(feature)
        pooled = torch.cat((feature.amax(dim=(2, 3)), feature.mean(dim=(2, 3))), dim=1)
        return self.bn(pooled)

class BodyEncoder:

    def __init__(self, device: torch.device, batch_size: int):
        self.device = device
        self.batch_size = batch_size
        self.dtype = torch.float16 if device.type == 'cuda' else torch.float32
        self.model = AIMBackbone()
        weights = torch.load(checkpoint('aim-ltcc.pth'), map_location='cpu', weights_only=True)
        self.model.load_state_dict(weights['model_state_dict'], strict=True)
        self.model.eval().to(device=device, dtype=self.dtype)
        self.mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def encode(self, crops: list[np.ndarray]) -> list[np.ndarray]:
        features = []
        for start in range(0, len(crops), self.batch_size):
            images = [np.asarray(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).resize((192, 384), Image.Resampling.BILINEAR)) for crop in crops[start:start + self.batch_size]]
            pixels = torch.from_numpy(np.stack(images)).to(self.device).permute(0, 3, 1, 2).float() / 255
            batch = ((pixels - self.mean) / self.std).to(self.dtype)
            feature = self.model(batch).float()
            feature += self.model(batch.flip(3)).float()
            features.extend(F.normalize(feature, dim=1).cpu().numpy())
        return features

def area(box) -> float:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

def intersection(a, b) -> float:
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))

def filter_foreground_arms(boxes):
    """Reject broad lower-frame hypotheses entering from a side before ID tracking."""
    frame_height, frame_width = boxes.orig_shape
    x1, y1, x2, y2 = boxes.xyxy.T
    width, height = (x2 - x1, y2 - y1)
    crosses_foreground = (width >= 0.5 * frame_width) & (width >= 1.55 * height) & (y1 >= 0.2 * frame_height) & (y2 >= 0.84 * frame_height)
    touches_side = (x1 <= 0.02 * frame_width) | (x2 >= 0.98 * frame_width)
    return boxes[~(crosses_foreground & touches_side)]

def filter_adjacent_limbs(boxes, frame, has_face):
    """Check small detections at the lower sides of a much larger nearby person."""
    frame_height, frame_width = boxes.orig_shape
    x1, y1, x2, y2 = boxes.xyxy.T
    widths, heights = (x2 - x1, y2 - y1)
    centers = (x1 + x2) / 2
    foreground = (widths >= 0.3 * frame_width) & (heights >= 0.55 * frame_height)
    fragments = np.zeros(len(boxes), dtype=bool)
    for index in np.flatnonzero(foreground):
        width, height = (widths[index], heights[index])
        side_distance = np.minimum(abs(centers - x1[index]), abs(centers - x2[index]))
        fragments |= (widths > 0) & (widths <= 0.42 * width) & (heights > 0) & (heights <= 0.32 * height) & (side_distance <= 0.15 * width) & (y1 >= y1[index] + 0.55 * height) & (y2 <= y2[index] + 0.05 * height)
    if not fragments.any():
        return boxes
    for index in np.flatnonzero(fragments):
        left, top, right, bottom = np.rint(boxes.xyxy[index]).astype(int)
        if has_face(frame[max(0, top):bottom, max(0, left):right]):
            fragments[index] = False
    return boxes[~fragments]

def deduplicate_people(boxes):
    """Remove repeated person hypotheses before they create separate tracks.

    Upper-body and full-body hypotheses can have low IoU despite sharing the
    same head. Nested boxes also need aligned tops, centers and similar widths
    so ordinary overlaps between different people remain available to track.
    """
    kept = []
    for index in np.argsort(-boxes.conf, kind='stable'):
        box = boxes.xyxy[index]
        box_size = area(box)
        if box_size <= 0:
            continue
        duplicate = False
        for previous in kept:
            other = boxes.xyxy[previous]
            other_size = area(other)
            overlap = intersection(box, other)
            iou = overlap / (box_size + other_size - overlap)
            widths = (box[2] - box[0], other[2] - other[0])
            min_width = min(widths)
            min_height = min(box[3] - box[1], other[3] - other[1])
            shared_upper_body = overlap / min(box_size, other_size) >= 0.92 and abs(box[1] - other[1]) <= 0.1 * min_height and (abs(box[0] + box[2] - (other[0] + other[2])) <= 0.3 * min_width) and (max(widths) <= 1.35 * min_width)
            if iou >= 0.85 or shared_upper_body:
                duplicate = True
                break
        if not duplicate:
            kept.append(int(index))
    return boxes[np.asarray(kept, dtype=np.int64)]

def crop_quality(frame: np.ndarray, box, other_boxes) -> float:
    x1, y1, x2, y2 = box
    if x2 - x1 < 26 or y2 - y1 < 32:
        return 0.0
    if any((intersection(box, other) / max(1, area(box)) > 0.45 for other in other_boxes)):
        return 0.0
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_32F).var()
    return float(min(1.0, sharpness / 60.0))

def occlusion_mask(box, other_boxes):
    """Exclude pixels shared with other detected people from part pooling."""
    x1, y1, x2, y2 = box
    mask = np.zeros((y2 - y1, x2 - x1), dtype=bool)
    for other in other_boxes:
        left, top = (max(x1, other[0]), max(y1, other[1]))
        right, bottom = (min(x2, other[2]), min(y2, other[3]))
        if right > left and bottom > top:
            mask[top - y1:bottom - y1, left - x1:right - x1] = True
    return mask

def preserve_verified_head(mask, box, support):
    """Preserve a verified head and return its region for a sharpness check."""
    x1, y1, x2, y2 = box
    cx, cy = support.anchor
    radius = 0.32 * support.shoulder_span
    left, right = (max(0, round(cx - radius) - x1), min(x2 - x1, round(cx + radius) - x1))
    top, bottom = (max(0, round(cy - 1.25 * radius) - y1), min(y2 - y1, round(cy + 1.25 * radius) - y1))
    if right - left >= 12 and bottom - top >= 12:
        mask[top:bottom, left:right] = False
        return (x1 + left, y1 + top, x1 + right, y1 + bottom)
    return None

class PersonTracking:

    def __init__(self, device: torch.device, image_size=768, reid_interval=0.2, batch_size=4, *, detector_weights=None):
        self.device = device
        self.image_size = image_size
        self.reid_interval = reid_interval
        self.detector_weights = Path(detector_weights).expanduser().resolve() if detector_weights is not None else detector_path()
        if not self.detector_weights.is_file():
            raise ValueError(f'사람 탐지 가중치 파일이 없습니다: {self.detector_weights}')
        with self.detector_weights.open('rb') as stream:
            self.detector_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
        self.detector = YOLO(str(self.detector_weights), task='detect')
        if detector_weights is not None and self.detector.names.get(0, '').casefold() != 'person':
            raise ValueError('사용자 지정 탐지 모델의 클래스 0 이름은 person이어야 합니다.')
        self.encoder = BodyEncoder(device, batch_size)
        self.partial_encoder = PartialBodyEncoder(device, batch_size, checkpoint('bpbreid-occluded-duke.pth'))
        self.face_encoder = FaceEncoder(checkpoint('face_detection_yunet_2026may.onnx'), checkpoint('face_recognition_sface_2021dec.onnx'))
        self.person_verifier = PersonVerifier(device, batch_size, checkpoint('yolo26n-pose.pt'))
        self.last_face_at = float('-inf')
        config = YAML.load(check_yaml('botsort.yaml'))
        config.update(gmc_method='orb', with_reid=False, track_buffer=30)
        self.motion_tracker = BOTSORT(SimpleNamespace(**config))
        self.identities = IdentityTracker(diagnostics=True)
        self.sampled_at = {}
        self.display_times = deque(maxlen=30)
        self.last_detection_boxes = np.empty((0, 4), dtype=np.float32)

    def reset_session(self):
        """Start assigning IDs from P001 again using the loaded models."""
        self.motion_tracker.reset()
        self.identities = IdentityTracker(self.identities.config, diagnostics=self.identities.diagnostics)
        self.sampled_at.clear()
        self.last_face_at = float('-inf')
        self.display_times.clear()

    @property
    def last_seen(self):
        return {person_id: identity.last_seen for person_id, identity in self.identities.identities.items()}

    def process(self, frame: np.ndarray, timestamp: float):
        """Return raw observations and global assignments before drawing overlays."""
        result = self.detector.predict(frame, classes=[0], conf=0.1, imgsz=self.image_size, device=str(self.device), quantize=16 if self.device.type == 'cuda' else 32, max_det=100, verbose=False)[0]
        detections = filter_foreground_arms(result.boxes.cpu().numpy())
        detections = deduplicate_people(filter_adjacent_limbs(detections, frame, self.face_encoder.has_face))
        self.last_detection_boxes = np.array(detections.xyxy, copy=True)
        tracks = self.motion_tracker.update(detections, frame)
        height, width = frame.shape[:2]
        observations = []
        crops, indices, exclusions = ([], [], [])
        boxes = []
        for row in tracks:
            x1, y1, x2, y2 = map(round, row[:4])
            boxes.append((max(0, x1), max(0, y1), min(width, x2), min(height, y2)))
        for index, (row, box) in enumerate(zip(tracks, boxes)):
            if not area(box):
                continue
            raw_id, confidence = (int(row[4]), float(row[5]))
            others = [other for i, other in enumerate(boxes) if i != index]
            quality = crop_quality(frame, box, others)
            observations.append(Observation(raw_id, box, confidence, quality=quality, allow_enrollment=False, enrollment_sampled=False))
            sharpness = crop_quality(frame, box, [])
            if sharpness >= 0.7 and timestamp - self.sampled_at.get(raw_id, float('-inf')) >= self.reid_interval:
                x1, y1, x2, y2 = box
                indices.append(len(observations) - 1)
                crops.append(frame[y1:y2, x1:x2])
                exclusions.append(occlusion_mask(box, others))
                self.sampled_at[raw_id] = timestamp
        whole_crops, whole_indices = ([], [])
        supports = self.person_verifier.verify(frame, [observations[i].box for i in indices])
        head_regions = []
        for index, exclusion, support in zip(indices, exclusions, supports):
            head_regions.append(preserve_verified_head(exclusion, observations[index].box, support) if support is not None else None)
        for index, crop, exclusion, support, head_region, (parts, visibility) in zip(indices, crops, exclusions, supports, head_regions, self.partial_encoder.encode(crops, exclusion_masks=exclusions)):
            observation = observations[index]
            observation.part_embeddings = parts
            observation.part_visibility = visibility
            visible_groups = {group for group, score in zip(PART_GROUPS, visibility) if score >= self.identities.config.part_min_visibility}
            verified = support is not None
            if verified:
                observation.person_anchor = support.anchor
                observation.person_confidence = support.score
                if float(exclusion.mean()) <= 0.9 and head_region is not None:
                    left, top, right, bottom = head_region
                    gray = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
                    head_quality = float(min(1.0, cv2.Laplacian(gray, cv2.CV_32F).var() / 60.0))
                    observation.quality = max(observation.quality, min(crop_quality(frame, observation.box, []), head_quality))
            observation.allow_enrollment = observation.quality >= 0.7 and verified
            observation.enrollment_sampled = observation.allow_enrollment
            if observation.quality >= 0.7 and (verified or len(visible_groups) >= self.identities.config.part_min_common):
                encoded_crop = crop.copy()
                encoded_crop[exclusion] = (104, 116, 124)
                whole_crops.append(encoded_crop)
                whole_indices.append(index)
        for index, embedding in zip(whole_indices, self.encoder.encode(whole_crops)):
            observations[index].embedding = embedding
        if observations and timestamp - self.last_face_at >= self.reid_interval:
            face_features = self.face_encoder.encode(frame, [o.box for o in observations])
            for observation, (embedding, quality), face_status in zip(observations, face_features, self.face_encoder.last_statuses):
                observation.face_sampled = True
                observation.face_status = face_status
                observation.face_embedding = embedding
                observation.face_quality = quality
                if embedding is not None and quality >= self.identities.config.face_min_quality:
                    observation.allow_enrollment = True
                    observation.enrollment_sampled = True
            self.last_face_at = timestamp
        assignments = self.identities.update(observations, timestamp)
        active_ids = {observation.raw_id for observation in observations}
        self.sampled_at = {key: stamp for key, stamp in self.sampled_at.items() if key in active_ids or timestamp - stamp < 30}
        return (observations, assignments)
