"""Live person detection and persistent ID association, on MPS/CUDA/CPU.

Run: python person_tracking.py --camera 1
YOLO26n detects people, BoT-SORT compensates camera motion, and AIM-LTCC
body embeddings connect observations to the long-term identity gallery.
BPBreID matches mutually visible body parts when furniture hides the body.
YuNet/SFace provides face-first identity evidence when a clear face is visible.
Sources: docs.ultralytics.com/modes/track, github.com/BoomShakaY/AIM-CCReID,
github.com/VlSomers/bpbreid and github.com/opencv/opencv_zoo
(licenses and notices in vendor/).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import time
from collections import deque
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from urllib.request import urlopen

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


CACHE = Path.home() / ".cache" / "aix-classroom-tracking"
CHECKPOINTS = {
    "yolo26n.pt": (
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt",
        "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef",
    ),
    "aim-ltcc.pth": (
        "https://drive.usercontent.google.com/download?"
        "id=1YhFuoEOpnwS2seMxFZzzNt5NySkKBJxk&export=download&confirm=t",
        "490d23a68d535f9a7c1bd0abc3424042069a2d6ad9a4f728fab0cfcb59e19924",
    ),
    "bpbreid-occluded-duke.pth": (
        "https://drive.usercontent.google.com/download?"
        "id=1o12pu04q52vZaqkNZ6gC1RGz6VVATlsb&export=download&confirm=t",
        "96aa67300880df450fc0cd20c4228c212618897572c0ffb3d528eb982721189c",
    ),
    "face_detection_yunet_2026may.onnx": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
        "48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet/"
        "face_detection_yunet_2026may.onnx",
        "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0",
    ),
    "face_recognition_sface_2021dec.onnx": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
        "48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_recognition_sface/"
        "face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
}
COLORS = ((110, 214, 83), (245, 170, 55), (127, 115, 249), (88, 209, 240), (227, 123, 188))


def checkpoint(name: str) -> Path:
    url, expected_hash = CHECKPOINTS[name]
    path = CACHE / name
    if path.is_file():
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() == expected_hash:
                return path
    CACHE.mkdir(parents=True, exist_ok=True)
    print(f"모델 다운로드 중  {name}", flush=True)
    with tempfile.TemporaryDirectory(dir=CACHE) as staging:
        temporary = Path(staging) / name
        with urlopen(url, timeout=90) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream, 1024 * 1024)
        with temporary.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected_hash:
                raise RuntimeError(f"가중치 해시를 확인하지 못했습니다. {name}")
        temporary.replace(path)
    return path


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type not in {"cuda", "mps", "cpu"}:
        raise ValueError("장치는 auto, mps, cuda:0, cpu 중에서 지정하세요.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("이 Python 환경에서 CUDA를 사용할 수 없습니다.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("이 Python 환경에서 MPS를 사용할 수 없습니다. ARM64 Python을 확인하세요.")
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
        self.dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.model = AIMBackbone()
        weights = torch.load(checkpoint("aim-ltcc.pth"), map_location="cpu", weights_only=True)
        self.model.load_state_dict(weights["model_state_dict"], strict=True)
        self.model.eval().to(device=device, dtype=self.dtype)
        self.mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def encode(self, crops: list[np.ndarray]) -> list[np.ndarray]:
        features = []
        for start in range(0, len(crops), self.batch_size):
            images = [
                np.asarray(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).resize(
                    (192, 384), Image.Resampling.BILINEAR))
                for crop in crops[start:start + self.batch_size]
            ]
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


def deduplicate_people(boxes):
    """Remove repeated person hypotheses before they create separate tracks.

    Upper-body and full-body hypotheses can have low IoU despite sharing the
    same head. Nested boxes also need aligned tops, centers and similar widths
    so ordinary overlaps between different people remain available to track.
    """
    kept = []
    for index in np.argsort(-boxes.conf, kind="stable"):
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
            shared_upper_body = (
                overlap / min(box_size, other_size) >= 0.92
                and abs(box[1] - other[1]) <= 0.10 * min_height
                and abs((box[0] + box[2]) - (other[0] + other[2])) <= 0.30 * min_width
                and max(widths) <= 1.35 * min_width
            )
            if iou >= 0.85 or shared_upper_body:
                duplicate = True
                break
        if not duplicate:
            kept.append(int(index))
    return boxes[np.asarray(kept, dtype=np.int64)]


def crop_quality(frame: np.ndarray, box, other_boxes) -> float:
    x1, y1, x2, y2 = box
    if x2 - x1 < 26 or y2 - y1 < 64:
        return 0.0
    if any(intersection(box, other) / max(1, area(box)) > 0.45 for other in other_boxes):
        return 0.0
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_32F).var()
    return float(min(1.0, sharpness / 60.0))


def occlusion_mask(box, other_boxes):
    """Exclude pixels shared with other detected people from part pooling."""
    x1, y1, x2, y2 = box
    mask = np.zeros((y2 - y1, x2 - x1), dtype=bool)
    for other in other_boxes:
        left, top = max(x1, other[0]), max(y1, other[1])
        right, bottom = min(x2, other[2]), min(y2, other[3])
        if right > left and bottom > top:
            mask[top - y1:bottom - y1, left - x1:right - x1] = True
    return mask


class PersonTracking:
    def __init__(self, device: torch.device, image_size=768, reid_interval=0.2, batch_size=4):
        self.device = device
        self.image_size = image_size
        self.reid_interval = reid_interval
        self.detector = YOLO(str(checkpoint("yolo26n.pt")), task="detect")
        self.encoder = BodyEncoder(device, batch_size)
        self.partial_encoder = PartialBodyEncoder(
            device, batch_size, checkpoint("bpbreid-occluded-duke.pth"))
        self.face_encoder = FaceEncoder(
            checkpoint("face_detection_yunet_2026may.onnx"),
            checkpoint("face_recognition_sface_2021dec.onnx"))
        self.last_face_at = float("-inf")
        config = YAML.load(check_yaml("botsort.yaml"))
        config.update(gmc_method="orb", with_reid=False, track_buffer=30)
        self.motion_tracker = BOTSORT(SimpleNamespace(**config))
        self.identities = IdentityTracker(diagnostics=True)
        self.sampled_at = {}
        self.display_times = deque(maxlen=30)

    def reset_session(self):
        """Start assigning IDs from P001 again using the loaded models."""
        self.motion_tracker.reset()
        self.identities = IdentityTracker(self.identities.config, diagnostics=self.identities.diagnostics)
        self.sampled_at.clear()
        self.last_face_at = float("-inf")
        self.display_times.clear()

    def process(self, frame: np.ndarray, timestamp: float):
        """Return raw observations and global assignments before drawing overlays."""
        result = self.detector.predict(
            frame, classes=[0], conf=0.10, imgsz=self.image_size,
            device=str(self.device), quantize=16 if self.device.type == "cuda" else 32,
            max_det=100, verbose=False,
        )[0]
        detections = deduplicate_people(result.boxes.cpu().numpy())
        tracks = self.motion_tracker.update(detections, frame)
        height, width = frame.shape[:2]
        observations = []
        crops, indices, exclusions = [], [], []
        boxes = []
        for row in tracks:
            x1, y1, x2, y2 = map(round, row[:4])
            boxes.append((max(0, x1), max(0, y1), min(width, x2), min(height, y2)))
        for index, (row, box) in enumerate(zip(tracks, boxes)):
            if not area(box):
                continue
            raw_id, confidence = int(row[4]), float(row[5])
            others = [other for i, other in enumerate(boxes) if i != index]
            quality = crop_quality(frame, box, others)
            observations.append(Observation(raw_id, box, confidence, quality=quality))
            # A clear head/torso can still be useful when the full person box
            # overlaps another person. Judge local parts after masking overlap.
            sharpness = crop_quality(frame, box, [])
            if sharpness >= 0.7 and timestamp - self.sampled_at.get(raw_id, float("-inf")) >= self.reid_interval:
                x1, y1, x2, y2 = box
                indices.append(len(observations) - 1)
                crops.append(frame[y1:y2, x1:x2])
                exclusions.append(occlusion_mask(box, others))
                self.sampled_at[raw_id] = timestamp
        whole_crops, whole_indices = [], []
        for index, crop, (parts, visibility) in zip(
                indices, crops, self.partial_encoder.encode(crops, exclusion_masks=exclusions)):
            observation = observations[index]
            observation.part_embeddings = parts
            observation.part_visibility = visibility
            visible_groups = {group for group, score in zip(PART_GROUPS, visibility)
                              if score >= self.identities.config.part_min_visibility}
            observation.allow_enrollment = len(visible_groups) >= self.identities.config.part_min_common
            # Keep the existing appearance branch on clear partial crops too:
            # a continuously tracked upper-body view can become a useful anchor.
            # Sparse part evidence blocks enrollment, not matching known views.
            if observation.quality >= 0.7:
                whole_crops.append(crop)
                whole_indices.append(index)
        for index, embedding in zip(whole_indices, self.encoder.encode(whole_crops)):
            observations[index].embedding = embedding
        # Face sampling is independent of shirt texture, body sharpness and
        # furniture overlap. Association sees every visible person box so one
        # physical face cannot be attached to two temporary tracks.
        if observations and timestamp - self.last_face_at >= self.reid_interval:
            face_features = self.face_encoder.encode(frame, [o.box for o in observations])
            for observation, (embedding, quality), face_status in zip(
                    observations, face_features, self.face_encoder.last_statuses):
                observation.face_sampled = True
                observation.face_status = face_status
                observation.face_embedding = embedding
                observation.face_quality = quality
                if embedding is not None and quality >= self.identities.config.face_min_quality:
                    observation.allow_enrollment = True
            self.last_face_at = timestamp
        assignments = self.identities.update(observations, timestamp)
        active_ids = {observation.raw_id for observation in observations}
        self.sampled_at = {key: stamp for key, stamp in self.sampled_at.items()
                           if key in active_ids or timestamp - stamp < 30}
        return observations, assignments

    def annotate(self, frame: np.ndarray, timestamp: float):
        observations, assignments = self.process(frame, timestamp)
        for observation, assignment in zip(observations, assignments):
            person_id = assignment.person_id
            label = f"P{person_id:03d}" if person_id is not None else "Checking"
            color = COLORS[(person_id - 1) % len(COLORS)] if person_id else (175, 175, 175)
            if assignment.status == "reidentified":
                label += "  matched"
            draw_label(frame, observation.box, label, color)
        self.display_times.append(time.monotonic())
        elapsed = self.display_times[-1] - self.display_times[0]
        fps = (len(self.display_times) - 1) / elapsed if elapsed else 0.0
        hud = f"{str(self.device).upper()} | {fps:.1f} FPS | Q / Esc to stop"
        bottom = frame.shape[0] - 12
        cv2.putText(frame, hud, (12, bottom), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, hud, (12, bottom), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return frame


def draw_label(frame, box, label, color):
    x1, y1, x2, y2 = box
    scale = min(0.85, max(0.48, frame.shape[1] / 1700))
    (width, height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    top = max(0, y1 - height - baseline - 8)
    left = min(x1, max(0, frame.shape[1] - width - 8))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    cv2.rectangle(frame, (left, top), (left + width + 8, top + height + baseline + 8), color, -1)
    cv2.putText(frame, label, (left + 4, top + height + 2), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (15, 20, 24), 2, cv2.LINE_AA)


def capture_latest(camera_index, frames, stop, errors):
    capture = None
    try:
        capture = cv2.VideoCapture(camera_index)
        if not capture.isOpened():
            raise RuntimeError(f"카메라 {camera_index}를 열지 못했습니다. --camera 번호와 카메라 권한을 확인하세요.")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        capture.set(cv2.CAP_PROP_FPS, 30)
        last_received = time.monotonic()
        while not stop.is_set():
            ok, frame = capture.read()
            timestamp = time.monotonic()
            if ok and frame is not None and frame.size:
                frames.append((timestamp, frame))
                last_received = timestamp
            elif timestamp - last_received >= 3:
                raise RuntimeError("카메라에서 3초 이상 프레임을 받지 못했습니다.")
            else:
                stop.wait(0.02)
    except Exception as error:
        errors.append(error)
    finally:
        if capture is not None:
            capture.release()


def run_camera(camera_index: int, tracking: PersonTracking):
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("카메라 창을 열 수 있는 데스크톱 환경에서 실행하세요.")
    try:
        from PySide6.QtCore import QLibraryInfo, QTimer
        from PySide6.QtWidgets import QApplication
        from tracking_window import TrackingWindow
    except ImportError as error:
        raise RuntimeError("UI를 사용하려면 requirements.txt의 패키지를 설치하세요.") from error

    # The Linux OpenCV wheel can point Qt at its own Qt 5 plugins.
    if os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH") == str(Path(cv2.__file__).parent / "qt/plugins"):
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
    app = QApplication.instance()
    arguments = [sys.argv[0]]
    if app is None and sys.platform == "darwin":
        from stat import UF_HIDDEN

        # Qt skips hidden plugins. iCloud-backed environments can mark this
        # wheel's platform plugins hidden. Load copies from local storage.
        plugins = Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath))
        platform_files = list((plugins / "platforms").glob("libq*.dylib"))
        if any(plugin.stat().st_flags & UF_HIDDEN for plugin in platform_files):
            with tempfile.TemporaryDirectory(prefix="aix-qt-") as directory:
                for plugin in platform_files:
                    shutil.copyfile(plugin, Path(directory) / plugin.name)
                app = QApplication(arguments + ["-platformpluginpath", directory])
    if app is None:
        app = QApplication(arguments)
    frames, errors = deque(maxlen=1), deque(maxlen=1)
    stop = Event()
    worker = Thread(target=capture_latest, args=(camera_index, frames, stop, errors), daemon=True)

    def reset_session():
        tracking.reset_session()
        frames.clear()

    window = TrackingWindow(camera_index, tracking.device, COLORS, reset_session)
    timer = QTimer(window)

    def poll_frame():
        if errors:
            window.close()
            return
        if not frames:
            return
        try:
            timestamp, frame = frames.pop()
            observations, assignments = tracking.process(frame, timestamp)
            window.show_frame(frame, observations, assignments, tracking.identities.identities)
        except (Exception, KeyboardInterrupt) as error:
            errors.append(error)
            window.close()

    timer.timeout.connect(poll_frame)
    window.show()
    try:
        worker.start()
        timer.start(10)
        app.exec()
        if errors:
            raise errors[0]
    finally:
        timer.stop()
        stop.set()
        if worker.ident is not None:
            worker.join(timeout=4)
        if worker.is_alive():
            print("카메라 응답이 지연되고 있습니다. 장치가 잠겨 있으면 Python 실행을 종료하세요.")


def main():
    parser = argparse.ArgumentParser(description="딥러닝 특징을 누적하는 실시간 사람 ID 추적")
    parser.add_argument("--camera", type=int, default=1, help="카메라 번호. 이 Mac의 iPhone은 1, 내장 카메라는 0")
    parser.add_argument("--device", default="auto", help="auto, mps, cuda:0, cpu")
    parser.add_argument("--size", type=int, default=768, help="검출 입력 크기")
    parser.add_argument("--reid-interval", type=float, default=0.2, help="외형 특징 갱신 간격(초)")
    parser.add_argument("--reid-batch", type=int, default=4, help="재식별 추론 배치 크기")
    args = parser.parse_args()
    if args.camera < 0 or args.size < 128 or args.reid_interval <= 0 or args.reid_batch < 1:
        parser.error("카메라는 0 이상, 입력 크기는 128 이상, 간격과 배치는 양수로 지정하세요.")
    try:
        device = select_device(args.device)
        if device.type == "cpu":
            torch.set_num_threads(min(torch.get_num_threads(), 6))
        print(f"실행 장치 {device} · 카메라 {args.camera}", flush=True)
        tracking = PersonTracking(device, args.size, args.reid_interval, args.reid_batch)
        run_camera(args.camera, tracking)
    except KeyboardInterrupt:
        print("추적을 종료했습니다.")
    except (RuntimeError, ValueError, OSError) as error:
        print(f"실행 오류 — {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
