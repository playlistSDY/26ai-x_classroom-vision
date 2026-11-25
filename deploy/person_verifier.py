"""Verify head/shoulder evidence independently of person ReID segmentation.

COCO keypoint ordering: https://docs.ultralytics.com/tasks/pose/
Crop-level inference preserves small seated students without requiring legs
or a face large enough for biometric comparison. A pose only verifies a human;
it never identifies that human.
"""

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class PersonSupport:
    anchor: tuple[float, float]
    shoulder_span: float
    score: float


def upper_body_support(keypoints, confidence, box):
    """Require an anatomically consistent head and shoulders, including bends."""
    points = np.asarray(keypoints, dtype=np.float32)
    if (points.shape != (17, 3) or not np.isfinite(points).all()
            or not np.isfinite(confidence) or confidence < 0.4):
        return None
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    if min(width, height) <= 0:
        return None
    head = points[:5]
    visible_head = head[:, 2] >= 0.5
    if visible_head.sum() < 2:
        return None
    shoulders = points[5:7]
    both_shoulders = bool(np.all(shoulders[:, 2] >= 0.45))
    partial_shoulders = (visible_head.sum() >= 3 and shoulders[:, 2].max() >= 0.75
                         and points[7:9, 2].max() >= 0.6)
    if not (both_shoulders or partial_shoulders):
        return None
    anchor = np.average(head[visible_head, :2], axis=0, weights=head[visible_head, 2])
    if not (x1 <= anchor[0] <= x2 and y1 <= anchor[1] <= y1 + 0.72 * height):
        return None
    reliable_shoulders = shoulders[:, 2] >= 0.45
    center = shoulders[reliable_shoulders, :2].mean(axis=0)
    if not (x1 - 0.15 * width <= center[0] <= x2 + 0.15 * width
            and y1 + 0.08 * height <= center[1] <= y2 + 0.1 * height):
        return None
    distance = float(np.linalg.norm(anchor - center))
    if not 0.04 * height <= distance <= 0.7 * max(width, height):
        return None
    span = float(np.linalg.norm(shoulders[0, :2] - shoulders[1, :2]))
    if both_shoulders and not 0.15 * width <= span <= 1.35 * width:
        return None
    return PersonSupport(tuple(map(float, anchor)), max(span, 0.25 * width), float(confidence))


def unique_supports(proposals):
    """One physical head cannot validate two overlapping detection boxes."""
    output = [None] * len(proposals)
    accepted = []
    for index, support in sorted(enumerate(proposals),
                                 key=lambda pair: pair[1].score if pair[1] else -1, reverse=True):
        if support is None:
            continue
        if any(np.linalg.norm(np.subtract(support.anchor, other.anchor))
               < max(4.0, 0.35 * min(support.shoulder_span, other.shoulder_span))
               for other in accepted):
            continue
        output[index] = support
        accepted.append(support)
    return output


class PersonVerifier:
    def __init__(self, device, batch_size, weights):
        self.device = device
        self.batch_size = batch_size
        self.model = YOLO(str(weights), task="pose")

    def verify(self, frame, boxes):
        if not boxes:
            return []
        height, width = frame.shape[:2]
        crops, origins = [], []
        for x1, y1, x2, y2 in boxes:
            # A small border helps clipped shoulders; accepted heads must still
            # belong inside the original detection, not a neighbouring person.
            dx, dy = round(0.08 * (x2 - x1)), round(0.05 * (y2 - y1))
            left, top = max(0, x1 - dx), max(0, y1 - dy)
            crops.append(frame[top:min(height, y2 + dy), left:min(width, x2 + dx)])
            origins.append((left, top))
        results = self.model.predict(crops, imgsz=256, conf=0.4, device=str(self.device),
                                     batch=self.batch_size, rect=False, verbose=False)
        proposals = []
        for result, origin, box in zip(results, origins, boxes):
            candidates = []
            for confidence, points in zip(result.boxes.conf.cpu().numpy(),
                                           result.keypoints.data.cpu().numpy()):
                points = points.copy()
                points[:, :2] += origin
                support = upper_body_support(points, float(confidence), box)
                if support is not None:
                    candidates.append(support)
            candidates.sort(key=lambda support: support.score, reverse=True)
            # Do not attach a neighbouring head when two people are equally
            # plausible inside a crop. A later unambiguous observation can vote.
            proposals.append(candidates[0] if candidates and (
                len(candidates) == 1 or candidates[0].score - candidates[1].score >= 0.08) else None)
        # Tight classroom annotations can clip a shoulder or profile face.
        # Recover missing anatomical context from a full-frame pose instead of
        # lowering the head/shoulder requirements or trusting the detector alone.
        if any(support is None for support in proposals):
            context = self.model.predict(frame, imgsz=960, conf=0.25, device=str(self.device),
                                         verbose=False)[0]
            context_poses = list(zip(context.boxes.conf.cpu().numpy(),
                                     context.keypoints.data.cpu().numpy()))
            for index, (support, box) in enumerate(zip(proposals, boxes)):
                if support is not None:
                    continue
                candidates = [candidate for confidence, points in context_poses
                              if (candidate := upper_body_support(points, float(confidence), box)) is not None]
                candidates.sort(key=lambda candidate: candidate.score, reverse=True)
                if candidates and (len(candidates) == 1
                                   or candidates[0].score - candidates[1].score >= 0.08):
                    proposals[index] = candidates[0]
        return unique_supports(proposals)
