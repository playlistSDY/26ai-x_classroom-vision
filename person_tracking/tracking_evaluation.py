"""Ground-truth evaluation of raw detections and confirmed session IDs."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Target:
    person_id: int
    box: tuple[float, float, float, float]


def load_ground_truth(path: Path) -> dict[int, list[Target]]:
    """Read a curated, person-only MOT GT file with 1-based frames and xywh."""
    frames: dict[int, list[Target]] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            if not row or not any(value.strip() for value in row):
                continue
            try:
                values = [float(value) for value in row]
                if len(values) != 9 or not all(math.isfinite(value) for value in values):
                    raise ValueError("9개의 유한한 숫자가 필요합니다.")
                frame, person_id, x, y, width, height, mark, category, visibility = values
                if frame < 1 or not frame.is_integer() or person_id < 1 or not person_id.is_integer():
                    raise ValueError("프레임과 ID는 1 이상의 정수여야 합니다.")
                if width <= 0 or height <= 0 or not 0 <= visibility <= 1:
                    raise ValueError("박스 크기는 양수, 가시성은 0~1이어야 합니다.")
                if mark != 1 or category != 1:
                    raise ValueError("평가용 사람 라벨(mark=1, class=1)만 지원합니다. ignore/distractor 전처리는 지원하지 않습니다.")
                targets = frames.setdefault(int(frame), [])
                if any(item.person_id == int(person_id) for item in targets):
                    raise ValueError("같은 프레임에 중복된 사람 ID가 있습니다.")
                targets.append(Target(int(person_id), (x, y, x + width, y + height)))
            except ValueError as error:
                raise ValueError(f"정답 파일 {line}행: {error}") from error
    if not frames:
        raise ValueError("정답 파일에 평가할 사람이 없습니다.")
    return frames


def box_iou(left, right):
    left = np.asarray(left, dtype=np.float64).reshape(-1, 4)
    right = np.asarray(right, dtype=np.float64).reshape(-1, 4)
    overlap = np.maximum(0, np.minimum(left[:, None, 2:], right[None, :, 2:])
                         - np.maximum(left[:, None, :2], right[None, :, :2])).prod(axis=2)
    left_area = np.maximum(0, left[:, 2:] - left[:, :2]).prod(axis=1)
    right_area = np.maximum(0, right[:, 2:] - right[:, :2]).prod(axis=1)
    union = left_area[:, None] + right_area[None, :] - overlap
    return np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)


def match_boxes(similarity, threshold):
    """Maximise valid match count first, IoU second; never match below threshold."""
    valid = similarity >= threshold
    scores = np.where(valid, 1 + similarity / (min(similarity.shape) + 1), 0)
    rows, columns = linear_sum_assignment(-scores)
    keep = valid[rows, columns]
    return rows[keep], columns[keep]


class SequenceEvaluator:
    def __init__(self, ground_truth, iou_threshold=0.5, reid_gap_frames=1):
        if not math.isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
            raise ValueError("IoU 기준은 0보다 크고 1 이하여야 합니다.")
        if reid_gap_frames < 1:
            raise ValueError("재입장 최소 부재 프레임 수는 1 이상이어야 합니다.")
        self.ground_truth = ground_truth
        self.iou_threshold = iou_threshold
        self.reid_gap_frames = reid_gap_frames
        self.frames = []
        self.detection_tp = 0
        self.detection_fp = 0
        self.detection_fn = 0

    def add_frame(self, detection_boxes, tracks: list[Target]):
        frame = len(self.frames) + 1
        targets = self.ground_truth.get(frame, [])
        if len({item.person_id for item in tracks}) != len(tracks):
            raise ValueError("예측 프레임 안에 중복된 세션 ID가 있습니다.")
        similarity = box_iou([item.box for item in targets], detection_boxes)
        rows, _ = match_boxes(similarity, self.iou_threshold)
        self.detection_tp += len(rows)
        self.detection_fp += similarity.shape[1] - len(rows)
        self.detection_fn += len(targets) - len(rows)
        self.frames.append((targets, tracks))

    def result(self):
        from trackeval.metrics import CLEAR, HOTA, Identity

        if not self.frames or max(self.ground_truth) > len(self.frames):
            raise ValueError("영상 프레임 수가 정답 라벨 범위보다 짧거나 영상이 비어 있습니다.")
        gt_map = {value: index for index, value in enumerate(sorted({
            target.person_id for targets, _ in self.frames for target in targets}))}
        tracker_map = {value: index for index, value in enumerate(sorted({
            track.person_id for _, tracks in self.frames for track in tracks}))}
        gt_ids = []
        tracker_ids = []
        similarities = []
        last_seen = {}
        last_matched_id = {}
        reid_events = 0
        reid_correct = 0
        ineligible_returns = 0
        for frame, (targets, tracks) in enumerate(self.frames, 1):
            similarity = box_iou([item.box for item in targets], [item.box for item in tracks])
            gt_ids.append(np.array([gt_map[item.person_id] for item in targets], dtype=np.int64))
            tracker_ids.append(np.array([tracker_map[item.person_id] for item in tracks], dtype=np.int64))
            similarities.append(similarity)
            rows, columns = match_boxes(similarity, self.iou_threshold)
            matches = {targets[row].person_id: tracks[column].person_id for row, column in zip(rows, columns)}
            for target in targets:
                person_id = target.person_id
                if person_id in last_seen and frame - last_seen[person_id] - 1 >= self.reid_gap_frames:
                    # Only past spatial matches establish the expected ID; future
                    # matches never rewrite the reference for this return event.
                    if person_id in last_matched_id:
                        reid_events += 1
                        reid_correct += matches.get(person_id) == last_matched_id[person_id]
                    else:
                        ineligible_returns += 1
                last_seen[person_id] = frame
                if person_id in matches:
                    last_matched_id[person_id] = matches[person_id]

        data: dict[str, str | int | list[np.ndarray]] = {
            "num_timesteps": len(self.frames), "num_gt_ids": len(gt_map),
            "num_tracker_ids": len(tracker_map),
            "num_gt_dets": sum(len(ids) for ids in gt_ids),
            "num_tracker_dets": sum(len(ids) for ids in tracker_ids),
            "gt_ids": gt_ids, "tracker_ids": tracker_ids, "similarity_scores": similarities,
        }
        config = {"THRESHOLD": self.iou_threshold, "PRINT_CONFIG": False}
        clear = CLEAR(config).eval_sequence(data)
        identity = Identity(config).eval_sequence(data)
        hota_metric = HOTA()
        hota = hota_metric.eval_sequence(data)
        hota_scores = np.asarray(hota["HOTA"]).reshape(-1)
        metrics = {
            "precision": self.detection_tp / max(1, self.detection_tp + self.detection_fp),
            "recall": self.detection_tp / max(1, self.detection_tp + self.detection_fn),
            "idf1": float(identity["IDF1"]), "id_switches": int(clear["IDSW"]),
            "hota": float(np.mean(hota_scores)),
            "reid_accuracy": reid_correct / reid_events if reid_events else None,
            "reid_events": reid_events, "reid_correct": reid_correct,
            "reid_ineligible_returns": ineligible_returns,
            "detection_tp": self.detection_tp, "detection_fp": self.detection_fp,
            "detection_fn": self.detection_fn,
        }
        return {
            "metrics": metrics,
            "protocol": {"iou_threshold": self.iou_threshold, "reid_gap_frames": self.reid_gap_frames,
                         "reid_definition": "first-return-frame previous-ID recovery; not retrieval Rank-1",
                         "unassigned_ids": "excluded from tracking predictions; still evaluated as detections",
                         "frames": len(self.frames), "gt_detections": data["num_gt_dets"],
                         "tracking_detections": data["num_tracker_dets"],
                         "reid_unavailable_reason": None if reid_events else "no eligible ground-truth return events"},
            "hota_by_iou": {f"{alpha:.2f}": float(value)
                            for alpha, value in zip(hota_metric.array_labels, hota_scores)},
        }
