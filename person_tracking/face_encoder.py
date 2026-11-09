"""Optional face evidence using OpenCV Zoo YuNet and SFace on the CPU.

The detector runs once per frame. Ambiguous face/person associations produce no
face evidence, leaving the caller free to use body appearance and motion.
Models: https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models
API: https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html
"""

from pathlib import Path

import cv2
import numpy as np


FACE_NOT_DETECTED = "얼굴 미검출"
FACE_ASSOCIATION_UNCERTAIN = "얼굴과 사람 박스 연결 불확실"
FACE_FEATURE_READY = "얼굴 특징 추출 완료"
FACE_FEATURE_FAILED = "얼굴 특징 추출 실패"


def _face_person_score(face, person_box):
    """Return the association score and whether the box is a tight close-up."""
    x, y, width, height = map(float, face[:4])
    x1, y1, x2, y2 = map(float, person_box)
    person_width, person_height = x2 - x1, y2 - y1
    if min(width, height, person_width, person_height) <= 0:
        return None
    cx, cy = x + width / 2, y + height / 2
    if not x1 <= cx <= x2 or width > person_width:
        return None
    overlap_width = max(0, min(x + width, x2) - max(x, x1))
    overlap_height = max(0, min(y + height, y2) - max(y, y1))
    containment = overlap_width * overlap_height / (width * height)
    if containment < 0.90:
        return None
    horizontal = abs(cx - (x1 + x2) / 2) / person_width
    close_up = height > 0.65 * person_height
    if close_up:
        if not (width >= 0.75 * person_width
                and 0.72 * person_height <= height <= person_height
                and horizontal <= 0.10
                and abs(cy - (y1 + y2) / 2) <= 0.10 * person_height
                and containment >= 0.95):
            return None
    elif not y1 <= cy <= y1 + 0.55 * person_height:
        return None
    vertical = abs(cy - (y1 + 0.16 * person_height)) / person_height
    return containment - 0.7 * horizontal - 0.3 * vertical, close_up


def associate_faces(faces, person_boxes, margin=0.12):
    """Return (face index, person index), never sharing a face or a person.

    A face belonging equally well to overlapping/nested person boxes is skipped.
    Likewise, a person box containing two equally plausible heads is skipped.
    """
    face_candidates = []
    faces_per_box = [0] * len(person_boxes)
    close_up_boxes = set()
    for face in faces:
        candidates = []
        for person_index, box in enumerate(person_boxes):
            match = _face_person_score(face, box)
            if match is not None:
                score, close_up = match
                candidates.append((score, person_index, close_up))
                faces_per_box[person_index] += 1
                if close_up:
                    close_up_boxes.add(person_index)
        candidates.sort(reverse=True)
        face_candidates.append(candidates)

    proposals = {}
    for face_index, candidates in enumerate(face_candidates):
        if not candidates:
            continue
        # A relaxed close-up link must be unique in both directions. Normal
        # head links still use the existing score margin for overlapping boxes.
        if len(candidates) > 1 and (any(close_up for _, _, close_up in candidates)
                                    or candidates[0][0] - candidates[1][0] < margin):
            continue
        score, person_index, _ = candidates[0]
        if person_index in close_up_boxes and faces_per_box[person_index] > 1:
            continue
        proposals.setdefault(person_index, []).append((score, face_index))
    matches = []
    for person_index, candidates in proposals.items():
        candidates.sort(reverse=True)
        if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < margin:
            continue
        matches.append((candidates[0][1], person_index))
    return matches


def _assess_face_quality(frame, face, min_face_size, min_sharpness):
    if face.shape != (15,) or not np.isfinite(face).all():
        return 0.0, "얼굴 이목구비 위치 불확실"
    if face[14] < 0.9:
        return 0.0, "얼굴 검출 신뢰도 부족"
    x, y, width, height = map(float, face[:4])
    if min(width, height) < min_face_size:
        return 0.0, "얼굴 크기 부족"
    image_height, image_width = frame.shape[:2]
    left, top = max(0, int(np.floor(x))), max(0, int(np.floor(y)))
    right = min(image_width, int(np.ceil(x + width)))
    bottom = min(image_height, int(np.ceil(y + height)))
    if right <= left or bottom <= top:
        return 0.0, "얼굴 일부 잘림"
    if (right - left) * (bottom - top) / (width * height) < 0.95:
        return 0.0, "얼굴 일부 잘림"
    points = face[4:14].reshape(5, 2)
    if (np.any(points[:, 0] < x - 0.05 * width) or np.any(points[:, 0] > x + 1.05 * width)
            or np.any(points[:, 1] < y - 0.05 * height) or np.any(points[:, 1] > y + 1.05 * height)):
        return 0.0, "얼굴 이목구비 위치 불확실"
    eyes, mouth = points[:2], points[3:5]
    eye_distance = float(np.linalg.norm(eyes[0] - eyes[1]))
    if not 0.18 * width <= eye_distance <= 0.8 * width:
        return 0.0, "얼굴 이목구비 위치 불확실"
    if abs(float(eyes[0, 1] - eyes[1, 1])) > 0.35 * height:
        return 0.0, "얼굴 이목구비 위치 불확실"
    eye_y, mouth_y = float(eyes[:, 1].mean()), float(mouth[:, 1].mean())
    if mouth_y - eye_y < 0.12 * height or not eye_y - 0.1 * height <= points[2, 1] <= mouth_y + 0.1 * height:
        return 0.0, "얼굴 이목구비 위치 불확실"
    gray = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    # Absolute Laplacian variance alone rejects sharp, low-contrast faces.
    # Normalize by image contrast as well, while retaining an absolute floor.
    contrast = float(gray.var())
    if sharpness < min_sharpness or sharpness / max(1.0, contrast) < 0.02:
        return 0.0, "얼굴 흐림"
    # Every accepted face reaches the caller's .7 quality threshold. Additional
    # sharpness improves quality without making minimum-size faces unusable.
    clarity = min(1.0, sharpness / (4 * min_sharpness))
    return float(np.clip(face[14] * (0.8 + 0.2 * clarity), 0, 1)), FACE_ASSOCIATION_UNCERTAIN


def face_quality(frame, face, min_face_size=40, min_sharpness=5):
    """Gate small, blurred, clipped or geometrically unreliable face crops."""
    return _assess_face_quality(frame, np.asarray(face, dtype=np.float32), min_face_size, min_sharpness)[0]


def _face_near_head(face, box):
    """Attribute rejected detections for diagnostics without enabling a link."""
    if face.ndim != 1 or face.size < 4 or not np.isfinite(face[:4]).all():
        return False
    x, y, width, height = map(float, face[:4])
    x1, y1, x2, y2 = map(float, box)
    if min(width, height, x2 - x1, y2 - y1) <= 0:
        return False
    cy = y + height / 2
    if not y1 <= cy <= y1 + 0.55 * (y2 - y1) and _face_person_score(face, box) is None:
        return False
    overlap_width = max(0, min(x + width, x2) - max(x, x1))
    overlap_height = max(0, min(y + height, y2) - max(y, y1))
    return overlap_width * overlap_height >= 0.25 * width * height


class FaceEncoder:
    def __init__(self, detector_path: str | Path, recognizer_path: str | Path):
        self.last_statuses: list[str] = []
        self.detector = cv2.FaceDetectorYN.create(
            str(detector_path), "", (320, 320), 0.9, 0.3, 5000,
            cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(
            str(recognizer_path), "", cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU,
        )

    def encode(self, frame: np.ndarray, person_boxes):
        """Return (unit embedding or None, quality) for each visible person box."""
        output = [(None, 0.0) for _ in person_boxes]
        self.last_statuses = [FACE_NOT_DETECTED] * len(person_boxes)
        if len(person_boxes) == 0 or frame.size == 0:
            return output
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("FaceEncoder expects a BGR image with three channels.")
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, detected = self.detector.detect(frame)
        if detected is None:
            return output
        faces, qualities = [], []
        status_priorities = [0] * len(person_boxes)
        for detected_face in detected:
            face = np.asarray(detected_face, dtype=np.float32)
            quality, status = _assess_face_quality(frame, face, min_face_size=40, min_sharpness=5)
            priority = 2 if quality >= 0.7 else 1
            for person_index, box in enumerate(person_boxes):
                if priority > status_priorities[person_index] and _face_near_head(face, box):
                    self.last_statuses[person_index] = status
                    status_priorities[person_index] = priority
            if quality >= 0.7:
                faces.append(face)
                qualities.append(quality)
        for face_index, person_index in associate_faces(faces, person_boxes):
            self.last_statuses[person_index] = FACE_FEATURE_FAILED
            try:
                aligned = self.recognizer.alignCrop(frame, faces[face_index])
                if aligned is None or aligned.size == 0:
                    continue
                # OpenCV may reuse its output buffer; own a copy per face.
                feature = np.asarray(self.recognizer.feature(aligned), dtype=np.float32).reshape(-1).copy()
            except cv2.error:
                continue
            norm = float(np.linalg.norm(feature))
            if feature.size and np.isfinite(feature).all() and np.isfinite(norm) and norm > 1e-8:
                output[person_index] = (feature / norm, qualities[face_index])
                self.last_statuses[person_index] = FACE_FEATURE_READY
        return output
