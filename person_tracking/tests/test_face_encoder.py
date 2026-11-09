"""Face evidence must not be shared between nearby or duplicated person boxes."""

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from face_encoder import FaceEncoder, associate_faces, face_quality


def face(x=30, y=20, width=60, height=80):
    return np.asarray([
        x, y, width, height,
        x + 0.3 * width, y + 0.35 * height,
        x + 0.7 * width, y + 0.35 * height,
        x + 0.5 * width, y + 0.55 * height,
        x + 0.35 * width, y + 0.8 * height,
        x + 0.65 * width, y + 0.8 * height,
        0.99,
    ], dtype=np.float32)


class FaceAssociationTests(unittest.TestCase):
    def test_separate_heads_belong_to_separate_people(self):
        pairs = associate_faces([face(), face(x=230)], [(0, 0, 120, 350), (200, 0, 320, 350)])
        self.assertEqual(set(pairs), {(0, 0), (1, 1)})

    def test_duplicate_upper_and_full_body_boxes_do_not_share_face(self):
        pairs = associate_faces([face()], [(0, 0, 120, 200), (0, 0, 120, 350)])
        self.assertEqual(pairs, [])

    def test_two_plausible_heads_inside_one_box_stay_ambiguous(self):
        pairs = associate_faces([face(x=20), face(x=120)], [(0, 0, 200, 350)])
        self.assertEqual(pairs, [])

    def test_lower_body_face_cannot_identify_person_above(self):
        self.assertEqual(associate_faces([face(y=260)], [(0, 0, 120, 350)]), [])

    def test_overlapping_people_still_associate_only_once(self):
        pairs = associate_faces([face(x=50)], [(0, 0, 160, 350), (70, 0, 230, 350)])
        self.assertEqual(pairs, [(0, 0)])

    def test_face_mostly_outside_person_box_is_rejected(self):
        self.assertEqual(associate_faces([face(x=75)], [(0, 0, 100, 350)]), [])

    def test_centered_face_can_fill_a_single_close_up_box(self):
        self.assertEqual(associate_faces([face()], [(25, 10, 95, 107)]), [(0, 0)])
        self.assertEqual(associate_faces([face()], [(25, 0, 95, 105)]), [(0, 0)])
        self.assertEqual(associate_faces([face()], [(10, 10, 90, 107)]), [])

    def test_close_up_with_another_plausible_person_box_stays_ambiguous(self):
        tight = (25, 10, 95, 107)
        self.assertEqual(associate_faces([face()], [tight, (0, 0, 120, 350)]), [])
        self.assertEqual(associate_faces([face()], [tight, tight]), [])

    def test_close_up_box_with_a_second_head_stays_ambiguous(self):
        detected = [face(), face(x=35, y=15, width=40, height=50)]
        self.assertEqual(associate_faces(detected, [(25, 10, 95, 107)]), [])


class FaceQualityTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(4).integers(0, 256, (350, 320, 3), dtype=np.uint8)

    def test_clear_well_formed_face_reaches_identity_quality_gate(self):
        self.assertGreaterEqual(face_quality(self.image, face()), 0.7)

    def test_tiny_face_does_not_create_identity_evidence(self):
        self.assertEqual(face_quality(self.image, face(width=25, height=35)), 0)

    def test_blurred_face_does_not_create_identity_evidence(self):
        self.assertEqual(face_quality(np.full_like(self.image, 128), face()), 0)

    def test_low_contrast_detail_survives_but_blurred_detail_is_rejected(self):
        y, x = np.indices(self.image.shape[:2])
        gray = (125 + ((x // 2 + y // 2) % 2) * 2).astype(np.uint8)
        textured = np.repeat(gray[:, :, None], 3, axis=2)
        self.assertGreaterEqual(face_quality(textured, face()), 0.7)
        blurred = cv2.GaussianBlur(textured, (0, 0), 2)
        self.assertEqual(face_quality(blurred, face()), 0)

    def test_invalid_landmarks_are_rejected(self):
        invalid = face()
        invalid[6:8] = invalid[4:6]
        self.assertEqual(face_quality(self.image, invalid), 0)

    @staticmethod
    def mock_encoder(detections):
        encoder = FaceEncoder.__new__(FaceEncoder)
        encoder.detector = Mock()
        encoder.detector.detect.return_value = (None, detections)
        encoder.recognizer = Mock()
        encoder.recognizer.alignCrop.return_value = np.zeros((112, 112, 3), dtype=np.uint8)
        encoder.recognizer.feature.return_value = np.array([[3., 4.]], dtype=np.float32)
        return encoder

    def test_face_only_box_returns_normalized_feature(self):
        encoder = self.mock_encoder(np.stack([face()]))
        result = encoder.encode(self.image, [(25, 10, 95, 107)])
        np.testing.assert_allclose(result[0][0], [0.6, 0.8])
        self.assertGreaterEqual(result[0][1], 0.7)
        self.assertEqual(encoder.last_statuses, ["얼굴 특징 추출 완료"])

    def test_diagnostic_status_distinguishes_rejections_for_each_person(self):
        invalid = face()
        invalid[6:8] = invalid[4:6]
        cases = (
            (self.image, face(width=25, height=35), "얼굴 크기 부족"),
            (np.full_like(self.image, 128), face(), "얼굴 흐림"),
            (self.image, face(y=-12), "얼굴 일부 잘림"),
            (self.image, invalid, "얼굴 이목구비 위치 불확실"),
            (self.image, face(x=75), "얼굴과 사람 박스 연결 불확실"),
        )
        for frame, detection, expected in cases:
            with self.subTest(status=expected):
                encoder = self.mock_encoder(np.stack([detection]))
                result = encoder.encode(frame, [(0, 0, 100, 350), (200, 0, 320, 350)])
                self.assertEqual(result, [(None, 0.0), (None, 0.0)])
                self.assertEqual(encoder.last_statuses, [expected, "얼굴 미검출"])
                encoder.recognizer.feature.assert_not_called()

    def test_close_up_ambiguity_has_status_for_both_candidate_boxes(self):
        encoder = self.mock_encoder(np.stack([face()]))
        result = encoder.encode(self.image, [(25, 0, 95, 105), (0, 0, 120, 350)])
        self.assertEqual(result, [(None, 0.0), (None, 0.0)])
        self.assertEqual(encoder.last_statuses, ["얼굴과 사람 박스 연결 불확실"] * 2)

    def test_feature_failure_does_not_reuse_previous_status(self):
        encoder = self.mock_encoder(np.stack([face()]))
        box = [(0, 0, 120, 350)]
        encoder.encode(self.image, box)
        self.assertEqual(encoder.last_statuses, ["얼굴 특징 추출 완료"])
        encoder.recognizer.alignCrop.side_effect = cv2.error("invalid face crop")
        self.assertEqual(encoder.encode(self.image, box), [(None, 0.0)])
        self.assertEqual(encoder.last_statuses, ["얼굴 특징 추출 실패"])
        encoder.detector.detect.return_value = (None, None)
        self.assertEqual(encoder.encode(self.image, box), [(None, 0.0)])
        self.assertEqual(encoder.last_statuses, ["얼굴 미검출"])

    def test_detector_runs_once_and_feature_buffers_are_copied(self):
        encoder = FaceEncoder.__new__(FaceEncoder)
        encoder.detector = Mock()
        encoder.detector.detect.return_value = (None, np.stack([face(), face(x=230)]))
        encoder.recognizer = Mock()
        encoder.recognizer.alignCrop.return_value = np.zeros((112, 112, 3), dtype=np.uint8)
        shared = np.zeros((1, 128), dtype=np.float32)
        calls = []

        def feature(_):
            shared.fill(0)
            shared[0, len(calls)] = 1
            calls.append(True)
            return shared

        encoder.recognizer.feature.side_effect = feature
        result = encoder.encode(self.image, np.asarray([(0, 0, 120, 350), (200, 0, 320, 350)]))
        encoder.detector.detect.assert_called_once()
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(float(result[0][0] @ result[1][0]), 0)
        self.assertEqual(int(result[0][0].argmax()), 0)
        self.assertEqual(int(result[1][0].argmax()), 1)

    def test_no_people_skips_detection(self):
        encoder = FaceEncoder.__new__(FaceEncoder)
        encoder.detector = Mock()
        encoder.last_statuses = ["얼굴 특징 추출 완료"]
        self.assertEqual(encoder.encode(self.image, []), [])
        self.assertEqual(encoder.last_statuses, [])
        encoder.detector.detect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
