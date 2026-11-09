"""Regression checks for the evidence shown beside an ID decision."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from identity_tracker import IdentityTracker, Observation


class TrackingDiagnosticsTests(unittest.TestCase):
    def test_new_id_keeps_the_face_veto_and_raw_body_score(self):
        tracker = IdentityTracker(diagnostics=True)
        body = np.array([1, 0], dtype=np.float32)
        different_angle = np.array([0.25, np.sqrt(1 - 0.25 ** 2)], dtype=np.float32)
        box = (100, 50, 260, 340)
        for index in range(8):
            tracker.update([Observation(1, box, 0.95, embedding=body,
                                        face_embedding=body, face_quality=0.9)], index * 0.22)
        for index in range(8, 16):
            result = tracker.update([Observation(1, box, 0.95, embedding=body,
                                                 face_embedding=different_angle, face_quality=0.9)], index * 0.22)[0]
            if result.status == "new":
                break
        self.assertEqual((result.person_id, result.status), (2, "new"))
        self.assertEqual(result.diagnostic.previous_person_id, 1)
        candidate, = result.diagnostic.candidates
        self.assertEqual(candidate.person_id, 1)
        self.assertAlmostEqual(candidate.face, 0.25)
        self.assertAlmostEqual(candidate.body, 1.0)
        self.assertIsNone(candidate.score)
        self.assertIn("얼굴 불일치로 제외", candidate.notes)

    def test_unsuccessful_face_sampling_is_visible_without_invented_scores(self):
        tracker = IdentityTracker(diagnostics=True)
        result = tracker.update([Observation(1, (10, 10, 50, 80), 0.9, face_sampled=True)], 0.0)[0]
        self.assertEqual(result.status, "checking")
        self.assertTrue(result.diagnostic.face_sampled)
        self.assertFalse(result.diagnostic.face_used)
        self.assertEqual(result.diagnostic.candidates, ())

    def test_part_breakdown_uses_the_co_observed_shared_parts(self):
        tracker = IdentityTracker(diagnostics=True)
        embeddings = np.tile(np.array([1, 0], dtype=np.float32), (8, 1))
        visibility = np.array([1, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)
        observation = Observation(1, (10, 10, 50, 80), 0.9,
                                  part_embeddings=embeddings, part_visibility=visibility)
        for index in range(8):
            result = tracker.update([observation], index * 0.22)[0]
        candidate, = result.diagnostic.candidates
        self.assertIsNone(candidate.face)
        self.assertIsNone(candidate.body)
        self.assertAlmostEqual(candidate.parts, 1.0)
        self.assertEqual(candidate.part_details, ((0, 1.0), (3, 1.0)))

        observation.part_visibility = np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)
        result = tracker.update([observation], 8 * 0.22)[0]
        candidate, = result.diagnostic.candidates
        self.assertEqual((result.person_id, result.status), (1, "tracked"))
        self.assertEqual(candidate.part_details, ((3, 1.0),))
        self.assertIn("공통 부위 한 그룹 · 연속 추적용", candidate.notes)

if __name__ == "__main__":
    unittest.main()
