"""Regression tests for identity continuity, independent of detector and camera."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from identity_tracker import IdentityTracker, Observation, TrackerConfig


def unit(*values):
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


class IdentityContinuityTests(unittest.TestCase):
    def setUp(self):
        self.tracker = IdentityTracker(TrackerConfig(
            confirmation_hits=2,
            new_identity_hits=3,
            new_identity_seconds=0.2,
            gallery_interval=0.0,
        ))
        self.now = 0.0
        self.a = unit(1, 0, 0, 0)
        self.b = unit(0, 0, 1, 0)

    def observation(self, raw_id, embedding, quality=1.0, box=None):
        if box is None:
            box = (raw_id * 100, 10, raw_id * 100 + 70, 180)
        return Observation(raw_id=raw_id, box=box, confidence=0.95,
                           embedding=embedding, quality=quality)

    def update(self, observations, step=0.11):
        self.now += step
        assignments = self.tracker.update(observations, self.now)
        by_raw = {assignment.raw_id: assignment for assignment in assignments}
        self.assertEqual(len(by_raw), len(assignments))
        ids = [assignment.person_id for assignment in assignments
               if assignment.person_id is not None]
        self.assertEqual(len(ids), len(set(ids)), "two visible people share an ID")
        return by_raw

    def settle(self, observations, repetitions=6):
        result = None
        for _ in range(repetitions):
            result = self.update(observations)
        return result

    def enroll(self, raw_id, feature):
        person_id = self.settle([self.observation(raw_id, feature)])[raw_id].person_id
        self.assertIsNotNone(person_id)
        return person_id

    def test_identity_survives_arbitrarily_long_absence(self):
        person_id = self.enroll(1, self.a)
        self.update([], step=86_400.0)
        returned = self.settle([self.observation(92, self.a)])[92]
        self.assertEqual(returned.person_id, person_id)

    def test_return_is_independent_of_old_position(self):
        person_id = self.enroll(1, self.a)
        self.update([], step=4.0)
        returned = self.settle([
            self.observation(10, self.a, box=(1500, 700, 1700, 1050)),
        ])[10]
        self.assertEqual(returned.person_id, person_id)

    def test_learned_side_view_is_available_after_return(self):
        person_id = self.enroll(1, self.a)
        intermediate = unit(0.85, 0.5268, 0, 0)
        side = unit(0.55, 0.8352, 0, 0)
        self.assertLess(float(self.a @ side), 0.68)
        self.assertGreater(float(intermediate @ side), 0.68)
        self.assertEqual(self.settle([self.observation(1, intermediate)])[1].person_id,
                         person_id)
        self.assertEqual(self.settle([self.observation(1, side)])[1].person_id,
                         person_id)
        self.update([], step=20.0)
        returned = self.settle([self.observation(18, side)])[18]
        self.assertEqual(returned.person_id, person_id)

    def test_continuous_side_view_below_match_threshold_is_learned(self):
        person_id = self.enroll(1, self.a)
        side = unit(0.6, 0.8, 0, 0)
        self.assertLess(float(self.a @ side), self.tracker.config.match_threshold)
        self.assertGreater(float(self.a @ side),
                           self.tracker.config.continuity_min_similarity)
        for _ in range(25):
            observed = self.update([self.observation(1, side)], step=0.2)[1]
            self.assertEqual(observed.person_id, person_id)
        self.update([], step=30.0)
        returned = self.settle([
            self.observation(80, side, box=(1400, 700, 1550, 1100)),
        ])[80]
        self.assertEqual(returned.person_id, person_id)
        self.assertEqual(len(self.tracker.identities), 1)

    def test_one_frame_novel_view_is_not_added_to_identity(self):
        person_id = self.enroll(1, self.a)
        outlier = unit(0.6, 0.8, 0, 0)
        observed = self.update([self.observation(1, outlier)])[1]
        self.assertEqual(observed.person_id, person_id)
        self.settle([self.observation(1, self.a)])
        gallery = self.tracker.identities[person_id].gallery
        self.assertLess(max(float(feature @ outlier) for feature in gallery),
                        self.tracker.config.match_threshold)
        self.update([], step=30.0)
        returned = self.settle([self.observation(80, self.a)])[80]
        self.assertEqual(returned.person_id, person_id)

    def test_side_view_confirmation_survives_unsampled_frames(self):
        person_id = self.enroll(1, self.a)
        side = unit(0.6, 0.8, 0, 0)
        for _ in range(12):
            self.update([self.observation(1, side)], step=0.1)
            for _ in range(4):
                observed = self.update([self.observation(1, None)], step=0.03)[1]
                self.assertEqual(observed.person_id, person_id)
        self.update([], step=30.0)
        returned = self.settle([self.observation(80, side)])[80]
        self.assertEqual(returned.person_id, person_id)

    def test_competing_identity_blocks_novel_view_learning(self):
        competitor = unit(0, 0.725, (1 - 0.725 ** 2) ** 0.5, 0)
        initial = self.settle([
            self.observation(1, self.a), self.observation(2, competitor),
        ])
        person_id = initial[1].person_id
        self.assertIsNotNone(person_id)
        self.assertIsNotNone(initial[2].person_id)
        side = unit(0.6, 0.8, 0, 0)
        self.assertLess(float(side @ self.a - side @ competitor),
                        self.tracker.config.ambiguity_margin)
        for _ in range(25):
            self.update([
                self.observation(1, side), self.observation(2, competitor),
            ], step=0.2)
        gallery = self.tracker.identities[person_id].gallery
        self.assertLess(max(float(feature @ side) for feature in gallery),
                        self.tracker.config.match_threshold)

    def test_inconsistent_novel_views_do_not_poison_identity(self):
        person_id = self.enroll(1, self.a)
        side_a = unit(0.6, 0.8, 0, 0)
        side_b = unit(0.6, -0.8, 0, 0)
        for _ in range(12):
            self.update([self.observation(1, side_a)], step=0.2)
            self.update([self.observation(1, side_b)], step=0.2)
        gallery = self.tracker.identities[person_id].gallery
        for side in (side_a, side_b):
            self.assertLess(max(float(feature @ side) for feature in gallery),
                            self.tracker.config.match_threshold)

    def test_raw_track_jump_cannot_teach_a_weak_view(self):
        person_id = self.enroll(1, self.a)
        side = unit(0.6, 0.8, 0, 0)
        for _ in range(25):
            jumped = self.update([
                self.observation(1, side, box=(1400, 700, 1470, 870)),
            ], step=0.2)[1]
            self.assertEqual(jumped.person_id, person_id)
        gallery = self.tracker.identities[person_id].gallery
        self.assertLess(max(float(feature @ side) for feature in gallery),
                        self.tracker.config.match_threshold)
        confirmed = self.update([
            self.observation(1, self.a, box=(1400, 700, 1470, 870)),
        ])[1]
        self.assertEqual(confirmed.person_id, person_id)
        for _ in range(6):
            self.update([
                self.observation(1, side, box=(1400, 700, 1470, 870)),
            ], step=0.2)
        self.update([], step=30.0)
        returned = self.settle([self.observation(80, side)])[80]
        self.assertEqual(returned.person_id, person_id)

    def test_occluded_unsampled_frame_breaks_novel_view_confirmation(self):
        person_id = self.enroll(1, self.a)
        side = unit(0.6, 0.8, 0, 0)
        for _ in range(2):
            self.update([self.observation(1, side)], step=0.2)
        self.update([self.observation(1, None, quality=0.1)], step=0.1)
        for _ in range(2):
            self.update([self.observation(1, side)], step=0.2)
        gallery = self.tracker.identities[person_id].gallery
        self.assertLess(max(float(feature @ side) for feature in gallery),
                        self.tracker.config.match_threshold)
        self.update([self.observation(1, side)], step=0.2)
        self.assertGreater(max(float(feature @ side) for feature in gallery),
                           self.tracker.config.match_threshold)

    def test_full_gallery_preserves_rare_old_view(self):
        self.tracker.config.gallery_size = 3
        person_id = self.enroll(1, self.a)
        rare_side = unit(0.6, 0.8, 0, 0)
        for _ in range(6):
            self.update([self.observation(1, rare_side)], step=0.2)
        for common in (unit(0.8, 0, 0.6, 0), unit(0.8, 0, 0, 0.6)):
            self.settle([self.observation(1, common)])
        gallery = self.tracker.identities[person_id].gallery
        self.assertEqual(len(gallery), 3)
        self.assertGreater(max(float(feature @ rare_side) for feature in gallery), 0.98)
        self.update([], step=30.0)
        returned = self.settle([self.observation(80, rare_side)])[80]
        self.assertEqual(returned.person_id, person_id)

    def test_marginal_quality_cannot_teach_a_weak_view(self):
        person_id = self.enroll(1, self.a)
        side = unit(0.6, 0.8, 0, 0)
        for _ in range(25):
            observed = self.update([
                self.observation(1, side, quality=0.5),
            ], step=0.2)[1]
            self.assertEqual(observed.person_id, person_id)
        gallery = self.tracker.identities[person_id].gallery
        self.assertLess(max(float(feature @ side) for feature in gallery),
                        self.tracker.config.match_threshold)

    def test_absence_resets_unconfirmed_novel_view(self):
        person_id = self.enroll(1, self.a)
        side = unit(0.6, 0.8, 0, 0)
        self.update([self.observation(1, side)])
        self.update([], step=2.0)
        for _ in range(2):
            self.update([self.observation(1, side)])
        gallery = self.tracker.identities[person_id].gallery
        self.assertLess(max(float(feature @ side) for feature in gallery),
                        self.tracker.config.match_threshold)
        self.update([], step=30.0)
        returned = self.settle([self.observation(90, self.a)])[90]
        self.assertEqual(returned.person_id, person_id)

    def test_enrollment_preserves_earlier_linked_views(self):
        intermediate = unit(0.85, 0.5268, 0, 0)
        side = unit(0.55, 0.8352, 0, 0)
        self.update([self.observation(1, self.a)])
        self.update([self.observation(1, intermediate)])
        enrolled = self.update([self.observation(1, side)])[1]
        self.assertIsNotNone(enrolled.person_id)
        self.update([], step=30.0)
        returned = self.settle([self.observation(40, self.a)])[40]
        self.assertEqual(returned.person_id, enrolled.person_id)

    def test_two_people_keep_distinct_ids_after_return_together(self):
        initial = self.settle([
            self.observation(1, self.a), self.observation(2, self.b),
        ])
        expected = {1: initial[1].person_id, 2: initial[2].person_id}
        self.assertIsNotNone(expected[1])
        self.assertIsNotNone(expected[2])
        self.assertNotEqual(expected[1], expected[2])
        self.update([], step=100.0)
        returned = self.settle([
            self.observation(90, self.b), self.observation(91, self.a),
        ])
        self.assertEqual(returned[90].person_id, expected[2])
        self.assertEqual(returned[91].person_id, expected[1])

    def test_raw_tracker_id_swap_is_corrected_by_appearance(self):
        initial = self.settle([
            self.observation(1, self.a), self.observation(2, self.b),
        ])
        expected_a, expected_b = initial[1].person_id, initial[2].person_id
        self.assertIsNotNone(expected_a)
        self.assertIsNotNone(expected_b)
        corrected = self.settle([
            self.observation(1, self.b), self.observation(2, self.a),
        ])
        self.assertEqual(corrected[1].person_id, expected_b)
        self.assertEqual(corrected[2].person_id, expected_a)

    def test_raw_id_swap_confirms_across_frames_without_embeddings(self):
        initial = self.settle([
            self.observation(1, self.a), self.observation(2, self.b),
        ])
        expected_a, expected_b = initial[1].person_id, initial[2].person_id
        first = self.update([
            self.observation(1, self.b), self.observation(2, self.a),
        ])
        self.assertIsNone(first[1].person_id)
        self.assertIsNone(first[2].person_id)
        for _ in range(4):
            waiting = self.update([
                self.observation(1, None), self.observation(2, None),
            ], step=0.03)
            self.assertIsNone(waiting[1].person_id)
            self.assertIsNone(waiting[2].person_id)
        corrected = self.update([
            self.observation(1, self.b), self.observation(2, self.a),
        ])
        self.assertEqual(corrected[1].person_id, expected_b)
        self.assertEqual(corrected[2].person_id, expected_a)

    def test_new_raw_return_confirms_across_frames_without_embeddings(self):
        person_id = self.enroll(1, self.a)
        self.update([], step=30.0)
        self.assertIsNone(self.update([self.observation(99, self.a)])[99].person_id)
        for _ in range(4):
            self.assertIsNone(self.update([self.observation(99, None)], step=0.03)[99].person_id)
        self.assertEqual(self.update([self.observation(99, self.a)])[99].person_id,
                         person_id)

    def test_known_evidence_interrupts_new_identity_enrollment(self):
        person_id = self.enroll(1, self.a)
        self.update([], step=30.0)
        side = unit(0.6, 0.8, 0, 0)
        for _ in range(2):
            self.assertIsNone(self.update([self.observation(99, side)])[99].person_id)
        self.assertIsNone(self.update([self.observation(99, self.a)])[99].person_id)
        self.assertIsNone(self.update([self.observation(99, side)])[99].person_id)
        confirmed = self.update([self.observation(99, self.a)])[99]
        self.assertEqual(confirmed.person_id, person_id)
        self.assertEqual(len(self.tracker.identities), 1)

    def test_return_confirmation_tolerates_intermediate_weak_view(self):
        person_id = self.enroll(1, self.a)
        self.update([], step=30.0)
        side = unit(0.6, 0.8, 0, 0)
        self.assertIsNone(self.update([self.observation(99, self.a)])[99].person_id)
        self.assertIsNone(self.update([self.observation(99, side)])[99].person_id)
        returned = self.update([self.observation(99, self.a)])[99]
        self.assertEqual(returned.person_id, person_id)

    def test_weak_views_cannot_supply_return_confirmation_votes(self):
        person_id = self.enroll(1, self.a)
        self.update([], step=30.0)
        side = unit(0.6, 0.8, 0, 0)
        self.assertIsNone(self.update([self.observation(99, self.a)])[99].person_id)
        for _ in range(20):
            returned = self.update([self.observation(99, side)], step=0.2)[99]
            self.assertNotEqual(returned.person_id, person_id)

    def test_competing_strong_evidence_resets_return_confirmation(self):
        initial = self.settle([
            self.observation(1, self.a), self.observation(2, self.b),
        ])
        person_id = initial[1].person_id
        self.update([], step=30.0)
        for _ in range(4):
            self.assertIsNone(self.update([self.observation(99, self.a)])[99].person_id)
            self.assertIsNone(self.update([self.observation(99, self.b)])[99].person_id)
        self.assertIsNone(self.update([self.observation(99, self.a)])[99].person_id)
        returned = self.update([self.observation(99, self.a)])[99]
        self.assertEqual(returned.person_id, person_id)

    def test_same_raw_return_without_features_waits_for_confirmation(self):
        person_id = self.enroll(1, self.a)
        self.update([], step=2.0)
        for _ in range(4):
            self.assertIsNone(self.update([self.observation(1, None)], step=0.03)[1].person_id)
        self.assertIsNone(self.update([self.observation(1, self.a)])[1].person_id)
        self.assertEqual(self.update([self.observation(1, self.a)])[1].person_id,
                         person_id)

    def test_new_raw_track_cannot_duplicate_a_visible_identity(self):
        person_id = self.enroll(1, self.a)
        for _ in range(12):
            result = self.update([
                self.observation(1, self.a), self.observation(2, self.a),
            ])
            assigned = [x.person_id for x in result.values() if x.person_id is not None]
            self.assertLessEqual(assigned.count(person_id), 1)

    def test_ambiguous_return_stays_unassigned(self):
        similar_b = unit(0.6, 0.8, 0, 0)
        initial = self.settle([
            self.observation(1, self.a), self.observation(2, similar_b),
        ])
        self.assertIsNotNone(initial[1].person_id)
        self.assertIsNotNone(initial[2].person_id)
        self.update([], step=10.0)
        ambiguous = unit(0.8944272, 0.4472136, 0, 0)
        self.assertAlmostEqual(float(ambiguous @ self.a),
                               float(ambiguous @ similar_b), places=5)
        for _ in range(20):
            result = self.update([self.observation(30, ambiguous)])
            self.assertIsNone(result[30].person_id)

    def test_low_quality_does_not_mint_repeated_identities(self):
        person_id = self.enroll(1, self.a)
        self.update([], step=3.0)
        for index in range(25):
            result = self.update([
                self.observation(100 + index, self.a, quality=0.1),
            ])
            self.assertIsNone(result[100 + index].person_id)
        returned = self.settle([self.observation(500, self.a)])[500]
        self.assertEqual(returned.person_id, person_id)

    def test_low_quality_features_do_not_poison_the_gallery(self):
        person_id = self.enroll(1, self.a)
        for _ in range(15):
            self.update([self.observation(1, self.b, quality=0.1)])
        self.update([], step=3.0)
        newcomer = self.settle([self.observation(77, self.b)])[77].person_id
        self.assertIsNotNone(newcomer)
        self.assertNotEqual(newcomer, person_id)
        self.update([], step=3.0)
        self.assertEqual(self.settle([self.observation(88, self.a)])[88].person_id,
                         person_id)

    def test_reused_raw_id_with_different_person_does_not_reuse_label(self):
        first_id = self.enroll(1, self.a)
        self.update([], step=30.0)
        second_id = self.settle([self.observation(1, self.b)])[1].person_id
        self.assertIsNotNone(second_id)
        self.assertNotEqual(second_id, first_id)
        self.update([], step=2.0)
        self.assertEqual(self.settle([self.observation(2, self.a)])[2].person_id,
                         first_id)


if __name__ == "__main__":
    unittest.main()
