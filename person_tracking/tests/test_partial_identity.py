"""Identity association with visible, co-observed BPBreID body parts."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from identity_tracker import IdentityTracker, Observation, TrackerConfig


class PartialIdentityTests(unittest.TestCase):
    # Released BPBreID ordering: head, arms, torso, legs, feet.
    HEAD, LEFT_ARM, RIGHT_ARM, TORSO, LEFT_LEG, RIGHT_LEG, LEFT_FOOT, RIGHT_FOOT = range(8)

    def setUp(self):
        self.tracker = IdentityTracker(TrackerConfig(
            confirmation_hits=2, new_identity_hits=3,
            new_identity_seconds=0.2, gallery_interval=0.0,
        ))
        self.now = 0.0
        self.a = np.array([1, 0, 0, 0], dtype=np.float32)
        self.b = np.array([0, 1, 0, 0], dtype=np.float32)

    def observation(self, raw_id, feature, visible=None, global_feature=None,
                    quality=0.0, box=None):
        parts = np.tile(feature, (8, 1)).astype(np.float32)
        visibility = np.zeros(8, dtype=np.float32)
        visibility[list(visible if visible is not None else (self.HEAD, self.TORSO))] = 1.0
        return Observation(
            raw_id=raw_id, box=box or (100, 50, 200, 300), confidence=0.95,
            embedding=global_feature, quality=quality,
            part_embeddings=parts, part_visibility=visibility,
        )

    def update(self, observations, step=0.21):
        self.now += step
        assignments = self.tracker.update(observations, self.now)
        ids = [item.person_id for item in assignments if item.person_id is not None]
        self.assertEqual(len(ids), len(set(ids)), "visible people share a global ID")
        return {item.raw_id: item for item in assignments}

    def settle(self, observations, repetitions=8):
        for _ in range(repetitions):
            result = self.update(observations)
        return result

    def enroll(self, raw_id, feature, **kwargs):
        person_id = self.settle([self.observation(raw_id, feature, **kwargs)])[raw_id].person_id
        self.assertIsNotNone(person_id)
        return person_id

    def test_new_upper_body_can_enroll_without_reliable_global_crop(self):
        person_id = self.enroll(1, self.a)
        self.assertEqual(len(self.tracker.identities), 1)
        observed = self.update([self.observation(1, self.a)])[1]
        self.assertEqual(observed.person_id, person_id)

    def test_weak_partial_global_crop_cannot_enroll_when_disallowed(self):
        weak_partial = Observation(1, (100, 50, 200, 300), 0.95,
                                   embedding=self.a, quality=1.0,
                                   allow_enrollment=False)
        for _ in range(12):
            self.assertIsNone(self.update([weak_partial])[1].person_id)
        self.assertEqual(len(self.tracker.identities), 0)

    def test_enrollment_denial_resets_the_novel_evidence_run(self):
        candidate = Observation(1, (100, 50, 200, 300), 0.95,
                                embedding=self.a, quality=1.0)
        for _ in range(2):
            self.assertIsNone(self.update([candidate])[1].person_id)
        candidate.allow_enrollment = False
        self.assertIsNone(self.update([candidate])[1].person_id)
        candidate.allow_enrollment = True
        for _ in range(2):
            self.assertIsNone(self.update([candidate])[1].person_id)
        self.assertIsNotNone(self.update([candidate])[1].person_id)

    def test_enrollment_denial_still_allows_view_learning_and_reentry(self):
        person_id = self.enroll(1, self.a, global_feature=self.a, quality=1.0)
        upper_view = np.array([0.6, 0.8, 0, 0], dtype=np.float32)
        partial = Observation(1, (100, 50, 200, 300), 0.95,
                              embedding=upper_view, quality=1.0,
                              allow_enrollment=False)
        for _ in range(8):
            self.assertEqual(self.update([partial])[1].person_id, person_id)
        self.assertTrue(any(float(upper_view @ view) > 0.99
                            for view in self.tracker.identities[person_id].gallery))
        self.update([], step=30.0)
        partial.raw_id = 99
        self.assertEqual(self.settle([partial])[99].person_id, person_id)
        self.assertEqual(len(self.tracker.identities), 1)

    def test_shared_generic_head_does_not_block_a_new_torso(self):
        person_id = self.enroll(1, self.a)
        for _ in range(12):
            newcomer = self.observation(99, self.b, box=(350, 50, 450, 300))
            newcomer.part_embeddings[self.HEAD] = self.a
            result = self.update([self.observation(1, self.a), newcomer])
        newcomer_id = result[99].person_id
        self.assertIsNotNone(newcomer_id)
        self.assertNotEqual(newcomer_id, person_id)
        self.assertEqual(len(self.tracker.identities), 2)
        self.update([], step=30.0)
        returned = self.observation(90, self.b)
        returned.part_embeddings[self.HEAD] = self.a
        self.assertEqual(self.settle([returned])[90].person_id, newcomer_id)

    def test_invalid_first_parts_do_not_lock_the_session_shape(self):
        for invalid_parts, invalid_visibility in (
                (np.zeros((0, 4)), np.zeros(0)),
                (np.zeros((8, 0)), np.ones(8)),
                (np.zeros((8, 2)), np.ones(8)),
                (np.full((8, 2), np.nan), np.ones(8))):
            with self.subTest(shape=invalid_parts.shape):
                self.setUp()
                invalid = self.observation(1, self.a)
                invalid.part_embeddings = invalid_parts
                invalid.part_visibility = invalid_visibility
                self.assertIsNone(self.update([invalid])[1].person_id)
                self.assertEqual(len(self.tracker.identities), 0)
                self.assertIsNotNone(self.enroll(2, self.a))

    def test_seated_to_full_body_uses_shared_head_and_torso(self):
        person_id = self.enroll(1, self.a, global_feature=self.a, quality=1.0)
        for _ in range(8):
            standing = self.update([
                self.observation(1, self.a, visible=range(8),
                                 global_feature=self.b, quality=1.0),
            ])[1]
            self.assertEqual(standing.person_id, person_id)
        self.assertEqual(len(self.tracker.identities), 1)

    def test_parts_match_does_not_poison_the_global_feature_gallery(self):
        person_id = self.enroll(1, self.a, global_feature=self.a, quality=1.0)
        self.settle([
            self.observation(1, self.a, visible=range(8),
                             global_feature=self.b, quality=1.0),
        ])
        self.update([], step=30.0)
        for _ in range(12):
            other = Observation(99, (100, 50, 200, 300), 0.95,
                                embedding=self.b, quality=1.0)
            result = self.update([other])[99]
            self.assertNotEqual(result.person_id, person_id)
        self.update([], step=30.0)
        original = Observation(90, (100, 50, 200, 300), 0.95,
                               embedding=self.a, quality=1.0)
        self.assertEqual(self.settle([original])[90].person_id, person_id)

    def test_part_view_learning_survives_unsampled_partial_frames(self):
        person_id = self.enroll(1, self.a, global_feature=self.a, quality=1.0)
        new_view = np.array([0.6, 0.8, 0, 0], dtype=np.float32)
        for _ in range(6):
            self.update([
                self.observation(1, new_view, global_feature=self.a, quality=1.0),
            ], step=0.1)
            for _ in range(4):
                unsampled = Observation(1, (100, 50, 200, 300), 0.95, quality=0.0)
                result = self.update([unsampled], step=0.03)[1]
                self.assertEqual(result.person_id, person_id)
        self.update([], step=30.0)
        returned = self.settle([self.observation(99, new_view)])[99]
        self.assertEqual(returned.person_id, person_id)

    def test_full_body_to_upper_body_preserves_label(self):
        person_id = self.enroll(1, self.a, visible=range(8),
                                global_feature=self.a, quality=1.0)
        for _ in range(8):
            seated = self.update([
                self.observation(1, self.a, global_feature=self.b, quality=0.0),
            ])[1]
            self.assertEqual(seated.person_id, person_id)

    def test_partial_reentry_ignores_old_position_and_bad_global_quality(self):
        person_id = self.enroll(1, self.a, visible=range(8),
                                global_feature=self.a, quality=1.0)
        self.update([], step=30.0)
        returned = self.settle([
            self.observation(99, self.a, global_feature=self.b, quality=0.0,
                             box=(1400, 700, 1550, 950)),
        ])[99]
        self.assertEqual(returned.person_id, person_id)
        self.assertEqual(len(self.tracker.identities), 1)

    def test_one_visible_part_cannot_enroll(self):
        for _ in range(20):
            observed = self.update([
                self.observation(1, self.a, visible=(self.HEAD,)),
            ])[1]
            self.assertIsNone(observed.person_id)
        self.assertEqual(len(self.tracker.identities), 0)

    def test_one_visible_part_cannot_reacquire_absent_person(self):
        self.enroll(1, self.a, visible=range(8))
        self.update([], step=30.0)
        for _ in range(20):
            observed = self.update([
                self.observation(99, self.a, visible=(self.HEAD,)),
            ])[99]
            self.assertIsNone(observed.person_id)
        self.assertEqual(len(self.tracker.identities), 1)

    def test_two_legs_do_not_count_as_two_independent_regions(self):
        for _ in range(20):
            observed = self.update([
                self.observation(1, self.a, visible=(self.LEFT_LEG, self.RIGHT_LEG)),
            ])[1]
            self.assertIsNone(observed.person_id)
        self.assertEqual(len(self.tracker.identities), 0)

    def test_disjoint_visible_parts_cannot_match(self):
        person_id = self.enroll(1, self.a)
        self.update([], step=30.0)
        for _ in range(20):
            observed = self.update([
                self.observation(99, self.a, visible=(self.LEFT_LEG, self.LEFT_FOOT)),
            ])[99]
            self.assertNotEqual(observed.person_id, person_id)

    def test_parts_from_separate_gallery_views_cannot_be_combined(self):
        person_id = self.enroll(1, self.a, visible=(self.HEAD, self.LEFT_ARM),
                                global_feature=self.a, quality=1.0)
        self.settle([
            self.observation(1, self.a, visible=(self.TORSO, self.LEFT_LEG),
                             global_feature=self.a, quality=1.0),
        ])
        self.update([], step=30.0)
        for _ in range(12):
            result = self.update([self.observation(99, self.a)])[99]
            self.assertIsNone(result.person_id)
        self.assertEqual(len(self.tracker.identities), 1)

    def test_one_part_can_bridge_current_track_but_not_a_new_track(self):
        person_id = self.enroll(1, self.a, visible=range(8))
        for _ in range(12):
            continued = self.update([
                self.observation(1, self.a, visible=(self.HEAD,)),
            ])[1]
            self.assertEqual(continued.person_id, person_id)
        self.update([], step=30.0)
        for _ in range(12):
            returned = self.update([
                self.observation(99, self.a, visible=(self.HEAD,)),
            ])[99]
            self.assertIsNone(returned.person_id)

    def test_partial_return_requires_the_part_specific_margin(self):
        similar_b = np.array([0.6, 0.8, 0, 0], dtype=np.float32)
        initial = self.settle([
            self.observation(1, self.a), self.observation(2, similar_b),
        ])
        self.assertIsNotNone(initial[1].person_id)
        self.assertIsNotNone(initial[2].person_id)
        self.update([], step=30.0)
        # Cosines are .87 and .80: a .07 margin clears the global .06 gate,
        # but cannot clear the stricter partial-body gate.
        query = np.array([0.87, 0.3475,
                          (1 - 0.87 ** 2 - 0.3475 ** 2) ** 0.5, 0], dtype=np.float32)
        self.assertAlmostEqual(float(query @ self.a - query @ similar_b), 0.07, places=5)
        for _ in range(12):
            result = self.update([self.observation(99, query)])[99]
            self.assertIsNone(result.person_id)
        self.assertEqual(len(self.tracker.identities), 2)

    def test_conflicting_parts_do_not_overwrite_known_people(self):
        initial = self.settle([
            self.observation(1, self.a), self.observation(2, self.b),
        ])
        person_a, person_b = initial[1].person_id, initial[2].person_id
        self.assertIsNotNone(person_a)
        self.assertIsNotNone(person_b)
        self.update([], step=30.0)
        for _ in range(10):
            mixed = self.observation(99, self.a)
            mixed.part_embeddings[self.TORSO] = self.b
            result = self.update([mixed])[99]
            self.assertNotIn(result.person_id, (person_a, person_b))
        self.update([], step=30.0)
        returned = self.settle([
            self.observation(90, self.a), self.observation(91, self.b),
        ])
        self.assertEqual(returned[90].person_id, person_a)
        self.assertEqual(returned[91].person_id, person_b)

    def test_two_partial_returns_receive_distinct_correct_ids(self):
        initial = self.settle([
            self.observation(1, self.a, visible=range(8)),
            self.observation(2, self.b, visible=range(8)),
        ])
        person_a, person_b = initial[1].person_id, initial[2].person_id
        self.assertIsNotNone(person_a)
        self.assertIsNotNone(person_b)
        self.assertNotEqual(person_a, person_b)
        self.update([], step=30.0)
        returned = self.settle([
            self.observation(90, self.b), self.observation(91, self.a),
        ])
        self.assertEqual(returned[90].person_id, person_b)
        self.assertEqual(returned[91].person_id, person_a)

    def test_duplicate_partial_query_cannot_share_one_visible_id(self):
        person_id = self.enroll(1, self.a)
        for _ in range(20):
            result = self.update([
                self.observation(1, self.a), self.observation(99, self.a),
            ])
            assigned = [item.person_id for item in result.values()]
            self.assertLessEqual(assigned.count(person_id), 1)


if __name__ == "__main__":
    unittest.main()
