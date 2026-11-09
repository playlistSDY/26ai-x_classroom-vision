"""Keep separate visible people separate even when clothing features match."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from identity_tracker import IdentityTracker, Observation, TrackerConfig


def unit(*values):
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


class SimilarPeopleTests(unittest.TestCase):
    LEFT = (100, 50, 180, 250)
    RIGHT = (450, 50, 530, 250)

    def setUp(self):
        self.tracker = IdentityTracker(TrackerConfig(
            confirmation_hits=2, new_identity_hits=3,
            new_identity_seconds=0.2, gallery_interval=0.0,
        ))
        self.now = 0.0
        self.a = unit(1, 0, 0, 0)
        self.b = unit(0.9, 0.19 ** 0.5, 0, 0)

    def observation(self, raw_id, feature, *, box=None, quality=1.0,
                    confidence=0.95, allow_enrollment=True, parts=False):
        return Observation(
            raw_id, box or (self.LEFT if raw_id == 1 else self.RIGHT), confidence,
            embedding=None if parts else feature, quality=quality,
            part_embeddings=np.tile(feature, (8, 1)) if parts and feature is not None else None,
            part_visibility=np.ones(8, dtype=np.float32) if parts and feature is not None else None,
            allow_enrollment=allow_enrollment,
        )

    def update(self, observations, step=0.11):
        self.now += step
        assignments = self.tracker.update(observations, self.now)
        ids = [item.person_id for item in assignments if item.person_id is not None]
        self.assertEqual(len(ids), len(set(ids)), "two visible people share one ID")
        return {item.raw_id: item for item in assignments}

    def settle(self, observations, repetitions=8):
        for _ in range(repetitions):
            result = self.update(observations)
        return result

    def enroll_first(self, *, parts=False):
        person_id = self.settle([
            self.observation(1, self.a, parts=parts, quality=0.0 if parts else 1.0),
        ])[1].person_id
        self.assertIsNotNone(person_id)
        return person_id

    def enroll_pair(self, *, parts=False, second=None):
        first_id = self.enroll_first(parts=parts)
        second = self.b if second is None else second
        for _ in range(12):
            result = self.update([
                self.observation(1, self.a, parts=parts, quality=0.0 if parts else 1.0),
                self.observation(2, second, parts=parts, quality=0.0 if parts else 1.0),
            ])
            self.assertEqual(result[1].person_id, first_id)
        second_id = result[2].person_id
        self.assertIsNotNone(second_id)
        self.assertNotEqual(second_id, first_id)
        return first_id, second_id

    def test_similar_newcomer_gets_a_distinct_id_while_owner_is_visible(self):
        first_id, second_id = self.enroll_pair()
        self.assertEqual(len(self.tracker.identities), 2)
        result = self.update([
            self.observation(1, self.a), self.observation(2, self.b),
        ])
        self.assertEqual(result[1].person_id, first_id)
        self.assertEqual(result[2].person_id, second_id)

    def test_newcomer_with_higher_appearance_score_cannot_steal_owner_id(self):
        first_id = self.enroll_first()
        changed_owner = unit(0.82, (1 - 0.82 ** 2) ** 0.5, 0, 0)
        for _ in range(12):
            result = self.update([
                self.observation(1, changed_owner), self.observation(2, self.b),
            ])
            self.assertEqual(result[1].person_id, first_id)
            self.assertNotEqual(result[2].person_id, first_id)
        self.assertIsNotNone(result[2].person_id)
        self.assertEqual(len(self.tracker.identities), 2)

    def test_sampling_gaps_and_small_appearance_changes_keep_both_ids(self):
        first_id, second_id = self.enroll_pair()
        near_a = unit(0.94, (1 - 0.94 ** 2) ** 0.5, 0, 0)
        near_b = unit(0.98, (1 - 0.98 ** 2) ** 0.5, 0, 0)
        for _ in range(12):
            result = self.update([
                self.observation(1, near_a), self.observation(2, near_b),
            ])
            self.assertEqual(result[1].person_id, first_id)
            self.assertEqual(result[2].person_id, second_id)
            for _ in range(4):
                unsampled = self.update([
                    self.observation(1, None), self.observation(2, None),
                ], step=0.03)
                self.assertEqual(unsampled[1].person_id, first_id)
                self.assertEqual(unsampled[2].person_id, second_id)

    def test_identical_descriptors_can_represent_two_separate_visible_people(self):
        first_id, second_id = self.enroll_pair(second=self.a)
        self.assertNotEqual(first_id, second_id)
        self.assertEqual(len(self.tracker.identities), 2)

    def test_visible_owner_without_recent_embeddings_still_excludes_a_new_person(self):
        first_id = self.enroll_first()
        for _ in range(15):
            self.assertEqual(self.update([self.observation(1, None)])[1].person_id, first_id)
        for _ in range(12):
            result = self.update([self.observation(1, None), self.observation(2, self.b)])
            self.assertEqual(result[1].person_id, first_id)
        self.assertIsNotNone(result[2].person_id)
        self.assertNotEqual(result[2].person_id, first_id)
        self.assertEqual(len(self.tracker.identities), 2)

    def test_partial_only_similar_people_receive_separate_ids(self):
        first_id, second_id = self.enroll_pair(parts=True)
        self.assertNotEqual(first_id, second_id)
        self.assertEqual(len(self.tracker.identities), 2)

    def test_identical_parts_keep_two_visible_people_separate(self):
        first_id, second_id = self.enroll_pair(parts=True, second=self.a)
        for _ in range(12):
            result = self.update([
                self.observation(1, self.a, parts=True, quality=0.0),
                self.observation(2, self.a, parts=True, quality=0.0),
            ])
            self.assertEqual(result[1].person_id, first_id)
            self.assertEqual(result[2].person_id, second_id)
        self.assertEqual(len(self.tracker.identities), 2)

    def test_bad_evidence_cannot_use_covisibility_to_enroll(self):
        for options in (
                dict(quality=0.1),
                dict(confidence=0.1),
                dict(allow_enrollment=False)):
            with self.subTest(options=options):
                self.setUp()
                first_id = self.enroll_first()
                for _ in range(20):
                    result = self.update([
                        self.observation(1, self.a),
                        self.observation(2, self.b, **options),
                    ])
                    self.assertEqual(result[1].person_id, first_id)
                    self.assertIsNone(result[2].person_id)
                self.assertEqual(len(self.tracker.identities), 1)

    def test_duplicate_boxes_cannot_use_covisibility_to_enroll(self):
        for duplicate_box in (
                self.LEFT,
                (101, 51, 181, 251),
                (95, 45, 185, 400)):
            with self.subTest(duplicate_box=duplicate_box):
                self.setUp()
                first_id = self.enroll_first()
                for _ in range(20):
                    result = self.update([
                        self.observation(1, self.a),
                        self.observation(2, self.b, box=duplicate_box),
                    ])
                    self.assertEqual(result[1].person_id, first_id)
                    self.assertIsNone(result[2].person_id)
                self.assertEqual(len(self.tracker.identities), 1)

    def test_both_people_can_return_with_distinct_unambiguous_features(self):
        first_id, second_id = self.enroll_pair()
        self.update([], step=30.0)
        result = self.settle([
            self.observation(90, self.b, box=self.LEFT),
            self.observation(91, self.a, box=self.RIGHT),
        ])
        self.assertEqual(result[90].person_id, second_id)
        self.assertEqual(result[91].person_id, first_id)
        self.assertEqual(len(self.tracker.identities), 2)

    def test_crossed_lookalike_scores_do_not_override_compatible_tracks(self):
        first_id, second_id = self.enroll_pair()
        for _ in range(8):
            result = self.update([
                self.observation(1, self.b), self.observation(2, self.a),
            ])
            self.assertEqual(result[1].person_id, first_id)
            self.assertEqual(result[2].person_id, second_id)
            self.update([
                self.observation(1, self.a), self.observation(2, self.b),
            ])

    def test_contradictory_raw_id_swap_still_corrects_labels(self):
        self.b = unit(0, 0, 1, 0)
        first_id, second_id = self.enroll_pair()
        result = self.settle([
            self.observation(1, self.b), self.observation(2, self.a),
        ])
        self.assertEqual(result[1].person_id, second_id)
        self.assertEqual(result[2].person_id, first_id)

    def test_ambiguous_return_after_both_leave_stays_unassigned(self):
        self.enroll_pair()
        self.update([], step=30.0)
        ambiguous = self.a + self.b
        ambiguous /= np.linalg.norm(ambiguous)
        for _ in range(20):
            result = self.update([
                self.observation(90, ambiguous, box=self.RIGHT),
            ])[90]
            self.assertIsNone(result.person_id)
        self.assertEqual(len(self.tracker.identities), 2)

    def test_ambiguous_simultaneous_returns_do_not_invent_new_people(self):
        self.enroll_pair(second=self.a)
        self.update([], step=30.0)
        for _ in range(20):
            result = self.update([
                self.observation(90, self.a, box=self.LEFT),
                self.observation(91, self.a, box=self.RIGHT),
            ])
            self.assertIsNone(result[90].person_id)
            self.assertIsNone(result[91].person_id)
        self.assertEqual(len(self.tracker.identities), 2)


if __name__ == "__main__":
    unittest.main()
