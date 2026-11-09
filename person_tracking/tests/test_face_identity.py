"""Qualified faces resolve clothing ambiguity without bypassing evidence gates."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from identity_tracker import IdentityTracker, Observation, TrackerConfig


def unit(*values):
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


class FaceIdentityTests(unittest.TestCase):
    LEFT = (100, 50, 200, 300)
    RIGHT = (450, 50, 550, 300)

    def setUp(self):
        self.tracker = IdentityTracker(TrackerConfig(
            confirmation_hits=2, new_identity_hits=3,
            new_identity_seconds=0.2, gallery_interval=0.0,
        ))
        self.now = 0.0
        self.a = unit(1, 0, 0, 0)
        self.b = unit(0, 1, 0, 0)

    def observation(self, raw_id, body, face, *, box=None, face_quality=1.0,
                    allow_enrollment=True):
        return Observation(
            raw_id=raw_id, box=box or (self.LEFT if raw_id == 1 else self.RIGHT),
            confidence=0.95, embedding=body, quality=1.0 if body is not None else 0.0,
            face_embedding=face, face_quality=face_quality if face is not None else 0.0,
            allow_enrollment=allow_enrollment,
        )

    def update(self, observations, step=0.11):
        self.now += step
        assignments = self.tracker.update(observations, self.now)
        ids = [item.person_id for item in assignments if item.person_id is not None]
        self.assertEqual(len(ids), len(set(ids)), "two visible people share one identity")
        return {item.raw_id: item for item in assignments}

    def settle(self, observations, repetitions=10):
        for _ in range(repetitions):
            result = self.update(observations)
        return result

    def enroll(self, raw_id, body, face):
        person_id = self.settle([self.observation(raw_id, body, face)])[raw_id].person_id
        self.assertIsNotNone(person_id)
        return person_id

    def enroll_pair(self, *, same_body=True):
        person_a = self.enroll(1, self.a, self.a)
        result = self.settle([
            self.observation(1, self.a, self.a),
            self.observation(2, self.a if same_body else self.b, self.b),
        ])
        self.assertEqual(result[1].person_id, person_a)
        person_b = result[2].person_id
        self.assertIsNotNone(person_b)
        self.assertNotEqual(person_b, person_a)
        return person_a, person_b

    def test_identical_clothing_and_distinct_faces_get_distinct_ids(self):
        person_a, person_b = self.enroll_pair()
        self.update([], step=30.0)
        result = self.settle([
            self.observation(90, self.a, self.b, box=self.LEFT),
            self.observation(91, self.a, self.a, box=self.RIGHT),
        ])
        self.assertEqual(result[90].person_id, person_b)
        self.assertEqual(result[91].person_id, person_a)
        self.assertEqual(len(self.tracker.identities), 2)

    def test_different_qualified_face_vetoes_an_identical_body(self):
        person_a = self.enroll(1, self.a, self.a)
        self.update([], step=30.0)
        for _ in range(10):
            result = self.update([self.observation(99, self.a, self.b)])[99]
            self.assertNotEqual(result.person_id, person_a)
        self.assertIsNotNone(result.person_id)
        self.assertEqual(len(self.tracker.identities), 2)

    def test_known_face_recovers_changed_clothing_at_a_different_position(self):
        person_a = self.enroll(1, self.a, self.a)
        self.update([], step=30.0)
        returned = self.settle([
            self.observation(99, self.b, self.a, box=(1400, 700, 1550, 1000)),
        ])[99]
        self.assertEqual(returned.person_id, person_a)
        self.assertEqual(len(self.tracker.identities), 1)

    def test_face_confirmation_survives_unsampled_frames(self):
        person_a = self.enroll(1, self.a, self.a)
        self.update([], step=30.0)
        first = self.update([self.observation(99, self.b, self.a)])[99]
        self.assertIsNone(first.person_id)
        for _ in range(4):
            waiting = self.update([self.observation(99, None, None)], step=0.03)[99]
            self.assertIsNone(waiting.person_id)
        confirmed = self.update([self.observation(99, self.b, self.a)])[99]
        self.assertEqual(confirmed.person_id, person_a)

    def test_missing_face_allows_the_body_fallback(self):
        person_a = self.enroll(1, self.a, self.a)
        self.update([], step=30.0)
        returned = self.settle([self.observation(99, self.a, None)])[99]
        self.assertEqual(returned.person_id, person_a)

    def test_verified_face_teaches_changed_clothes_before_the_face_is_hidden(self):
        person_a = self.enroll(1, self.a, self.a)
        for _ in range(10):
            result = self.update([self.observation(1, self.b, self.a)])
            self.assertEqual(result[1].person_id, person_a)
        self.update([], step=30.0)
        returned = self.settle([self.observation(99, self.b, None)])[99]
        self.assertEqual(returned.person_id, person_a)
        self.assertEqual(len(self.tracker.identities), 1)

    def test_low_quality_face_neither_vetoes_nor_poisons_identity(self):
        person_a = self.enroll(1, self.a, self.a)
        for _ in range(12):
            current = self.update([
                self.observation(1, self.a, self.b, face_quality=0.1),
            ])[1]
            self.assertEqual(current.person_id, person_a)
        self.update([], step=30.0)
        other = self.settle([self.observation(99, self.b, self.b)])[99]
        self.assertIsNotNone(other.person_id)
        self.assertNotEqual(other.person_id, person_a)
        self.update([], step=30.0)
        original = self.settle([self.observation(90, self.b, self.a)])[90]
        self.assertEqual(original.person_id, person_a)

    def test_distinct_faces_override_compatible_body_track_bindings(self):
        person_a, person_b = self.enroll_pair()
        result = self.settle([
            self.observation(1, self.a, self.b),
            self.observation(2, self.a, self.a),
        ])
        self.assertEqual(result[1].person_id, person_b)
        self.assertEqual(result[2].person_id, person_a)

    def test_ambiguous_face_cannot_be_bypassed_by_a_clear_body_match(self):
        self.enroll_pair(same_body=False)
        self.update([], step=30.0)
        ambiguous = unit(1, 1, 0, 0)
        for _ in range(20):
            result = self.update([self.observation(99, self.a, ambiguous)])[99]
            self.assertIsNone(result.person_id)
        self.assertEqual(len(self.tracker.identities), 2)

    def test_uncertain_known_face_does_not_mint_another_identity(self):
        person_a = self.enroll(1, self.a, self.a)
        self.update([], step=30.0)
        uncertain = unit(0.4, (1 - 0.4 ** 2) ** 0.5, 0, 0)
        for _ in range(20):
            result = self.update([self.observation(99, self.a, uncertain)])[99]
            self.assertIsNone(result.person_id)
        self.assertEqual(len(self.tracker.identities), 1)
        confirmed = self.settle([self.observation(99, self.b, self.a)])[99]
        self.assertEqual(confirmed.person_id, person_a)

    def test_known_face_contesting_an_occupied_id_cannot_create_a_third_id(self):
        person_a, person_b = self.enroll_pair()
        for _ in range(20):
            result = self.update([
                self.observation(1, self.a, self.b),
                self.observation(2, self.a, None),
            ])
            self.assertIsNone(result[1].person_id)
            self.assertEqual(result[2].person_id, person_b)
            self.assertEqual(len(self.tracker.identities), 2)
        restored = self.settle([
            self.observation(1, self.a, self.a),
            self.observation(2, self.a, self.b),
        ])
        self.assertEqual(restored[1].person_id, person_a)
        self.assertEqual(restored[2].person_id, person_b)

    def test_duplicate_boxes_with_distinct_faces_do_not_mint_people(self):
        person_a = self.enroll(1, self.a, self.a)
        for _ in range(20):
            result = self.update([
                self.observation(1, self.a, self.a),
                self.observation(2, self.a, self.b, box=(101, 51, 201, 301)),
            ])
            self.assertEqual(result[1].person_id, person_a)
            self.assertIsNone(result[2].person_id)
        self.assertEqual(len(self.tracker.identities), 1)

    def test_qualified_face_can_enroll_without_a_visible_body(self):
        person_a = self.enroll(1, None, self.a)
        self.update([], step=30.0)
        returned = self.settle([self.observation(99, None, self.a)])[99]
        self.assertEqual(returned.person_id, person_a)
        self.assertEqual(len(self.tracker.identities), 1)

    def test_face_evidence_respects_the_enrollment_flag(self):
        for _ in range(20):
            result = self.update([
                self.observation(1, None, self.a, allow_enrollment=False),
            ])[1]
            self.assertIsNone(result.person_id)
        self.assertEqual(len(self.tracker.identities), 0)

    def test_first_lookalike_face_cannot_rename_a_body_only_visible_owner(self):
        person_a = self.enroll(1, self.a, self.a)
        result = self.settle([
            self.observation(1, self.a, self.a),
            self.observation(2, self.b, None),
        ])
        person_b = result[2].person_id
        self.assertIsNotNone(person_b)
        self.assertNotEqual(person_b, person_a)
        for _ in range(10):
            result = self.update([
                self.observation(1, self.a, self.a),
                self.observation(2, self.b, self.a),
            ])
            self.assertEqual(result[1].person_id, person_a)
            self.assertEqual(result[2].person_id, person_b)
            self.assertEqual(len(self.tracker.identities), 2)
        result = self.settle([
            self.observation(1, self.a, self.a),
            self.observation(2, self.b, self.b),
        ])
        self.assertEqual(result[2].person_id, person_b)
        self.update([], step=30.0)
        returned = self.settle([self.observation(99, None, self.b)])[99]
        self.assertEqual(returned.person_id, person_b)

    def test_conflicting_parts_cannot_seed_a_face_into_the_body_owner(self):
        def with_parts(raw_id, body, parts, face):
            observation = self.observation(raw_id, body, face)
            observation.part_embeddings = np.tile(parts, (8, 1))
            observation.part_visibility = np.ones(8, dtype=np.float32)
            return observation

        initial = self.settle([
            with_parts(1, self.a, self.a, None),
            with_parts(2, self.b, self.b, None),
        ])
        person_a, person_b = initial[1].person_id, initial[2].person_id
        self.assertIsNotNone(person_a)
        self.assertIsNotNone(person_b)
        self.assertNotEqual(person_a, person_b)
        for _ in range(10):
            self.update([
                with_parts(1, self.a, self.b, self.b),
                with_parts(2, self.b, self.b, None),
            ])
        self.assertFalse(self.tracker.faces.galleries.get(person_a),
                         "part conflict contaminated the body owner's face gallery")
        self.assertEqual(len(self.tracker.identities), 2)
        clean = self.settle([
            with_parts(1, self.a, self.a, self.a),
            with_parts(2, self.b, self.b, None),
        ])
        self.assertEqual(clean[1].person_id, person_a)
        self.update([], step=30.0)
        returned = self.settle([self.observation(99, None, self.a)])[99]
        self.assertEqual(returned.person_id, person_a)


if __name__ == "__main__":
    unittest.main()
