"""Regression cases for duplicate person boxes and genuine crowd overlap."""

from pathlib import Path
import sys
import unittest

import numpy as np
from ultralytics.engine.results import Boxes

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from person_tracking import deduplicate_people


def detections(*rows):
    data = np.asarray(rows, dtype=np.float32).reshape(-1, 6)
    return Boxes(data, orig_shape=(1080, 1920))


class DuplicatePersonTests(unittest.TestCase):
    def test_seated_person_and_bag_extension_keep_upper_body(self):
        small = (430, 342, 753, 696, 0.70, 0)
        large = (431, 350, 805, 969, 0.39, 0)
        result = deduplicate_people(detections(small, large))
        self.assertIsInstance(result, Boxes)
        self.assertIsInstance(result.data, np.ndarray)
        np.testing.assert_allclose(result.data, [small])

    def test_overlapping_side_by_side_people_are_both_retained(self):
        first = (100, 100, 300, 500, 0.90, 0)
        second = (160, 100, 360, 500, 0.80, 0)
        result = deduplicate_people(detections(first, second))
        self.assertEqual(len(result), 2)

    def test_nested_people_with_different_head_positions_are_retained(self):
        adult = (100, 100, 400, 700, 0.90, 0)
        child = (150, 320, 350, 680, 0.80, 0)
        result = deduplicate_people(detections(adult, child))
        self.assertEqual(len(result), 2)

    def test_disjoint_people_are_both_retained(self):
        left = (100, 100, 300, 500, 0.90, 0)
        right = (500, 100, 700, 500, 0.80, 0)
        result = deduplicate_people(detections(left, right))
        self.assertEqual(len(result), 2)

    def test_nearly_identical_boxes_keep_higher_confidence(self):
        lower = (100, 100, 300, 500, 0.40, 0)
        higher = (105, 105, 305, 505, 0.95, 0)
        original = detections(lower, higher)
        saved = original.data.copy()
        result = deduplicate_people(original)
        np.testing.assert_allclose(result.data, [higher])
        np.testing.assert_array_equal(original.data, saved)

    def test_equal_confidence_duplicate_keeps_first_input(self):
        first = (100, 100, 300, 500, 0.80, 0)
        second = (105, 105, 305, 505, 0.80, 0)
        result = deduplicate_people(detections(first, second))
        np.testing.assert_allclose(result.data, [first])

    def test_empty_detections_preserve_boxes_type(self):
        result = deduplicate_people(detections())
        self.assertIsInstance(result, Boxes)
        self.assertIsInstance(result.data, np.ndarray)
        self.assertEqual(result.data.shape, (0, 6))

    def test_containment_with_incompatible_widths_keeps_both(self):
        narrow = (100, 100, 200, 300, 0.90, 0)
        wide = (50, 100, 250, 500, 0.80, 0)
        result = deduplicate_people(detections(narrow, wide))
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
