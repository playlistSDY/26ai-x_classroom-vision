"""Check that overlapping person pixels are excluded in crop coordinates."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from person_tracking import occlusion_mask


class OcclusionMaskTests(unittest.TestCase):
    def test_overlap_preserves_clear_head_above_other_person(self):
        mask = occlusion_mask((100, 40, 160, 140), [(130, 90, 200, 170)])
        expected = np.zeros((100, 60), dtype=bool)
        expected[50:, 30:] = True
        np.testing.assert_array_equal(mask, expected)
        self.assertFalse(mask[:50].any())

    def test_disjoint_boxes_do_not_remove_pixels(self):
        mask = occlusion_mask((100, 40, 160, 140), [(0, 0, 20, 20), (160, 40, 190, 140)])
        self.assertEqual(mask.shape, (100, 60))
        self.assertFalse(mask.any())


if __name__ == "__main__":
    unittest.main()
