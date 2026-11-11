"""Known-answer cases protect metric definitions and annotation boundaries."""

import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tracking_evaluation import SequenceEvaluator, Target, box_iou, load_ground_truth, match_boxes


BOX = (0.0, 0.0, 10.0, 10.0)
OTHER = (20.0, 0.0, 30.0, 10.0)


class AnnotationTests(unittest.TestCase):
    def load(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gt.txt"
            path.write_text(text, encoding="utf-8")
            return load_ground_truth(path)

    def test_mot_xywh_is_converted_and_absent_frames_are_preserved(self):
        result = self.load("1,12,10,20,30,40,1,1,0.5\n3,12,10,20,30,40,1,1,1\n")
        self.assertEqual(result[1], [Target(12, (10, 20, 40, 60))])
        self.assertNotIn(2, result)

    def test_invalid_or_unsupported_annotations_are_rejected(self):
        invalid = ["", "frame,id,x,y,w,h,mark,class,visibility",
                   "1,1,0,0,10,10,1,1,nan", "0,1,0,0,10,10,1,1,1",
                   "1.5,1,0,0,10,10,1,1,1", "1,1,0,0,-10,10,1,1,1",
                   "1,1,0,0,10,10,0,1,1", "1,1,0,0,10,10,1,2,1",
                   "1,1,0,0,10,10,1,1,1\n1,1,0,0,10,10,1,1,1"]
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.load(text)

    def test_matching_prioritizes_valid_pairs_not_unthresholded_iou(self):
        rows, columns = match_boxes(np.array([[0.9, 0.51], [0.51, 0.49]]), 0.5)
        self.assertEqual(list(zip(rows, columns)), [(0, 1), (1, 0)])

    def test_iou_and_empty_matrices(self):
        np.testing.assert_array_equal(box_iou([BOX, OTHER], [BOX]), [[1], [0]])
        self.assertEqual(box_iou([], [BOX]).shape, (0, 1))
        self.assertEqual(len(match_boxes(box_iou([], []), 0.5)[0]), 0)


@unittest.skipUnless(importlib.util.find_spec("trackeval"), "optional evaluation dependency not installed")
class TrackingMetricTests(unittest.TestCase):
    def test_perfect_tracking_with_return(self):
        evaluator = SequenceEvaluator({1: [Target(42, BOX)], 3: [Target(42, BOX)]})
        evaluator.add_frame([BOX], [Target(900, BOX)])
        evaluator.add_frame([], [])
        evaluator.add_frame([BOX], [Target(900, BOX)])
        metrics = evaluator.result()["metrics"]
        for key in ("precision", "recall", "idf1", "hota", "reid_accuracy"):
            self.assertEqual(metrics[key], 1)
        self.assertEqual(metrics["id_switches"], 0)
        self.assertEqual(metrics["reid_events"], 1)

    def test_switch_reduces_idf1_hota_and_reid_accuracy_not_detection(self):
        evaluator = SequenceEvaluator({1: [Target(1, BOX)], 3: [Target(1, BOX)]})
        evaluator.add_frame([BOX], [Target(7, BOX)])
        evaluator.add_frame([], [])
        evaluator.add_frame([BOX], [Target(8, BOX)])
        metrics = evaluator.result()["metrics"]
        self.assertEqual(metrics["precision"], 1)
        self.assertEqual(metrics["recall"], 1)
        self.assertEqual(metrics["idf1"], 0.5)
        hota = metrics["hota"]
        assert isinstance(hota, float)
        self.assertAlmostEqual(hota, math.sqrt(0.5))
        self.assertEqual(metrics["id_switches"], 1)
        self.assertEqual(metrics["reid_accuracy"], 0)

    def test_false_positives_and_missing_detections(self):
        evaluator = SequenceEvaluator({1: [Target(1, BOX)], 2: [Target(1, BOX)]})
        evaluator.add_frame([BOX, OTHER], [Target(1, BOX), Target(2, OTHER)])
        evaluator.add_frame([], [])
        metrics = evaluator.result()["metrics"]
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["detection_tp"], 1)
        self.assertEqual(metrics["detection_fp"], 1)
        self.assertEqual(metrics["detection_fn"], 1)

    def test_two_people_exchanging_ids_count_as_two_switches(self):
        gt = {frame: [Target(1, BOX), Target(2, OTHER)] for frame in (1, 2)}
        evaluator = SequenceEvaluator(gt)
        evaluator.add_frame([BOX, OTHER], [Target(7, BOX), Target(8, OTHER)])
        evaluator.add_frame([BOX, OTHER], [Target(8, BOX), Target(7, OTHER)])
        metrics = evaluator.result()["metrics"]
        self.assertEqual(metrics["precision"], 1)
        self.assertEqual(metrics["recall"], 1)
        self.assertEqual(metrics["id_switches"], 2)
        self.assertEqual(metrics["idf1"], 0.5)

    def test_checking_counts_for_detection_but_not_identity(self):
        evaluator = SequenceEvaluator({1: [Target(1, BOX)]})
        evaluator.add_frame([BOX], [])
        metrics = evaluator.result()["metrics"]
        self.assertEqual(metrics["precision"], 1)
        self.assertEqual(metrics["recall"], 1)
        self.assertEqual(metrics["idf1"], 0)
        self.assertEqual(metrics["hota"], 0)
        self.assertIsNone(metrics["reid_accuracy"])

    def test_no_predictions_is_zero_not_perfect(self):
        evaluator = SequenceEvaluator({1: [Target(1, BOX)]})
        evaluator.add_frame([], [])
        metrics = evaluator.result()["metrics"]
        for key in ("precision", "recall", "idf1", "hota"):
            self.assertEqual(metrics[key], 0)

    def test_undetected_first_return_is_failure_even_if_later_recovered(self):
        evaluator = SequenceEvaluator({1: [Target(1, BOX)], 3: [Target(1, BOX)], 4: [Target(1, BOX)]})
        evaluator.add_frame([BOX], [Target(7, BOX)])
        evaluator.add_frame([], [])
        evaluator.add_frame([BOX], [])
        evaluator.add_frame([BOX], [Target(7, BOX)])
        metrics = evaluator.result()["metrics"]
        self.assertEqual(metrics["reid_events"], 1)
        self.assertEqual(metrics["reid_accuracy"], 0)

    def test_unobserved_past_id_does_not_fabricate_reid_score(self):
        evaluator = SequenceEvaluator({1: [Target(1, BOX)], 3: [Target(1, BOX)]})
        evaluator.add_frame([], [])
        evaluator.add_frame([], [])
        evaluator.add_frame([BOX], [Target(7, BOX)])
        result = evaluator.result()
        self.assertIsNone(result["metrics"]["reid_accuracy"])
        self.assertEqual(result["metrics"]["reid_ineligible_returns"], 1)
        self.assertIsNotNone(result["protocol"]["reid_unavailable_reason"])

    def test_continuous_gt_missed_by_tracker_is_not_a_return(self):
        evaluator = SequenceEvaluator({frame: [Target(1, BOX)] for frame in range(1, 4)})
        evaluator.add_frame([BOX], [Target(7, BOX)])
        evaluator.add_frame([], [])
        evaluator.add_frame([BOX], [Target(7, BOX)])
        self.assertEqual(evaluator.result()["metrics"]["reid_events"], 0)

    def test_gap_threshold_and_full_ground_truth_coverage(self):
        gt = {1: [Target(1, BOX)], 3: [Target(1, BOX)]}
        evaluator = SequenceEvaluator(gt, reid_gap_frames=2)
        evaluator.add_frame([BOX], [Target(7, BOX)])
        with self.assertRaises(ValueError):
            evaluator.result()
        evaluator.add_frame([], [])
        evaluator.add_frame([BOX], [Target(7, BOX)])
        self.assertIsNone(evaluator.result()["metrics"]["reid_accuracy"])

    def test_duplicate_prediction_id_rejected(self):
        evaluator = SequenceEvaluator({1: [Target(1, BOX)]})
        with self.assertRaises(ValueError):
            evaluator.add_frame([BOX, OTHER], [Target(7, BOX), Target(7, OTHER)])
