"""AIM ablation must skip model loading and leave other identity evidence enabled."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import person_tracking as tracking


class AimAblationTests(unittest.TestCase):
    def test_disabled_aim_is_not_loaded_or_listed_as_an_active_checkpoint(self):
        with (patch.object(tracking, "checkpoint", side_effect=lambda name: name) as checkpoint,
              patch.object(tracking, "YOLO"),
              patch.object(tracking, "BodyEncoder") as body,
              patch.object(tracking, "PartialBodyEncoder") as parts,
              patch.object(tracking, "FaceEncoder") as face):
            tracker = tracking.PersonTracking(torch.device("cpu"), use_aim=False)
        body.assert_not_called()
        self.assertIsNone(tracker.encoder)
        self.assertNotIn("aim-ltcc.pth", tracker.checkpoints)
        self.assertEqual(len(tracker.checkpoints), 4)
        self.assertNotIn("aim-ltcc.pth", [call.args[0] for call in checkpoint.call_args_list])
        parts.assert_called_once()
        face.assert_called_once()

    def test_disabled_aim_processes_body_parts_without_whole_body_features(self):
        with (patch.object(tracking, "checkpoint", side_effect=lambda name: name),
              patch.object(tracking, "YOLO"),
              patch.object(tracking, "BodyEncoder") as body,
              patch.object(tracking, "PartialBodyEncoder"),
              patch.object(tracking, "FaceEncoder"),
              patch.object(tracking, "BOTSORT"),
              patch.object(tracking, "deduplicate_people"),
              patch.object(tracking, "crop_quality", return_value=0.9)):
            tracker = tracking.PersonTracking(torch.device("cpu"), use_aim=False)
            tracker.motion_tracker.update.return_value = np.array([[0, 0, 80, 120, 1, 0.9]])
            tracker.partial_encoder.encode.return_value = [(np.ones((8, 512)), np.ones(8))]
            tracker.face_encoder.encode.return_value = [(None, 0.0)]
            tracker.face_encoder.last_statuses = ["얼굴 미검출"]
            observations, assignments = tracker.process(np.zeros((120, 80, 3), dtype=np.uint8), 0.0)
        body.assert_not_called()
        self.assertEqual(len(assignments), 1)
        self.assertIsNone(observations[0].embedding)
        self.assertIsNotNone(observations[0].part_embeddings)
        self.assertTrue(observations[0].face_sampled)

    def test_default_keeps_baseline_model_and_manifest(self):
        with (patch.object(tracking, "checkpoint", side_effect=lambda name: name),
              patch.object(tracking, "YOLO"),
              patch.object(tracking, "BodyEncoder") as body,
              patch.object(tracking, "PartialBodyEncoder"),
              patch.object(tracking, "FaceEncoder")):
            tracker = tracking.PersonTracking(torch.device("cpu"))
        body.assert_called_once()
        self.assertIs(tracker.encoder, body.return_value)
        self.assertEqual(tracker.checkpoints, tracking.CHECKPOINTS)
