"""Exercise the optional tracking boundary against an actual MLflow store."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_tracking import TrackingExperiment
from identity_tracker import TrackerConfig


@unittest.skipUnless(importlib.util.find_spec("mlflow"), "optional MLflow dependency not installed")
class ExperimentTrackingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.uri = f"sqlite:///{self.root / 'tracking.db'}"
        from mlflow import MlflowClient
        self.client = MlflowClient(tracking_uri=self.uri)
        self.client.create_experiment("test", artifact_location=(self.root / "artifacts").as_uri())

    def test_metrics_manifest_weights_and_reset(self):
        weights = self.root / "sample.pt"
        weights.write_bytes(b"test-checkpoint")
        with TrackingExperiment(self.uri, "test", "smoke") as experiment:
            experiment.configure({"device": "cpu"}, TrackerConfig(),
                                 {weights.name: ("https://example.com/model", "test-hash")},
                                 self.root, upload_models=True)
            observations = [SimpleNamespace(face_sampled=True, face_embedding=object()),
                            SimpleNamespace(face_sampled=False, face_embedding=None)]
            assignments = [SimpleNamespace(person_id=1, status="reidentified"),
                           SimpleNamespace(person_id=None, status="checking")]
            experiment.record(observations, assignments, 1, 20)
            experiment.flush()
            experiment.reset_session()
            experiment.record([], [], 0, 10)
        run = self.client.get_run(experiment.run_id)
        self.assertEqual(run.info.status, "FINISHED")
        self.assertEqual(run.data.params["identity.face_min_quality"], "0.7")
        self.assertEqual(run.data.metrics["frames_processed"], 2)
        self.assertEqual(run.data.metrics["mean_process_ms"], 15)
        self.assertEqual(run.data.metrics["mean_visible_people"], 1)
        self.assertEqual(run.data.metrics["checking_ratio"], 0.5)
        self.assertEqual(run.data.metrics["face_feature_ratio"], 1)
        self.assertEqual(run.data.metrics["session_resets"], 1)
        self.assertEqual(run.data.metrics["session_identities"], 0)
        self.assertEqual([item.step for item in self.client.get_metric_history(
            experiment.run_id, "frames_processed")], [1, 2])
        downloaded = self.client.download_artifacts(experiment.run_id, "models/weights/sample.pt")
        self.assertEqual(Path(downloaded).read_bytes(), b"test-checkpoint")
        manifest = self.client.download_artifacts(experiment.run_id, "models/manifest.json")
        self.assertIn("test-hash", Path(manifest).read_text())

    def test_failure_is_marked_failed(self):
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            with TrackingExperiment(self.uri, "test") as experiment:
                raise RuntimeError("inference failed")
        self.assertEqual(self.client.get_run(experiment.run_id).info.status, "FAILED")

    def test_interrupt_is_marked_killed(self):
        with self.assertRaises(KeyboardInterrupt):
            with TrackingExperiment(self.uri, "test") as experiment:
                raise KeyboardInterrupt
        self.assertEqual(self.client.get_run(experiment.run_id).info.status, "KILLED")
