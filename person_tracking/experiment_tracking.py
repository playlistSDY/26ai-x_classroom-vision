"""Opt-in MLflow logging of aggregate tracking metrics, never identity features."""

from __future__ import annotations

import os
import platform
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path


class TrackingExperiment:
    def __init__(self, tracking_uri=None, experiment_name=None, run_name=None):
        # Keep MLflow optional for ordinary camera use.
        from mlflow import MlflowClient

        default_uri = "sqlite:///" + str(Path(__file__).resolve().parents[1] / "mlflow.db")
        self.client = MlflowClient(tracking_uri=tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or default_uri)
        name = experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME") or "classroom-vision"
        experiment = self.client.get_experiment_by_name(name)
        experiment_id = experiment.experiment_id if experiment else self.client.create_experiment(name)
        self.run_id = self.client.create_run(
            experiment_id, tags={"mlflow.runName": run_name or "camera-tracking",
                                 "task": "person-tracking", "data_policy": "aggregate-only"},
        ).info.run_id
        self.frames = 0
        self.resets = 0
        self.total_ms = 0.0
        self.people = 0
        self.checking = 0
        self.reidentified = 0
        self.face_samples = 0
        self.face_features = 0
        self.session_identities = 0
        self.window_frames = 0
        self.window_ms = 0.0
        self.pipeline_started = None
        self.pipeline_completed = None
        self.pipeline_frames = 0
        self.last_logged = time.monotonic()
        print(f"MLflow run: {self.run_id}", flush=True)

    def __enter__(self):
        return self

    def configure(self, params, config, checkpoints, cache, upload_models=False):
        from mlflow.entities import Param

        values = dict(params)
        values.update({f"identity.{key}": value for key, value in asdict(config).items()})
        self.client.log_batch(self.run_id, params=[Param(key, str(value)) for key, value in values.items()])
        # checkpoint() verified these hashes before this method is called.
        manifest = {name: {"source": url, "sha256": digest} for name, (url, digest) in checkpoints.items()}
        self.client.log_dict(self.run_id, manifest, "models/manifest.json")
        self.client.log_dict(self.run_id, {
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: version(name) for name in ("mlflow", "torch", "ultralytics", "opencv-python")},
        }, "environment.json")
        if upload_models:
            for name in checkpoints:
                self.client.log_artifact(self.run_id, str(cache / name), artifact_path="models/weights")

    def record(self, observations, assignments, identity_count, elapsed_ms):
        self.frames += 1
        self.window_frames += 1
        self.total_ms += elapsed_ms
        self.window_ms += elapsed_ms
        self.people += len(observations)
        self.checking += sum(item.person_id is None for item in assignments)
        self.reidentified += sum(item.status == "reidentified" for item in assignments)
        self.face_samples += sum(item.face_sampled for item in observations)
        self.face_features += sum(item.face_sampled and item.face_embedding is not None for item in observations)
        self.session_identities = identity_count
        if time.monotonic() - self.last_logged >= 5:
            self.flush()

    def reset_session(self):
        self.resets += 1
        self.session_identities = 0

    def start_pipeline(self, timestamp=None):
        """Start wall-clock timing after model loading, before input acquisition."""
        self.pipeline_started = time.monotonic() if timestamp is None else timestamp

    def complete_pipeline_frame(self, timestamp=None):
        """Called after decoding/capture, inference, logging and presentation."""
        self.pipeline_completed = time.monotonic() if timestamp is None else timestamp
        self.pipeline_frames += 1

    def log_evaluation(self, report):
        from mlflow.entities import Metric

        timestamp = int(time.time() * 1000)
        self.client.log_batch(self.run_id, metrics=[
            Metric(key, float(value), timestamp, self.frames)
            for key, value in report["metrics"].items() if value is not None
        ])
        self.client.log_dict(self.run_id, report, "evaluation/summary.json")

    def flush(self):
        from mlflow.entities import Metric

        metrics = {
            "frames_processed": self.frames,
            "session_resets": self.resets,
            "session_identities": self.session_identities,
            "mean_visible_people": self.people / self.frames if self.frames else 0,
            "checking_ratio": self.checking / self.people if self.people else 0,
            "reidentified_observations": self.reidentified,
            "face_samples": self.face_samples,
            "face_feature_ratio": self.face_features / self.face_samples if self.face_samples else 0,
            "mean_process_ms": self.total_ms / self.frames if self.frames else 0,
        }
        if self.window_frames:
            metrics["window_process_ms"] = self.window_ms / self.window_frames
        if self.pipeline_started is not None and self.pipeline_completed is not None:
            elapsed = self.pipeline_completed - self.pipeline_started
            if elapsed > 0:
                metrics["actual_fps"] = self.pipeline_frames / elapsed
                metrics["pipeline_frames"] = self.pipeline_frames
                metrics["pipeline_seconds"] = elapsed
        timestamp = int(time.time() * 1000)
        self.client.log_batch(self.run_id, metrics=[
            Metric(key, float(value), timestamp, self.frames) for key, value in metrics.items()
        ])
        self.window_frames = 0
        self.window_ms = 0.0
        self.last_logged = time.monotonic()

    def __exit__(self, exc_type, exc_value, traceback):
        status = "FINISHED" if exc_type is None else "KILLED" if exc_type is KeyboardInterrupt else "FAILED"
        try:
            self.flush()
        finally:
            self.client.set_terminated(self.run_id, status=status)
        return False
