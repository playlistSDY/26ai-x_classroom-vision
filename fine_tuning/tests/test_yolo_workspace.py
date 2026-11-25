"""Validate data preflight and train/validation API wiring without GPU or weights."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import yaml

spec = importlib.util.spec_from_file_location("yolo_run", Path(__file__).parents[1] / "yolo/run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        for split in ("train", "val"):
            images = self.root / "data/images" / split
            labels = self.root / "data/labels" / split
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            (images / "sample.jpg").touch()
            (labels / "sample.txt").write_text("0 0.5 0.5 0.2 0.4\n")
        self.data = {"path": str(self.root / "data"), "train": "images/train",
                     "val": "images/val", "names": {0: "person"}}
        (self.root / "data.yaml").write_text(yaml.safe_dump(self.data))
        (self.root / "config.yaml").write_text(yaml.safe_dump({"data": "data.yaml", "epochs": 1,
                                                               "batch": 1, "project": "runs/yolo"}))
        self.weights = self.root / "model.pt"
        self.weights.touch()
        self.args = ["--config", str(self.root / "config.yaml"), "--weights", str(self.weights)]

    def test_missing_labels_and_invalid_values_are_rejected(self):
        label = self.root / "data/labels/train/sample.txt"
        for text in ("1 0.5 0.5 0.2 0.4", "0 nan 0.5 0.2 0.4", "0 0.5 0.5 0 0.4", "0 0.5"):
            label.write_text(text)
            with self.subTest(text=text), self.assertRaises(ValueError):
                runner.check_split(self.data, "train")
        label.unlink()
        with self.assertRaises(ValueError):
            runner.check_split(self.data, "train")

    def test_explicit_background_label_is_accepted(self):
        (self.root / "data/labels/train/sample.txt").write_text("")
        self.assertEqual(runner.check_split(self.data, "train")["images"], 1)

    def test_preflight_never_loads_a_model_or_creates_runs(self):
        yolo = Mock()
        with patch.object(runner, "ROOT", self.root), patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=yolo)}):
            self.assertEqual(runner.main(["train", *self.args, "--check"]), 0)
        yolo.assert_not_called()
        self.assertFalse((self.root / "runs").exists())

    def test_train_and_val_use_persisted_absolute_dataset(self):
        for mode in ("train", "val"):
            yolo = Mock()
            with self.subTest(mode=mode), patch.object(runner, "ROOT", self.root), patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=yolo)}):
                self.assertEqual(runner.main([mode, *self.args]), 0)
            method = getattr(yolo.return_value, mode)
            method.assert_called_once()
            dataset = Path(method.call_args.kwargs["data"])
            self.assertTrue(dataset.is_file())
            self.assertEqual(yaml.safe_load(dataset.read_text())["path"], self.data["path"])
            self.assertEqual(method.call_args.kwargs["device"], "cpu")
