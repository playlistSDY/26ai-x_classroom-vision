"""Protect class remapping and atomic preparation of the Roboflow export."""

from pathlib import Path
import sys
import tempfile
import unittest

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "yolo"))
from prepare_classroom_dodzk import prepare


class ClassroomPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source, self.output = self.root / "raw", self.root / "person"
        self.source.mkdir()
        (self.source / "data.yaml").write_text(yaml.safe_dump({"names": [str(i) for i in range(10)]}))
        for split in ("train", "valid", "test"):
            images, labels = self.source / split / "images", self.source / split / "labels"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            (images / "sample.jpg").write_bytes(b"fixture-image")
            (labels / "sample.txt").write_text("0 0.5 0.3 0.1 0.1\n8 0.5 0.5 0.4 0.6\n9 0.2 0.5 0.2 0.4\n")

    def test_person_mapping_preserves_raw_labels_and_splits(self):
        label = self.source / "train/labels/sample.txt"
        original = label.read_bytes()
        manifest = prepare(self.source, self.output)
        self.assertEqual(label.read_bytes(), original)
        for split in ("train", "valid", "test"):
            self.assertEqual((self.output / split / "labels/sample.txt").read_text(),
                             "0 0.5 0.5 0.4 0.6\n0 0.2 0.5 0.2 0.4\n")
            self.assertEqual(manifest["splits"][split]["boxes"], 2)
            self.assertEqual((self.output / split / "images/sample.jpg").read_bytes(), b"fixture-image")
        self.assertEqual(yaml.safe_load((self.output / "data.yaml").read_text())["names"], {0: "person"})

    def test_mapping_uses_class_names_when_original_ids_are_reordered(self):
        names = [str(i) for i in reversed(range(10))]
        (self.source / "data.yaml").write_text(yaml.safe_dump({"names": names}))
        for split in ("train", "valid", "test"):
            (self.source / split / "labels/sample.txt").write_text("0 0.5 0.5 0.4 0.6\n1 0.2 0.5 0.2 0.4\n")
        prepare(self.source, self.output)
        self.assertEqual(len((self.output / "train/labels/sample.txt").read_text().splitlines()), 2)

    def test_invalid_excluded_box_rejects_partial_output(self):
        (self.source / "test/labels/sample.txt").write_text("5 nan 0.5 0.2 0.4\n")
        with self.assertRaises(ValueError):
            prepare(self.source, self.output)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".person-prepare-*")))

    def test_missing_or_orphan_label_is_rejected(self):
        label = self.source / "train/labels/sample.txt"
        label.unlink()
        with self.assertRaises(ValueError):
            prepare(self.source, self.output)
        label.write_text("")
        (label.parent / "orphan.txt").write_text("")
        with self.assertRaises(ValueError):
            prepare(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_existing_output_is_preserved(self):
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("keep")
        with self.assertRaises(ValueError):
            prepare(self.source, self.output)
        self.assertEqual(marker.read_text(), "keep")

    def test_uncovered_head_excludes_the_whole_image(self):
        images, labels = self.source / "valid/images", self.source / "valid/labels"
        (images / "incomplete.jpg").write_bytes(b"incomplete-image")
        (labels / "incomplete.txt").write_text("0 0.9 0.2 0.1 0.1\n8 0.5 0.5 0.4 0.6\n")
        manifest = prepare(self.source, self.output)
        self.assertFalse((self.output / "valid/images/incomplete.jpg").exists())
        self.assertFalse((self.output / "valid/labels/incomplete.txt").exists())
        self.assertEqual(manifest["splits"]["valid"]["images"], 1)
        self.assertEqual(manifest["splits"]["valid"]["excluded_images"], 1)
        self.assertEqual(manifest["splits"]["valid"]["excluded_files"], ["valid/images/incomplete.jpg"])
        self.assertEqual(manifest["excluded_images"][0]["image"], "valid/images/incomplete.jpg")
        self.assertEqual(manifest["excluded_images"][0]["uncovered_heads"][0]["class"], "0")

    def test_explicit_background_is_preserved(self):
        (self.source / "valid/labels/sample.txt").write_text("")
        manifest = prepare(self.source, self.output)
        self.assertEqual((self.output / "valid/labels/sample.txt").read_text(), "")
        self.assertEqual(manifest["splits"]["valid"]["empty_images"], 1)


if __name__ == "__main__":
    unittest.main()
