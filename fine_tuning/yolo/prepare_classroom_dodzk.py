"""Prepare the audited classroom-dodzk v1 upper-body labels as person boxes."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import yaml

from run import IMAGE_SUFFIXES, ROOT, check_split, load_yaml


SOURCE_URL = "https://universe.roboflow.com/reddy-projects-zulke/classroom-dodzk/dataset/1"
DEFAULT_DATA = ROOT / "data" / "roboflow-classroom-dodzk"
PERSON_CLASSES = {"8", "9"}
HEAD_CLASSES = {"0", "1", "2", "7"}


def prepare(source: Path, output: Path):
    source, output = source.expanduser().resolve(), output.expanduser().resolve()
    if output.exists():
        raise ValueError(f"출력 폴더가 이미 있습니다: {output}")
    if source in output.parents:
        raise ValueError("출력 폴더는 원본 데이터 폴더 밖에 지정하세요.")
    source_yaml = source / "data.yaml"
    names = load_yaml(source_yaml).get("names")
    classes = dict(enumerate(names)) if isinstance(names, list) else names
    if (not isinstance(classes, dict) or len(classes) != 10
            or set(classes.values()) != {str(index) for index in range(10)}):
        raise ValueError("classroom-dodzk v1의 숫자 클래스 0~9가 필요합니다.")
    classes = {int(index): name for index, name in classes.items()}
    if set(classes) != set(range(10)):
        raise ValueError("원본 클래스 번호는 0~9여야 합니다.")
    data = {"path": ".", "train": "train/images", "val": "valid/images",
            "test": "test/images", "names": {0: "person"}}
    manifest = {"source_url": SOURCE_URL, "source_version": 1,
                "source_yaml_sha256": hashlib.sha256(source_yaml.read_bytes()).hexdigest(),
                "mapping_by_source_name": {name: 0 for name in sorted(PERSON_CLASSES)},
                "selection_basis": "Visual review of contextual crops and original annotations",
                "coverage_check": "Head centers must fall inside a retained upper-body box",
                "excluded_images": [],
                "splits": {}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".person-prepare-", dir=output.parent) as temporary:
        staging = Path(temporary)
        for split in ("train", "valid", "test"):
            images_dir, labels_dir = source / split / "images", source / split / "labels"
            images = sorted(path for path in images_dir.rglob("*")
                            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
            if not images:
                raise ValueError(f"이미지가 없습니다: {images_dir}")
            expected_labels = {path.relative_to(images_dir).with_suffix(".txt") for path in images}
            actual_labels = {path.relative_to(labels_dir) for path in labels_dir.rglob("*.txt")}
            if expected_labels != actual_labels:
                raise ValueError(f"이미지와 라벨 파일이 일치하지 않습니다: {split}")
            counts = Counter(source_images=len(images), images=0, excluded_images=0,
                             excluded_head_centers=0, boxes=0, empty_images=0)
            original_classes = Counter()
            excluded_files = []
            for image in images:
                relative = image.relative_to(images_dir)
                label = labels_dir / relative.with_suffix(".txt")
                selected, body_boxes, heads = [], [], []
                for number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                    fields = line.split()
                    if len(fields) != 5:
                        raise ValueError(f"YOLO 5열 라벨이 필요합니다: {label}:{number}")
                    category, x, y, width, height = map(float, fields)
                    if (not all(math.isfinite(value) for value in (category, x, y, width, height))
                            or category not in classes or not 0 <= x <= 1 or not 0 <= y <= 1
                            or not 0 < width <= 1 or not 0 < height <= 1
                            or min(x - width / 2, y - height / 2) < -1e-5
                            or max(x + width / 2, y + height / 2) > 1 + 1e-5):
                        raise ValueError(f"유효하지 않은 원본 박스입니다: {label}:{number}")
                    name = classes[int(category)]
                    original_classes[name] += 1
                    if name in PERSON_CLASSES:
                        selected.append("0 " + " ".join(fields[1:]))
                        body_boxes.append((x - width / 2, y - height / 2,
                                           x + width / 2, y + height / 2))
                    elif name in HEAD_CLASSES:
                        heads.append({"class": name, "center": [x, y], "line": number})
                uncovered = [head for head in heads if not any(
                    left <= head["center"][0] <= right and top <= head["center"][1] <= bottom
                    for left, top, right, bottom in body_boxes)]
                if uncovered:
                    counts["excluded_images"] += 1
                    counts["excluded_head_centers"] += len(uncovered)
                    excluded_files.append(f"{split}/images/{relative.as_posix()}")
                    manifest["excluded_images"].append({
                        "image": f"{split}/images/{relative.as_posix()}",
                        "reason": "head_center_without_person_box", "uncovered_heads": uncovered})
                    continue
                selected = list(dict.fromkeys(selected))
                target_image = staging / split / "images" / relative
                target_label = staging / split / "labels" / relative.with_suffix(".txt")
                target_image.parent.mkdir(parents=True, exist_ok=True)
                target_label.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(image, target_image)
                except OSError:
                    shutil.copy2(image, target_image)
                target_label.write_text("\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")
                counts["images"] += 1
                counts["boxes"] += len(selected)
                counts["empty_images"] += not selected
            manifest["splits"][split] = {**counts, "source_class_counts": dict(original_classes),
                                         "excluded_files": excluded_files}
        validation = {**data, "path": str(staging)}
        for split in ("train", "val", "test"):
            check_split(validation, split)
        (staging / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        (staging / "preparation.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if output.exists():
            raise ValueError(f"출력 폴더가 이미 있습니다: {output}")
        staging.rename(output)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="classroom-dodzk v1의 상반신 박스를 person으로 준비")
    parser.add_argument("--source", type=Path, default=DEFAULT_DATA / "raw")
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA / "person")
    args = parser.parse_args(argv)
    try:
        manifest = prepare(args.source, args.output)
    except (OSError, ValueError, TypeError, yaml.YAMLError) as error:
        parser.exit(1, f"준비 오류: {error}\n")
    print(json.dumps({"output": str(args.output.resolve()), **manifest}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
