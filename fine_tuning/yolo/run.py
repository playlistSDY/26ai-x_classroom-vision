"""Train/validate classroom person detection independently of the desktop app."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve(path, base=None):
    base = ROOT if base is None else base
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_yaml(path):
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML 매핑이 필요합니다: {path}")
    return value


def check_split(data, split):
    """Require explicit labels, including empty files for background images."""
    relative = data.get(split)
    if not isinstance(relative, str):
        raise ValueError(f"데이터 YAML에 {split} 이미지 폴더를 지정하세요.")
    folder = resolve(relative, Path(data["path"]))
    if not folder.is_dir():
        raise ValueError(f"이미지 폴더가 없습니다: {folder}")
    images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"이미지가 없습니다: {folder}")
    for image in images:
        # Match Ultralytics' images/... -> labels/... convention.
        parts = list(image.parts)
        if "images" not in parts:
            raise ValueError(f"images/ 아래에 이미지를 배치하세요: {image}")
        index = len(parts) - 1 - parts[::-1].index("images")
        parts[index] = "labels"
        label = Path(*parts).with_suffix(".txt")
        if not label.is_file():
            raise ValueError(f"라벨이 없습니다 (배경 이미지는 빈 파일 사용): {label}")
        for number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"YOLO 5열 라벨이 필요합니다: {label}:{number}")
            values = [float(value) for value in fields]
            if (not all(math.isfinite(v) for v in values) or values[0] != 0
                    or not all(0 <= v <= 1 for v in values[1:])
                    or values[3] <= 0 or values[4] <= 0):
                raise ValueError(f"class=0, 정규화 좌표, 양수 크기가 필요합니다: {label}:{number}")
    return {"images": len(images), "folder": str(folder)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="강의실 YOLO 파인튜닝 / 탐지 검증")
    parser.add_argument("mode", choices=("train", "val"))
    parser.add_argument("--config", default="configs/yolo-train.yaml", help="fine_tuning/ 기준 경로")
    parser.add_argument("--weights", type=Path, help="기존 .pt 가중치. 생략 시 앱의 캐시된 YOLO26n 사용")
    parser.add_argument("--device", default="cpu", help="cpu, mps, 0 (CUDA)")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--check", action="store_true", help="데이터·설정만 검사. 모델 로딩·학습 없음")
    args = parser.parse_args(argv)
    try:
        config = load_yaml(resolve(args.config))
        data_file = resolve(config.pop("data"))
        data = load_yaml(data_file)
        if data.get("names") not in ({0: "person"}, ["person"]):
            raise ValueError("현재 파이프라인은 names: {0: person} 단일 클래스만 지원합니다.")
        if "download" in data:
            raise ValueError("로컬 데이터만 사용하세요. download 설정은 지원하지 않습니다.")
        data["path"] = str(resolve(data.get("path", "."), data_file.parent))
        splits = ("train", "val") if args.mode == "train" else (args.split,)
        summary = {split: check_split(data, split) for split in splits}
        weights = (args.weights.expanduser().resolve() if args.weights else
                   Path.home() / ".cache/aix-classroom-tracking/yolo26n.pt")
        if not weights.is_file():
            raise ValueError(f"기존 YOLO 가중치를 --weights로 지정하세요: {weights}")
        for key in ("epochs", "batch"):
            value = getattr(args, key)
            if value is not None:
                config[key] = value
            if config.get(key, 1) <= 0:
                raise ValueError(f"{key}는 양수여야 합니다.")
        config["project"] = str(resolve(config.get("project", "runs/yolo")))
        print(json.dumps({"mode": args.mode, "weights": str(weights), "device": args.device,
                          "data": summary, "config": config}, indent=2, ensure_ascii=False))
        if args.check:
            return 0
        # Persist the resolved dataset so run args remain reusable after training.
        serialized = yaml.safe_dump(data, allow_unicode=True)
        digest = hashlib.sha256(serialized.encode()).hexdigest()[:12]
        prepared = ROOT / "runs" / "_datasets" / f"{digest}.yaml"
        prepared.parent.mkdir(parents=True, exist_ok=True)
        prepared.write_text(serialized, encoding="utf-8")
        from ultralytics import YOLO
        model = YOLO(str(weights), task="detect")
        if args.mode == "train":
            model.train(data=str(prepared), device=args.device, **config)
        else:
            model.val(data=str(prepared), device=args.device, split=args.split,
                      imgsz=config.get("imgsz", 768), batch=config.get("batch", 4),
                      workers=config.get("workers", 0), project=config["project"],
                      name=f"{config.get('name', 'classroom-person')}-{args.split}",
                      exist_ok=False, plots=True)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        parser.exit(1, f"준비 오류: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
