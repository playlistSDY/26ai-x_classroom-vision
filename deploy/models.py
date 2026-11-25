"""Prepare and verify the small, fixed set of models required by the demo."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MANIFEST = json.loads((MODEL_DIR / "manifest.json").read_text(encoding="utf-8"))
FILES = {item["name"]: item for item in MANIFEST["files"]}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def valid(path, expected):
    return path.is_file() and digest(path) == expected


def checkpoint(name, *, download=True, cache=None):
    if name not in FILES:
        raise ValueError(f"배포 모델 목록에 없는 파일입니다: {name}")
    item = FILES[name]
    target = MODEL_DIR / name
    if valid(target, item["sha256"]):
        return target
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cache = Path(cache or os.environ.get("AIX_MODEL_CACHE", Path.home() / ".cache/aix-classroom-tracking"))
    cached = cache / name
    with tempfile.TemporaryDirectory(prefix=".download-", dir=MODEL_DIR) as directory:
        temporary = Path(directory) / name
        if valid(cached, item["sha256"]):
            shutil.copyfile(cached, temporary)
        elif not download:
            raise RuntimeError(f"모델이 없거나 손상됐습니다: {name}\n인터넷 연결 후 python models.py --prepare 를 실행하세요.")
        elif item["url"]:
            print(f"모델 준비 중: {name}", flush=True)
            with urlopen(item["url"], timeout=90) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, 1024 * 1024)
        else:
            raise RuntimeError(f"동봉 가중치가 없거나 손상됐습니다: {name}\ndeploy 폴더를 다시 받아 주세요.")
        if not valid(temporary, item["sha256"]):
            raise RuntimeError(f"모델 검증 실패: {name}. 다시 준비해 주세요.")
        temporary.replace(target)
    return target


def detector_path():
    return checkpoint(MANIFEST["detector"])


def prepare(*, download=True, cache=None):
    paths = [checkpoint(name, download=download, cache=cache) for name in FILES]
    print(f"모델 {len(paths)}개 확인 완료 · {MODEL_DIR}", flush=True)
    return paths


def main():
    parser = argparse.ArgumentParser(description="시연에 필요한 모델을 한 번 준비하거나 오프라인 상태를 확인합니다.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true", help="다운로드 또는 기존 캐시에서 모델 준비")
    mode.add_argument("--check", action="store_true", help="다운로드 없이 파일과 해시 확인")
    args = parser.parse_args()
    try:
        prepare(download=args.prepare)
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
