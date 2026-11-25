"""Discover camera names without opening capture streams; refresh in a new process."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys


@dataclass(frozen=True)
class Camera:
    name: str
    index: int
    backend: int
    path: str = ""


def enumerate_local():
    import cv2
    from cv2_enumerate_cameras import enumerate_cameras
    backend = (cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else
               cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2)
    return [Camera(str(camera.name), int(camera.index), int(camera.backend), str(camera.path or ""))
            for camera in enumerate_cameras(backend)]


def discover():
    result = subprocess.run([sys.executable, str(Path(__file__).resolve())], capture_output=True,
                            text=True, encoding="utf-8", timeout=20,
                            env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1"))
    if result.returncode:
        raise RuntimeError("카메라 이름을 확인하지 못했습니다. 준비 셀을 다시 실행해 주세요.\n" + result.stderr[-500:])
    return [Camera(**data) for data in json.loads(result.stdout)]


if __name__ == "__main__":
    print(json.dumps([asdict(camera) for camera in enumerate_local()], ensure_ascii=False))
