"""Classroom Vision: standalone camera/video demonstration entry point."""
from __future__ import annotations

import argparse
from collections import deque
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile
from threading import Event, Lock, Thread
import time


class Session:
    """Keep model inference off the UI thread and discard pre-reset results."""
    def __init__(self, args):
        self.args = args
        self.stop = Event()
        self.reset = Event()
        self.lock = Lock()
        self.generation = 0
        self.frames = deque(maxlen=1)
        self.results = deque(maxlen=1)
        self.error = None
        self.done = False
        self.worker = Thread(target=self.run, daemon=True, name="inference")
        self.capture = None

    def reset_session(self):
        with self.lock:
            self.generation += 1
            self.reset.set()
            self.results.clear()
        return True

    def read_camera(self):
        import cv2
        camera = None
        try:
            index, backend = self.args.camera, getattr(self.args, "camera_backend", 0)
            name = getattr(self.args, "camera_name", None) or "선택한 카메라"
            identifier = getattr(self.args, "camera_id", None)
            if identifier:
                from camera_devices import discover
                selected = next((item for item in discover()
                                 if item.path == identifier and item.backend == backend), None)
                if selected is None:
                    raise RuntimeError(f"{name}의 연결이 바뀌었습니다. 카메라를 새로고침해 주세요.")
                index, backend = selected.index, selected.backend
            camera = cv2.VideoCapture(index, backend)
            if not camera.isOpened():
                raise RuntimeError(f"{name}를 열 수 없습니다. 연결과 카메라 권한을 확인하세요.")
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            while not self.stop.is_set():
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("카메라 연결이 끊겼습니다.")
                self.frames.append((time.monotonic(), frame))
        except Exception as error:
            self.error = error
        finally:
            if camera is not None:
                camera.release()

    def run(self):
        source = None
        try:
            from tracking_core import PersonTracking, select_device
            tracker = PersonTracking(select_device(self.args.device), image_size=self.args.size,
                                     detector_weights=self.args.weights)
            if self.args.video:
                from video_source import VideoSource
                source = VideoSource(self.args.video)
            else:
                self.capture = Thread(target=self.read_camera, daemon=True, name="camera")
                self.capture.start()
            next_sample = 0.0
            while not self.stop.is_set() and self.error is None:
                if source:
                    item = source.read()
                    if item is None:
                        self.done = True
                        break
                    timestamp, frame = item
                    if timestamp + 1e-6 < next_sample:
                        continue
                    next_sample += 1 / self.args.video_fps
                else:
                    try:
                        timestamp, frame = self.frames.pop()
                    except IndexError:
                        self.stop.wait(.01)
                        continue
                with self.lock:
                    generation = self.generation
                    if self.reset.is_set():
                        tracker.reset_session()
                        self.reset.clear()
                observations, assignments = tracker.process(frame, timestamp)
                result = (frame, timestamp, observations, assignments, tracker.last_seen, str(tracker.device))
                with self.lock:
                    if generation == self.generation:
                        self.results.append((generation, result))
        except Exception as error:
            self.error = error
        finally:
            if source:
                source.close()
            self.stop.set()

    def close(self):
        self.stop.set()
        if self.worker.ident is not None:
            self.worker.join(timeout=4)
        if self.capture is not None:
            self.capture.join(timeout=2)


def create_application():
    import cv2
    from PySide6.QtCore import QLibraryInfo
    from PySide6.QtWidgets import QApplication
    if os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH") == str(Path(cv2.__file__).parent / "qt/plugins"):
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
    arguments = [sys.argv[0]]
    directory = None
    if sys.platform == "darwin":
        from stat import UF_HIDDEN
        plugins = Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)) / "platforms"
        files = list(plugins.glob("libq*.dylib"))
        if any(path.stat().st_flags & UF_HIDDEN for path in files):
            directory = tempfile.TemporaryDirectory(prefix="classroom-qt-")
            for path in files:
                shutil.copyfile(path, Path(directory.name) / path.name)
            arguments += ["-platformpluginpath", directory.name]
    application = QApplication(arguments)
    application._platform_directory = directory
    return application


def main():
    parser = argparse.ArgumentParser(description="Classroom Vision · 카메라 또는 영상 시연")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, help="카메라 번호 (기본 0)")
    source.add_argument("--video", type=Path, help="시연할 영상 경로 · HDR MOV 자동 색상 변환")
    parser.add_argument("--camera-backend", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--camera-id", help=argparse.SUPPRESS)
    parser.add_argument("--camera-name", help=argparse.SUPPRESS)
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cuda:0", "cpu"))
    parser.add_argument("--size", type=int, default=960, help="탐지 크기 · 속도 우선이면 640")
    parser.add_argument("--video-fps", type=float, default=8, help="영상에서 처리할 최대 초당 프레임 수")
    parser.add_argument("--weights", type=Path, help="다른 person 탐지 가중치 경로")
    parser.add_argument("--offline", action="store_true", help="모델 다운로드 없이 실행")
    parser.add_argument("--check", action="store_true", help="카메라 없이 실행 환경과 모델 추론 확인")
    parser.add_argument("--fullscreen", action="store_true", help="전체 화면 시연")
    args = parser.parse_args()
    if args.size < 128 or not 0 < args.video_fps <= 60 or (args.camera is not None and args.camera < 0):
        parser.error("탐지 크기는 128 이상, 영상 FPS는 0 초과 60 이하, 카메라 번호는 0 이상이어야 합니다.")
    args.camera = 0 if args.camera is None else args.camera
    if args.video is not None and not args.video.expanduser().is_file():
        parser.error(f"영상을 찾을 수 없습니다: {args.video}")
    try:
        from models import prepare, detector_path
        prepare(download=not args.offline)
        args.weights = (args.weights.expanduser().resolve() if args.weights else detector_path())
        if not args.weights.is_file():
            raise FileNotFoundError(f"가중치를 찾을 수 없습니다: {args.weights}")
        if args.check:
            import numpy as np
            from tracking_core import PersonTracking, select_device
            tracker = PersonTracking(select_device(args.device), image_size=args.size,
                                     detector_weights=args.weights)
            tracker.process(np.zeros((720, 1280, 3), dtype=np.uint8), 0.0)
            print(f"실행 준비 완료 · {tracker.device} · {args.weights.name}")
            return 0
        application = create_application()
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QMessageBox
        from tracking_core import COLORS
        from tracking_window import TrackingWindow
        session = Session(args)
        label = args.video.name if args.video else (args.camera_name or args.camera)
        window = TrackingWindow(label, args.device, COLORS, session.reset_session)
        window.setWindowTitle("Classroom Vision")
        window.status.setText("모델 준비 중…")
        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            if hasattr(signal, name):
                signal.signal(getattr(signal, name), lambda *_: window.close())
        timer = QTimer(window)

        announced_ready = False

        def poll():
            nonlocal announced_ready
            if session.error is not None:
                timer.stop()
                print(f"실행 오류: {session.error}", flush=True)
                QMessageBox.critical(window, "실행 확인", str(session.error))
                window.close()
                return
            with session.lock:
                item = session.results.pop() if session.results else None
                if item is not None and item[0] == session.generation:
                    frame, timestamp, observations, assignments, last_seen, device = item[1]
                    source_label = f"{args.video.name} · {timestamp:.1f}초" if args.video else (args.camera_name or f"카메라 {args.camera}")
                    window._status_prefix = f"{source_label} · {device.upper()}"
                    window.show_frame(frame, observations, assignments, last_seen)
                    window.update_absences(timestamp, last_seen)
                    if not announced_ready:
                        window.raise_();window.activateWindow()
                        if sys.platform == "darwin" and application.platformName() != "offscreen":
                            try:
                                from AppKit import NSApplication
                                NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
                            except ImportError:
                                pass
                        print(f"추적 시작 완료 · {source_label} · {device.upper()}", flush=True)
                        announced_ready = True
            if session.done:
                timer.stop()
                window.status.setText(window.status.text() + " · 재생 완료")

        timer.timeout.connect(poll)
        application.aboutToQuit.connect(session.stop.set)
        window.showFullScreen() if args.fullscreen else window.show()
        session.worker.start()
        timer.start(16)
        try:
            application.exec()
            return 1 if session.error else 0
        finally:
            timer.stop()
            session.close()
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"실행을 확인해 주세요: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
