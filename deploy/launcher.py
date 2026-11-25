"""Local desktop launch controls; no notebook widget renderer or network UI."""
from __future__ import annotations

from collections import deque
import os
from pathlib import Path
import signal
import subprocess
import sys
from threading import Thread

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QHBoxLayout,
                              QLabel, QLineEdit, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget)
from app import create_application
from camera_devices import discover

ROOT = Path(__file__).resolve().parent


class Launcher(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Classroom Vision · 실행 준비")
        self.resize(640, 540)
        self.setStyleSheet('''
            QWidget { background: #101620; color: #e6edf7; font-size: 14px; }
            QLineEdit, QComboBox, QPlainTextEdit { background: #202733; border: 1px solid #3b4659;
                border-radius: 6px; padding: 7px; }
            QPushButton { background: #283449; border: none; border-radius: 6px; padding: 10px 16px; }
            QPushButton:hover { background: #3b4a64; }
            QPushButton:disabled { color: #7d8797; background: #1c2330; }
            QPushButton#start { background: #5757cc; font-weight: 600; }
            QComboBox QAbstractItemView { background: #202733; selection-background-color: #5757cc; }
        ''')
        self.process = None
        self.log = None
        self.checking = False
        self.hidden_for_run = False
        self.scans = deque()
        self.scanning = False
        title = QLabel("Classroom Vision")
        title.setStyleSheet("font-size: 26px; font-weight: 700;")
        subtitle = QLabel("설정을 고른 뒤 추적 시작을 누르면 기존 추적 화면이 열립니다.")
        subtitle.setWordWrap(True)
        self.source = QComboBox()
        self.source.addItem("카메라", "camera")
        self.source.addItem("녹화 영상", "video")
        self.camera = QComboBox()
        self.camera.addItem("카메라를 확인하는 중…", None)
        self.refresh = QPushButton("새로고침")
        camera_row = QHBoxLayout()
        camera_row.addWidget(self.camera, 1); camera_row.addWidget(self.refresh)
        self.video = QLineEdit()
        sample = Path.home() / "Downloads/IMG_6463.MOV"
        if sample.is_file(): self.video.setText(str(sample))
        self.video.setPlaceholderText("시연할 영상 파일을 선택하세요")
        self.browse = QPushButton("파일 선택")
        video_row = QHBoxLayout()
        video_row.addWidget(self.video, 1); video_row.addWidget(self.browse)
        self.device = QComboBox()
        self.device.addItem("자동 · 사용 가능 장치 선택", "auto")
        self.device.addItem("CPU", "cpu")
        self.size = QComboBox()
        self.size.addItem("정확도 우선 · 960", 960)
        self.size.addItem("속도 우선 · 640", 640)
        self.fullscreen = QCheckBox("발표용 전체 화면")
        form = QFormLayout()
        form.setSpacing(12)
        form.addRow("입력", self.source)
        form.addRow("카메라", camera_row)
        form.addRow("영상", video_row)
        form.addRow("처리 장치", self.device)
        form.addRow("탐지 설정", self.size)
        form.addRow("", self.fullscreen)
        self.start = QPushButton("추적 시작")
        self.start.setObjectName("start")
        self.stop = QPushButton("중지")
        self.check = QPushButton("환경 확인")
        buttons = QHBoxLayout()
        for button in (self.start, self.stop, self.check):buttons.addWidget(button)
        self.status = QLabel("장치를 확인하고 있습니다…")
        self.status.setWordWrap(True)
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMaximumBlockCount(100)
        self.output.setMaximumHeight(120)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(14)
        for widget in (title, subtitle):layout.addWidget(widget)
        layout.addLayout(form);layout.addLayout(buttons)
        layout.addWidget(self.status);layout.addWidget(self.output)
        self.refresh.clicked.connect(lambda: self.scan(False))
        self.source.currentIndexChanged.connect(self.update_controls)
        self.browse.clicked.connect(self.choose_video)
        self.start.clicked.connect(lambda: self.launch(False))
        self.check.clicked.connect(lambda: self.launch(True))
        self.stop.clicked.connect(self.stop_process)
        self.timer = QTimer(self);self.timer.timeout.connect(self.poll);self.timer.start(150)
        self.update_controls()
        self.scan(True)

    def scan(self, devices=False):
        if self.scanning:return
        self.scanning = True;self.refresh.setEnabled(False)
        self.status.setText("카메라 이름과 처리 장치를 확인하고 있습니다…")
        def work():
            cameras, options, errors = [], [], []
            try:cameras = discover()
            except Exception as error:errors.append(str(error))
            if devices:
                try:
                    import torch
                    if torch.cuda.is_available():options.append(("CUDA · " + torch.cuda.get_device_name(0), "cuda:0"))
                    if torch.backends.mps.is_available():options.append(("Mac MPS", "mps"))
                except Exception as error:errors.append(str(error))
            self.scans.append((cameras, options, errors))
        Thread(target=work, daemon=True).start()

    def update_controls(self):
        camera_mode = self.source.currentData() == "camera"
        running = self.process is not None and self.process.poll() is None
        self.camera.setEnabled(camera_mode and not running)
        self.refresh.setEnabled(camera_mode and not running and not self.scanning)
        self.video.setEnabled(not camera_mode and not running)
        self.browse.setEnabled(not camera_mode and not running)
        for widget in (self.source, self.device, self.size, self.fullscreen):widget.setEnabled(not running)
        self.start.setEnabled(not running and (not camera_mode or self.camera.currentData() is not None))
        self.check.setEnabled(not running)
        self.stop.setEnabled(running)

    def choose_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "시연 영상 선택", str(Path.home()),
                                            "영상 (*.mp4 *.mov *.MOV *.avi *.mkv);;모든 파일 (*)")
        if path:self.video.setText(path)

    def launch(self, checking=False):
        if self.process is not None and self.process.poll() is None:return
        command = [sys.executable, "-u", str(ROOT / "app.py"), "--offline",
                   "--device", self.device.currentData(), "--size", str(self.size.currentData())]
        if checking:command.append("--check")
        elif self.source.currentData() == "camera":
            camera = self.camera.currentData()
            if camera is None:
                self.status.setText("카메라를 연결한 뒤 새로고침해 주세요.");return
            command += ["--camera", str(camera.index), "--camera-backend", str(camera.backend),
                        "--camera-name", camera.name]
            if camera.path:command += ["--camera-id", camera.path]
        else:
            path = Path(self.video.text().strip().strip('"').strip("'")).expanduser()
            if not path.is_file():
                self.status.setText("영상 파일을 선택해 주세요.");return
            command += ["--video", str(path.resolve())]
        if self.fullscreen.isChecked() and not checking:command.append("--fullscreen")
        output_dir = ROOT / "outputs";output_dir.mkdir(exist_ok=True)
        self.log_path = output_dir / f"demo-{os.getpid()}.log"
        self.log = self.log_path.open("w", encoding="utf-8")
        options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        try:self.process = subprocess.Popen(command, cwd=ROOT, stdout=self.log, stderr=subprocess.STDOUT,
                                                  env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1"), **options)
        except OSError as error:
            self.log.close();self.status.setText(str(error));return
        self.checking = checking;self.last_log = ""
        self.hidden_for_run = False
        self.output.clear()
        self.status.setText("환경 확인 중…" if checking else "추적 화면을 준비하고 있습니다…")
        self.update_controls()

    def poll(self):
        if self.scans:
            cameras, options, errors = self.scans.pop()
            previous = self.camera.currentData()
            self.camera.clear()
            counts = {}
            for camera in cameras:
                counts[camera.name] = counts.get(camera.name, 0) + 1
                suffix = f" · {counts[camera.name]}" if counts[camera.name] > 1 else ""
                self.camera.addItem(camera.name + suffix, camera)
                if previous is not None and ((previous.path and previous.path == camera.path)
                        or (not previous.path and previous.backend == camera.backend
                            and previous.index == camera.index and previous.name == camera.name)):
                    self.camera.setCurrentIndex(self.camera.count()-1)
            if not cameras:self.camera.addItem("연결된 카메라가 없습니다", None)
            for label, value in options:
                if self.device.findData(value) < 0:self.device.addItem(label, value)
            self.scanning = False
            self.status.setText("준비 완료 · 입력과 장치를 선택하세요." if not errors else " · ".join(errors))
            self.update_controls()
        if self.process is not None:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            if text != self.last_log:
                self.output.setPlainText(text);self.last_log = text
                scrollbar = self.output.verticalScrollBar();scrollbar.setValue(scrollbar.maximum())
            code = self.process.poll()
            if (not self.checking and not self.hidden_for_run
                    and "추적 시작 완료" in text and code is None):
                self.hidden_for_run = True
                self.status.setText("추적 실행 중 · 별도 추적 창을 사용하세요.")
                self.hide()
            if code is not None:
                self.log.close();self.process = None
                self.status.setText("환경 확인 완료" if self.checking and code == 0 else
                                    "종료되었습니다." if code in (0, -15) else "실행을 확인해 주세요. 아래 메시지에 원인이 표시됩니다.")
                self.update_controls()
                if self.hidden_for_run or code not in (0, -15):
                    self.showNormal();self.raise_();self.activateWindow()
                    if sys.platform == "darwin" and QApplication.instance().platformName() != "offscreen":
                        from AppKit import NSApplication
                        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
                self.hidden_for_run = False

    def stop_process(self):
        process = self.process
        if process is None or process.poll() is not None:return
        self.status.setText("종료 중…")
        try:
            if os.name == "nt":process.send_signal(signal.CTRL_BREAK_EVENT)
            else:process.terminate()
        except OSError:process.terminate()
        QTimer.singleShot(8000, lambda: process.kill() if process.poll() is None else None)

    def closeEvent(self, event):
        self.stop_process()
        if self.process is not None:
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:self.process.kill()
        event.accept()


def main():
    application = create_application()
    application.setApplicationName("Classroom Vision 실행 준비")
    window = Launcher();window.show();window.raise_();window.activateWindow()
    if sys.platform == "darwin" and application.platformName() != "offscreen":
        from AppKit import NSApplication
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
