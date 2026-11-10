"""Native controls and Unicode video labels for the live tracker."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from html import escape
from pathlib import Path
import time
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, QSignalBlocker, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QFontDatabase, QFontMetrics, QIcon, QImage, QKeySequence, QPainter, QPainterPath, QPalette, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QComboBox, QDialog, QFrame, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QToolButton, QStyle, QStyledItemDelegate, QStyleOptionToolButton, QStyleOptionViewItem, QStyleOptionComboBox, QStyleOptionSpinBox, QStylePainter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

if TYPE_CHECKING:
    from identity_tracker import Assignment


PART_LABELS = ("머리", "왼팔", "오른팔", "몸통", "왼다리", "오른다리", "왼발", "오른발")
STATUS_LABELS = {"checking": "확인 중", "tracked": "추적 유지", "new": "새 번호", "reidentified": "재연결"}


STYLE = """
QWidget#trackingWindow { background: #101620; color: #e6edf7; }
QLabel { color: #e6edf7; }
QLabel#title { font-size: 24px; font-weight: 700; }
QLabel[detail="true"] { color: #a2b0c4; }
QLabel[section="true"] { font-size: 14px; font-weight: 600; }
QFrame#sidebar { background: #111214; border: 1px solid #34363d; border-radius: 12px; }
QFrame#visibleCountCard, QFrame#cumulativeCountCard { background: #202027; border: none; border-radius: 9px; }
QLabel#visibleCount { color: #9696eb; }
QLabel#cumulativeCount { color: #f5f5f7; }
QLabel[metric="true"] { font-size: 32px; font-weight: 700; }
QLabel[metricCaption="true"] { color: #d4d5db; font-size: 13px; font-weight: 600; }
QLabel[metricUnit="true"], QFrame#sidebar QLabel[detail="true"] { color: #a4a7b3; }
QListWidget, QWidget#identityDiagnostics {
    background: #1f2025; color: #f5f5f7; border: 1px solid #383a42; border-radius: 8px;
}
QListWidget { padding: 5px 8px 5px 5px; outline: 0; }
QListWidget:focus { border-color: #383a42; }
QListWidget::item { padding: 7px 8px; border: none; border-left: 3px solid transparent; border-radius: 5px; }
QListWidget::item:hover:!selected { background: #292935; }
QListWidget::item:selected { background: rgba(87, 87, 204, 36); border-left: 3px solid transparent; }
QListWidget::item:selected:hover { background: rgba(87, 87, 204, 48); }
QComboBox QAbstractItemView::item:selected { background: #5757cc; color: white; }
QWidget#sidebarDetails, QScrollArea#detailsScroll { background: transparent; border: none; }
QFrame#disclosure { background: transparent; border: none; }
QToolButton#disclosureHeader { background: transparent; color: #b9bcc8; border: none; border-radius: 6px; text-align: left; padding: 9px 6px 9px 24px; font-weight: 600; }
QToolButton#disclosureHeader:hover { background: #25262e; color: #f5f5f7; }
QToolButton#disclosureHeader:checked { color: #9696eb; background: #23233a; }

QLabel#visibleCount { font-size: 48px; color: #9696eb; }
QFrame#absenceCard { background: #202027; border: none; border-radius: 9px; }
QLabel#absenceValue { font-size: 32px; font-weight: 700; color: #858894; }
QFrame#absenceCard[warning="true"] { background: #33232f; border: none; }
QFrame#absenceCard[warning="true"] QLabel#absenceValue { color: #f294a9; }

QScrollBar:vertical { background: transparent; width: 5px; margin: 0; }
QScrollBar::handle:vertical { background: #444650; border-radius: 2px; min-height: 24px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QLineEdit, QComboBox, QSpinBox {
    background: #2b2e36; color: #f5f5f7; border: 1px solid #383a42;
    border-radius: 7px; padding: 6px 8px; selection-background-color: #5757cc;
}
QSpinBox { padding-right: 24px; }
QSpinBox::up-button {
    subcontrol-origin: border; subcontrol-position: top right;
    width: 22px; background: transparent; border: none;
}
QSpinBox::down-button {
    subcontrol-origin: border; subcontrol-position: bottom right;
    width: 22px; background: transparent; border: none;
}
QSpinBox::up-arrow, QSpinBox::down-arrow { image: none; }
QComboBox { padding-right: 24px; }
QComboBox::drop-down {
    subcontrol-origin: border; subcontrol-position: top right;
    width: 22px; background: transparent; border: none;
}
QComboBox::down-arrow { image: none; }
QComboBox QAbstractItemView { background: #1f2025; color: white; selection-background-color: #5757cc; outline: 0; }
QLineEdit:focus, QSpinBox:focus, QPushButton:focus { border-color: #8787e1; }
QLineEdit:disabled, QPushButton:disabled { color: #9699a4; background: #24262d; }
QTableWidget { background: #1f2025; alternate-background-color: #24262c; color: #f5f5f7; border: none; }
QHeaderView::section { background: #2b2e36; color: #c8cad3; border: none; padding: 4px 2px; }
QPushButton { padding: 7px 12px; border: 1px solid #383a42; border-radius: 7px; background: #2b2e36; color: #f5f5f7; }
QPushButton:hover { background: #353842; }
QFrame#nameInputGroup { background: #2b2e36; border: 1px solid #383a42; border-radius: 8px; }
QLineEdit#nameEdit { background: transparent; border: none; padding: 6px 8px; }
QPushButton#saveNameButton { padding: 0; border-radius: 5px; }
QPushButton#saveNameButton:enabled { background: #5757cc; border-color: #5757cc; color: #ffffff; }
QPushButton#saveNameButton:enabled:hover { background: #6868d6; border-color: #6868d6; }
QPushButton#saveNameButton:enabled:pressed { background: #4848b5; border-color: #4848b5; }
QPushButton#resetButton { background: transparent; border-color: transparent; border-radius: 8px; padding: 0; }
QPushButton#resetButton:hover { border-color: #45456d; background: #292941; }
QPushButton#resetButton:pressed { background: #333351; }
"""


@dataclass(frozen=True)
class _DiagnosticRecord:
    current: Assignment
    comparison: Assignment | None = None
    decision: Assignment | None = None
    face_comparison: Assignment | None = None


def _reset_icon():
    icon = QIcon()
    for mode, color in ((QIcon.Mode.Normal, "#b5b7c3"),
                        (QIcon.Mode.Active, "#9696eb"),
                        (QIcon.Mode.Disabled, "#666872")):
        pixmap = QPixmap(48, 48)
        pixmap.setDevicePixelRatio(2)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(color), 1.8)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        path = QPainterPath(QPointF(19, 9))
        path.cubicTo(17, 3, 8, 3, 5, 9)
        path.cubicTo(2, 15, 8, 21, 14, 19)
        path.cubicTo(17, 18, 19, 16, 19, 14)
        painter.drawPath(path)
        painter.drawLine(QPointF(14, 9), QPointF(19, 9))
        painter.drawLine(QPointF(19, 9), QPointF(19, 4))
        painter.end()
        icon.addPixmap(pixmap, mode)
    return icon


class _ResetDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent, Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setWindowTitle("추적 기록 초기화")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(parent.mapToGlobal(QPoint(0, 0)).x(),
                         parent.mapToGlobal(QPoint(0, 0)).y(), parent.width(), parent.height())
        self.setStyleSheet("""
            QFrame#resetCard { background: #1f2025; border: 1px solid #383a42; border-radius: 12px; }
            QLabel#resetTitle { color: #f5f5f7; font-size: 21px; font-weight: 700; }
            QLabel#resetDescription { color: #a4a7b3; font-size: 14px; }
            QFrame#resetDivider { background: #34363d; border: none; }
            QPushButton { background: #2b2e36; color: #f5f5f7; border: none; border-radius: 7px; padding: 9px 18px; }
            QPushButton:hover { background: #383b46; }
            QPushButton:focus { border: none; }
            QPushButton#confirmReset { background: #5757cc; color: #ffffff; }
            QPushButton#confirmReset:hover { background: #6868d6; }
            QPushButton#confirmReset:pressed { background: #4848b5; }
        """)
        card = QFrame()
        card.setObjectName("resetCard")
        card.setFixedWidth(min(440, parent.width() - 48))
        content = QVBoxLayout(card)
        content.setContentsMargins(24, 24, 24, 20)
        content.setSpacing(12)
        title = QLabel("추적 기록 초기화")
        title.setObjectName("resetTitle")
        content.addWidget(title)
        description = QLabel("등록된 번호, 표시 이름, 누적 인원을 초기화할까요?")
        description.setObjectName("resetDescription")
        description.setWordWrap(True)
        content.addWidget(description)
        content.addSpacing(12)
        divider = QFrame()
        divider.setObjectName("resetDivider")
        divider.setFixedHeight(1)
        content.addWidget(divider)
        content.addSpacing(4)
        actions = QHBoxLayout()
        actions.setSpacing(8)
        actions.addStretch()
        cancel = QPushButton("취소")
        cancel.setDefault(True)
        cancel.clicked.connect(self.reject)
        confirm = QPushButton("초기화")
        confirm.setObjectName("confirmReset")
        confirm.setAutoDefault(False)
        confirm.clicked.connect(self.accept)
        actions.addWidget(cancel)
        actions.addWidget(confirm)
        content.addLayout(actions)
        layout = QVBoxLayout(self)
        layout.addStretch()
        layout.addWidget(card, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addStretch()
        cancel.setFocus()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 165))


class _DisclosureButton(QToolButton):
    def paintEvent(self, event):
        option = QStyleOptionToolButton()
        self.initStyleOption(option)
        option.text = ""
        painter = QStylePainter(self)
        painter.drawComplexControl(QStyle.ComplexControl.CC_ToolButton, option)
        painter.setFont(self.font())
        painter.setPen(self.palette().color(QPalette.ColorRole.ButtonText))
        text_rect = self.rect().adjusted(24, 0, -8, 0)
        text = self.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideRight, text_rect.width())
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor("#9696eb" if self.isChecked() else "#a4a7b3"), 1.5)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        center_y = self.height() / 2
        if self.isChecked():
            path = QPainterPath(QPointF(8, center_y - 2))
            path.lineTo(12, center_y + 2)
            path.lineTo(16, center_y - 2)
        else:
            path = QPainterPath(QPointF(10, center_y - 4))
            path.lineTo(14, center_y)
            path.lineTo(10, center_y + 4)
        painter.drawPath(path)


class _Disclosure(QFrame):
    def __init__(self, title, content):
        super().__init__()
        self.setObjectName("disclosure")
        self.header = _DisclosureButton()
        self.header.setObjectName("disclosureHeader")
        self.header.setText(title)
        self.header.setAccessibleName(title + " 펼치기")
        self.header.setCheckable(True)
        self.header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.content = content
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.header)
        layout.addWidget(content)
        self.header.toggled.connect(self._toggle)
        self._toggle(False)

    def _toggle(self, expanded):
        self.header.update()
        self.header.setAccessibleName(self.header.text() + (" 접기" if expanded else " 펼치기"))
        self.content.setVisible(expanded)


class _NoFocusItemDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        # The selected background indicates the current item.
        option.state &= ~QStyle.StateFlag.State_HasFocus


class _MetricLabel(QLabel):
    """Draw the number and its smaller unit on one shared text baseline."""

    def sizeHint(self):
        size = super().sizeHint()
        return QSize(size.width() + 24, size.height())

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        font = self.font()
        metrics = QFontMetrics(font)
        baseline = (self.height() - metrics.height()) / 2 + metrics.ascent()
        painter.setFont(font)
        painter.setPen(self.palette().color(QPalette.ColorRole.WindowText))
        painter.drawText(QPointF(0, baseline), self.text())
        unit_font = QFont(font)
        unit_font.setPixelSize(14)
        unit_font.setWeight(QFont.Weight.Normal)
        painter.setFont(unit_font)
        painter.setPen(QColor("#a4a7b3"))
        painter.drawText(QPointF(metrics.horizontalAdvance(self.text()) + 5, baseline), "명")


class _IdentityItemDelegate(_NoFocusItemDelegate):
    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        return QSize(size.width(), max(42, size.height()))

    def paint(self, painter, option, index):
        styled = QStyleOptionViewItem(option)
        self.initStyleOption(styled, index)
        name, _, status = styled.text.partition("  ·  ")
        styled.text = ""
        style = styled.widget.style() if styled.widget else QApplication.style()
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, styled, painter, styled.widget)
        painter.save()
        painter.setClipRect(option.rect)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        status_font = QFont(option.font)
        status_font.setPixelSize(11)
        status_font.setWeight(QFont.Weight.Normal)
        status_metrics = QFontMetrics(status_font)
        right = option.rect.right() - 10
        status_width = status_metrics.horizontalAdvance(status)
        status_left = right - status_width
        painter.setFont(status_font)
        painter.setPen(QColor("#a4a7b3"))
        baseline = option.rect.center().y() + (status_metrics.ascent() - status_metrics.descent()) / 2
        painter.drawText(QPointF(status_left, baseline), status)
        dot_color = "#7ab89e" if status == "화면 안" else "#757986"
        if status.startswith("확인 필요"):
            dot_color = "#f294a9"
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(dot_color))
        painter.drawEllipse(QPointF(status_left - 8, option.rect.center().y()), 2.5, 2.5)
        name_font = QFont(option.font)
        name_font.setPixelSize(13)
        name_font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(name_font)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        painter.setPen(QColor("#aaaaf2" if selected else "#e6e7ef"))
        name_metrics = QFontMetrics(name_font)
        left = option.rect.left() + 12
        text = name_metrics.elidedText(name, Qt.TextElideMode.ElideRight, max(0, status_left - left - 20))
        baseline = option.rect.center().y() + (name_metrics.ascent() - name_metrics.descent()) / 2
        painter.drawText(QPointF(left, baseline), text)
        painter.restore()


class _ThresholdSpinBox(QSpinBox):
    def paintEvent(self, event):
        super().paintEvent(event)
        option = QStyleOptionSpinBox()
        self.initStyleOption(option)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        controls = ((QStyle.SubControl.SC_SpinBoxUp, -1, self.value() < self.maximum()),
                    (QStyle.SubControl.SC_SpinBoxDown, 1, self.value() > self.minimum()))
        for control, direction, enabled in controls:
            rect = self.style().subControlRect(QStyle.ComplexControl.CC_SpinBox, option, control, self)
            center = rect.center()
            path = QPainterPath(QPointF(center.x() - 4, center.y() - direction * 2))
            path.lineTo(center.x() + 4, center.y() - direction * 2)
            path.lineTo(center.x(), center.y() + direction * 2)
            path.closeSubpath()
            painter.setBrush(QColor("#b9bcc8" if enabled and self.isEnabled() else "#626574"))
            painter.drawPath(path)


class _ComparisonComboBox(QComboBox):
    def paintEvent(self, event):
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        option.state &= ~QStyle.StateFlag.State_HasFocus
        painter = QStylePainter(self)
        painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, option)
        painter.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, option)
        arrow = self.style().subControlRect(QStyle.ComplexControl.CC_ComboBox, option,
                                           QStyle.SubControl.SC_ComboBoxArrow, self)
        center = arrow.center()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#b9bcc8" if self.isEnabled() else "#626574"))
        path = QPainterPath(QPointF(center.x() - 4, center.y() - 2))
        path.lineTo(center.x() + 4, center.y() - 2)
        path.lineTo(center.x(), center.y() + 2)
        path.closeSubpath()
        painter.drawPath(path)


class _VideoView(QWidget):
    person_selected = Signal(int)

    def __init__(self):
        super().__init__()
        self.setAccessibleName("추적 카메라 영상")
        self.setMinimumSize(320, 220)
        self._image = QImage()
        self._tracks = []
        self._hit_regions = []
        self._selected_raw_id = None

    def set_frame(self, frame, tracks):
        pixels = np.ascontiguousarray(frame)
        height, width = pixels.shape[:2]
        self._image = QImage(pixels.data, width, height, pixels.strides[0],
                             QImage.Format.Format_BGR888).copy()
        self.set_tracks(tracks)

    def set_tracks(self, tracks):
        self._tracks = tracks
        self.update()

    def select(self, raw_id):
        if self._selected_raw_id != raw_id:
            self._selected_raw_id = raw_id
            self.update()

    def clear(self):
        self._image = QImage()
        self._hit_regions.clear()
        self.set_tracks([])

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0a1018"))
        self._hit_regions.clear()
        if self._image.isNull():
            painter.setPen(QColor("#a2b0c4"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "카메라 영상을 기다리는 중")
            return

        scale = min(self.width() / self._image.width(), self.height() / self._image.height())
        width, height = self._image.width() * scale, self._image.height() * scale
        image_rect = QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(image_rect, self._image)
        painter.setClipRect(image_rect)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont(self.font())
        font.setPixelSize(15)
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        metrics = QFontMetrics(font)
        for (x1, y1, x2, y2), raw_id, text, color in self._tracks:
            rect = QRectF(image_rect.left() + x1 * scale, image_rect.top() + y1 * scale,
                          (x2 - x1) * scale, (y2 - y1) * scale)
            selected = raw_id == self._selected_raw_id
            painter.setPen(QPen(color, 3.5 if selected else 2.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(rect)

            text = metrics.elidedText(text, Qt.TextElideMode.ElideRight, max(1, int(width) - 14))
            label_width, label_height = metrics.horizontalAdvance(text) + 12, metrics.height() + 6
            left = max(image_rect.left(), min(rect.left(), image_rect.right() - label_width))
            top = max(image_rect.top(), rect.top() - label_height - 2)
            label_rect = QRectF(left, top, label_width, label_height)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawRoundedRect(label_rect, 4, 4)
            painter.setPen(QColor("#101620"))
            painter.drawText(QPointF(left + 6, top + metrics.ascent() + 3), text)
            self._hit_regions.append((rect, label_rect, raw_id))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            point = event.position()
            for _, label, raw_id in reversed(self._hit_regions):
                if label.contains(point):
                    self.person_selected.emit(raw_id)
                    return
            boxes = [(rect.width() * rect.height(), raw_id)
                     for rect, _, raw_id in self._hit_regions if rect.contains(point)]
            if boxes:
                self.person_selected.emit(min(boxes)[1])
                return
        super().mousePressEvent(event)


def _apply_pretendard():
    if "Pretendard" not in QFontDatabase.families():
        font_directory = Path(__file__).resolve().parent / "assets" / "fonts"
        for weight in ("Regular", "SemiBold", "Bold"):
            font_path = font_directory / f"Pretendard-{weight}.otf"
            if QFontDatabase.addApplicationFont(str(font_path)) == -1:
                raise RuntimeError(f"글꼴을 불러올 수 없습니다: {font_path}")
    QApplication.instance().setFont(QFont("Pretendard", 10))


class TrackingWindow(QWidget):
    def __init__(self, camera_index, device, colors, reset_callback):
        super().__init__()
        _apply_pretendard()
        self.setFont(QApplication.font())
        self._colors = [QColor(red, green, blue) for blue, green, red in colors]
        self._reset_callback = reset_callback
        self._aliases = {}
        self._items = {}
        self._visible_ids = set()
        self._absence_durations: dict[int, float] = {}
        self._absence_warnings: dict[int, float] = {}
        self._visible_tracks = []
        self._selected_id = None
        self._diagnostic_raw_id = None
        self._raw_reports = {}
        self._identity_reports = {}
        self._visible_raw_ids = set()
        self._frame_times = deque(maxlen=30)
        self.setObjectName("trackingWindow")
        self.setWindowTitle("실시간 사람 추적")
        self.setMinimumSize(730, 515)
        self.resize(1190, 730)
        self.setStyleSheet(STYLE)

        title = QLabel("실시간 사람 추적")
        title.setObjectName("title")
        self._status_prefix = f"카메라 {camera_index}   ·   {str(device).upper()}"
        self.status = QLabel(f"{self._status_prefix}   ·   0.0 FPS")
        self.status.setProperty("detail", True)
        self.status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        widest_status = f"{self._status_prefix}   ·   999.9 FPS"
        self.status.setMinimumWidth(self.status.fontMetrics().horizontalAdvance(widest_status) + 4)
        header = QHBoxLayout()
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.status)

        self.video = _VideoView()
        self.video.setObjectName("cameraView")
        self.video.person_selected.connect(self._select_raw)
        content = QHBoxLayout()
        content.setSpacing(14)
        content.addWidget(self.video, 1)
        content.addWidget(self._sidebar())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 18)
        layout.setSpacing(14)
        layout.addLayout(header)
        layout.addLayout(content, 1)

        self._quit_shortcuts = [QShortcut(QKeySequence(key), self) for key in ("Q", "Esc")]
        for shortcut in self._quit_shortcuts:
            shortcut.activated.connect(self.close)
        self.name_edit.installEventFilter(self)
        self._selection_changed(None, None)
        self._refresh_absence_warnings()

    def _sidebar(self):
        panel = QFrame()
        panel.setObjectName("sidebar")
        panel.setFixedWidth(320)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        visible_card, self.visible_count = self._count_card("현재 인원", "visibleCount")
        visible_card.setToolTip("현재 인원에는 번호 확인 중인 사람도 포함됩니다.")
        layout.addWidget(visible_card)
        cards = QHBoxLayout()
        cards.setSpacing(8)
        cumulative_card, self.cumulative_count = self._count_card("누적 인원", "cumulativeCount")
        cards.addWidget(cumulative_card, 1)
        self.absence_card = absence_card = QFrame()
        absence_card.setObjectName("absenceCard")
        absence_layout = QVBoxLayout(absence_card)
        absence_layout.setContentsMargins(11, 8, 11, 8)
        absence_layout.setSpacing(2)
        absence_title = QLabel("자리 비움 경고")
        absence_title.setProperty("metricCaption", True)
        absence_layout.addWidget(absence_title)
        self.absence_count = _MetricLabel("0")
        self.absence_count.setObjectName("absenceValue")
        self.absence_count.setAccessibleName("자리 비움 경고 인원")
        absence_layout.addWidget(self.absence_count)
        cards.addWidget(absence_card, 1)
        layout.addLayout(cards)

        self.absence_threshold = _ThresholdSpinBox()
        self.absence_threshold.setRange(1, 600)
        self.absence_threshold.setValue(60)
        self.absence_threshold.setSuffix("초")
        self.absence_threshold.setKeyboardTracking(False)
        self.absence_threshold.setAccessibleName("자리 비움 경고 기준 시간")
        self.absence_threshold.setToolTip("이 시간 동안 화면에 보이지 않는 등록 인원을 경고합니다.")
        self.absence_threshold.valueChanged.connect(self._refresh_absence_warnings)

        self.id_list = QListWidget()
        self.id_list.setObjectName("identityList")
        self.id_list.setItemDelegate(_IdentityItemDelegate(self.id_list))
        self.id_list.setMouseTracking(True)
        self.id_list.setAccessibleName("등록된 사람 ID 목록")
        self.id_list.setUniformItemSizes(True)
        self.id_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.id_list.currentItemChanged.connect(self._selection_changed)
        self.id_list.itemClicked.connect(self._diagnose_selected_identity)
        self.diagnostics = self._diagnostic_panel()
        self.id_list.setFixedHeight(180)
        self.diagnostics.setMinimumHeight(300)
        people_panel = QWidget()
        people_layout = QVBoxLayout(people_panel)
        people_layout.setContentsMargins(0, 0, 0, 0)
        people_layout.setSpacing(12)
        people_layout.addWidget(self.id_list)
        self.people_section = _Disclosure("등록된 사람", people_panel)
        self.diagnostic_section = _Disclosure("ID 진단", self.diagnostics)
        self.diagnostic_section.header.toggled.connect(self._refresh_diagnostics)
        details = QWidget()
        details.setObjectName("sidebarDetails")
        details_layout = QVBoxLayout(details)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(6)
        details_layout.addWidget(self.people_section)
        details_layout.addWidget(self.diagnostic_section)
        absence_settings = QWidget()
        absence_settings_layout = QHBoxLayout(absence_settings)
        absence_settings_layout.setContentsMargins(4, 6, 4, 6)
        absence_settings_label = QLabel("화면 미검출 시간")
        absence_settings_label.setProperty("detail", True)
        absence_settings_layout.addWidget(absence_settings_label, 1)
        absence_settings_layout.addWidget(self.absence_threshold)
        self.absence_section = _Disclosure("자리 비움 경고 설정", absence_settings)
        details_layout.addWidget(self.absence_section)
        self.name_panel = QWidget()
        name_layout = QVBoxLayout(self.name_panel)
        name_layout.setContentsMargins(0, 4, 0, 0)
        name_layout.setSpacing(8)
        self.name_title = QLabel("표시 이름")
        self.name_title.setProperty("section", True)
        name_layout.addWidget(self.name_title)
        self.name_edit = QLineEdit()
        self.name_edit.setObjectName("nameEdit")
        self.name_edit.setAccessibleName("선택한 사람 표시 이름")
        self.name_edit.setPlaceholderText("표시할 이름 입력")
        self.name_edit.setMaxLength(36)
        self.name_edit.returnPressed.connect(self._save_name)
        self.save_button = QPushButton("저장")
        self.save_button.setObjectName("saveNameButton")
        self.save_button.setAccessibleName("표시 이름 저장")
        self.save_button.clicked.connect(self._save_name)
        input_group = QFrame()
        input_group.setObjectName("nameInputGroup")
        input_group.setFixedHeight(42)
        editor = QHBoxLayout(input_group)
        editor.setContentsMargins(0, 4, 4, 4)
        editor.setSpacing(0)
        self.name_edit.setMinimumWidth(0)
        self.name_edit.setFixedHeight(32)
        self.save_button.setFixedSize(50, 32)
        editor.addWidget(self.name_edit, 1)
        editor.addWidget(self.save_button)
        name_layout.addWidget(input_group)
        people_layout.addWidget(self.name_panel)
        details_layout.addStretch()
        scroll = QScrollArea()
        scroll.setObjectName("detailsScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(details)
        layout.addWidget(scroll, 1)
        self.reset_button = QPushButton()
        self.reset_button.setObjectName("resetButton")
        self.reset_button.setIcon(_reset_icon())
        self.reset_button.setIconSize(QSize(22, 22))
        self.reset_button.setFixedSize(36, 36)
        self.reset_button.setToolTip("라벨 및 인원 기록 초기화")
        self.reset_button.setAccessibleName("라벨 및 인원 기록 초기화")
        self.reset_button.clicked.connect(self._confirm_reset)
        reset_row = QHBoxLayout()
        reset_row.addStretch()
        reset_row.addWidget(self.reset_button)
        layout.addLayout(reset_row)
        return panel

    def _diagnostic_panel(self):
        panel = QWidget()
        panel.setObjectName("identityDiagnostics")
        panel.setAccessibleName("선택한 사람의 ID 비교")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.diagnostic_target = QLabel("—")
        self.diagnostic_target.setProperty("section", True)
        self.diagnostic_target.setTextFormat(Qt.TextFormat.PlainText)
        self.diagnostic_target.setMinimumWidth(0)
        self.diagnostic_state = QLabel("선택 대기")
        self.diagnostic_state.setProperty("detail", True)
        self.diagnostic_state.setFixedWidth(68)
        self.diagnostic_state.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        target_row = QHBoxLayout()
        target_row.addWidget(self.diagnostic_target, 1)
        target_row.addWidget(self.diagnostic_state)
        layout.addLayout(target_row)

        self.diagnostic_source = _ComparisonComboBox()
        self.diagnostic_source.setItemDelegate(_NoFocusItemDelegate(self.diagnostic_source))
        self.diagnostic_source.setAccessibleName("비교 시점")
        self.diagnostic_source.addItems(["최근 비교", "번호 결정"])
        self.diagnostic_source.currentIndexChanged.connect(self._refresh_diagnostics)
        self.diagnostic_transition = QLabel("—")
        self.diagnostic_transition.setProperty("detail", True)
        self.diagnostic_transition.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        source_row = QHBoxLayout()
        source_row.addWidget(self.diagnostic_source)
        source_row.addWidget(self.diagnostic_transition, 1)
        layout.addLayout(source_row)

        table = QTableWidget(6, 5)
        self.diagnostic_table = table
        table.setObjectName("diagnosticScoreTable")
        table.setAccessibleName("ID 후보별 얼굴 외형 부위 점수")
        table.setHorizontalHeaderLabels(["ID", "얼굴", "외형", "부위", "판정"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().hide()
        table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        table.verticalHeader().setDefaultSectionSize(26)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        table.setAlternatingRowColors(True)
        table.setShowGrid(False)
        table.horizontalHeaderItem(4).setToolTip("얼굴 우선과 후보 제외 규칙을 반영한 점수")
        for row in range(table.rowCount()):
            for column in range(table.columnCount()):
                item = QTableWidgetItem("—")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(row, column, item)
        layout.addWidget(table, 1)
        legend = QLabel("—  점수 없음")
        legend.setProperty("detail", True)
        layout.addWidget(legend)
        self.diagnostic_face_status = QLabel("얼굴 · 미측정")
        self.diagnostic_face_status.setProperty("detail", True)
        self.diagnostic_face_status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.diagnostic_face_status)
        return panel

    @staticmethod
    def _count_card(title, object_name):
        card = QFrame()
        card.setObjectName(f"{object_name}Card")
        label = QLabel(title)
        label.setProperty("metricCaption", True)
        value = _MetricLabel("0")
        value.setObjectName(object_name)
        value.setAccessibleName("현재 화면 인원 수" if title == "현재 인원" else title)
        value.setProperty("metric", True)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(11, 8, 11, 8)
        layout.setSpacing(1)
        layout.addWidget(label)
        layout.addWidget(value)
        return card, value

    def eventFilter(self, watched, event):
        if watched is self.name_edit and event.type() in (QEvent.Type.FocusIn, QEvent.Type.FocusOut):
            for shortcut in self._quit_shortcuts:
                shortcut.setEnabled(event.type() == QEvent.Type.FocusOut)
        return super().eventFilter(watched, event)

    def show_frame(self, frame, observations, assignments, identity_ids):
        self._visible_ids = {item.person_id for item in assignments if item.person_id is not None}
        self._visible_tracks = [(observation.box, item.raw_id, item.person_id, item.status == "reidentified")
                                for observation, item in zip(observations, assignments)]
        self._record_diagnostics(assignments)
        identity_ids = set(identity_ids)
        self.visible_count.setText(str(len(observations)))
        self.cumulative_count.setText(str(len(identity_ids)))
        for person_id in self._items.keys() - identity_ids:
            item = self._items.pop(person_id)
            self._identity_reports.pop(person_id, None)
            self.id_list.takeItem(self.id_list.row(item))
        for person_id in sorted(identity_ids - self._items.keys()):
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, person_id)
            self._items[person_id] = item
            self.id_list.addItem(item)
        self._update_list_labels()
        self._sync_name_from_raw()
        self._sync_video_selection()
        self.video.set_frame(frame, self._draw_tracks())
        self._refresh_diagnostics()
        self._frame_times.append(time.monotonic())
        elapsed = self._frame_times[-1] - self._frame_times[0]
        fps = (len(self._frame_times) - 1) / elapsed if elapsed else 0.0
        self.status.setText(f"{self._status_prefix}   ·   {fps:.1f} FPS")

    def _label(self, person_id):
        if person_id is None:
            return "Checking"
        alias = self._aliases.get(person_id)
        return f"P{person_id:03d}  {alias}" if alias else f"P{person_id:03d}"

    def _draw_tracks(self):
        return [(box, raw_id, self._label(person_id) + ("  matched" if matched else ""),
                 self._colors[(person_id - 1) % len(self._colors)] if person_id is not None
                 else QColor("#b7c1ce")) for box, raw_id, person_id, matched in self._visible_tracks]

    def _record_diagnostics(self, assignments):
        self._visible_raw_ids = {item.raw_id for item in assignments}
        for item in assignments:
            previous = self._raw_reports.pop(item.raw_id, None)
            comparison = previous.comparison if previous else None
            decision = previous.decision if previous else None
            face_comparison = previous.face_comparison if previous else None
            if (face_comparison is not None and face_comparison.person_id is not None
                    and item.person_id is not None and face_comparison.person_id != item.person_id):
                face_comparison = None
            diagnostic = getattr(item, "diagnostic", None)
            if diagnostic is not None:
                if diagnostic.sampled:
                    comparison = item
                if item.status in ("new", "reidentified"):
                    decision = item
                if diagnostic.face_sampled or diagnostic.face_used:
                    face_comparison = item
            record = _DiagnosticRecord(item, comparison, decision, face_comparison)
            self._raw_reports[item.raw_id] = record
            if item.person_id is not None:
                self._identity_reports[item.person_id] = record

        # Keep recent unassigned tracks available without accumulating every
        # raw ID in a long session. Registered IDs keep their own last record.
        for raw_id in tuple(self._raw_reports):
            if len(self._raw_reports) <= 256:
                break
            if raw_id not in self._visible_raw_ids and raw_id != self._diagnostic_raw_id:
                del self._raw_reports[raw_id]

    def _selected_report(self):
        if self._diagnostic_raw_id is not None:
            return (self._raw_reports.get(self._diagnostic_raw_id),
                    self._diagnostic_raw_id in self._visible_raw_ids)
        return self._identity_reports.get(self._selected_id), self._selected_id in self._visible_ids

    def _sync_video_selection(self):
        report, visible = self._selected_report()
        self.video.select(report.current.raw_id if report and visible else None)

    def _refresh_diagnostics(self):
        if not self.diagnostic_section.header.isChecked():
            return
        record, visible = self._selected_report()
        current = record.current if record is not None else None
        target = self._label(current.person_id) if current is not None else "—"
        target_width = max(1, self.diagnostic_target.width())
        self.diagnostic_target.setText(self.diagnostic_target.fontMetrics().elidedText(
            target, Qt.TextElideMode.ElideRight, target_width))
        if current is None:
            state = "선택 대기"
        elif not visible:
            state = "화면 밖"
        else:
            state = STATUS_LABELS.get(current.status, current.status)
        self.diagnostic_state.setText(state)

        decision_mode = self.diagnostic_source.currentIndex() == 1
        snapshot = None
        face_snapshot = None
        if record is not None:
            snapshot = record.decision if decision_mode else record.comparison
            face_snapshot = snapshot if decision_mode else record.face_comparison
        transition = "—"
        candidates = {}
        if snapshot is not None:
            diagnostic = snapshot.diagnostic
            candidates = {candidate.person_id: candidate for candidate in diagnostic.candidates}
            if decision_mode:
                previous = diagnostic.previous_person_id
                before = f"P{previous:03d}" if previous is not None else "—"
                transition = f"{before} → P{snapshot.person_id:03d}"
            else:
                transition = f"임시 #{snapshot.raw_id}"
        self.diagnostic_transition.setText(transition)

        face_candidates = {}
        face_status = "미측정"
        if face_snapshot is not None:
            face_diagnostic = face_snapshot.diagnostic
            face_candidates = {candidate.person_id: candidate for candidate in face_diagnostic.candidates
                               if candidate.face is not None}
            face_status = face_diagnostic.face_status.removeprefix("얼굴 ") or (
                "비교 완료" if face_candidates else "특징 없음" if face_diagnostic.face_sampled else "미측정")
            if visible and not decision_mode and current.diagnostic is not None:
                age = max(0.0, current.diagnostic.timestamp - face_diagnostic.timestamp)
                if age >= 0.1:
                    face_status += f" · {age:.1f}초 전"
        if not self._visible_raw_ids and not decision_mode:
            face_status = "사람 박스 없음"
        face_label = f"얼굴 · {face_status}"
        self.diagnostic_face_status.setText(self.diagnostic_face_status.fontMetrics().elidedText(
            face_label, Qt.TextElideMode.ElideRight, max(1, self.diagnostic_face_status.width())))
        self.diagnostic_face_status.setToolTip(face_label)

        candidate_ids = list(candidates)
        candidate_ids.extend(person_id for person_id in face_candidates if person_id not in candidates)
        table = self.diagnostic_table
        for row in range(table.rowCount()):
            values = ("—",) * table.columnCount()
            tooltip = ""
            matched = False
            candidate = None
            if row < len(candidate_ids):
                person_id = candidate_ids[row]
                candidate = candidates.get(person_id)
                face_candidate = face_candidates.get(person_id)
                scores = (face_candidate.face if face_candidate else None,
                          candidate.body if candidate else None,
                          candidate.parts if candidate else None,
                          candidate.score if candidate else None)
                values = (f"P{person_id:03d}",) + tuple(
                    f"{score:.3f}" if score is not None else "—" for score in scores)
                details = [self._label(person_id)]
                if candidate is not None:
                    details.extend(candidate.notes)
                    details.extend(f"{PART_LABELS[index]} {score:.3f}"
                                   for index, score in candidate.part_details if 0 <= index < len(PART_LABELS))
                    matched = person_id == snapshot.person_id and candidate.score is not None
                if face_candidate is not None and face_snapshot is not snapshot:
                    details.append("얼굴 점수는 직전 얼굴 표본의 값입니다.")
                tooltip = "<qt>" + "<br>".join(escape(detail) for detail in details) + "</qt>"
            background = QBrush(QColor("#30304e")) if matched else QBrush()
            for column, value in enumerate(values):
                item = table.item(row, column)
                item.setText(value)
                item.setToolTip(tooltip)
                item.setBackground(background)
                color = "#8494a9" if value == "—" else "#e6edf7"
                if candidate is not None and column == 4 and candidate.score is None and candidate.notes:
                    color = "#ffb4a9"
                item.setForeground(QColor(color))

    def _update_list_labels(self):
        for person_id, item in self._items.items():
            visible = person_id in self._visible_ids
            state = "화면 안" if visible else "화면 밖"
            elapsed = None if visible else self._absence_durations.get(person_id)
            warning = not visible and person_id in self._absence_warnings
            if elapsed is not None:
                minutes, seconds = divmod(int(elapsed), 60)
                state = f'{"확인 필요" if warning else state} ({minutes:02d}:{seconds:02d})'
            label = f"{self._label(person_id)}  ·  {state}"
            if item.text() != label:
                item.setText(label)
                item.setToolTip(label)
                item.setForeground(QBrush(QColor("#f294a9")) if warning else QBrush())

    def update_absences(self, timestamp, last_seen):
        self._absence_durations = {person_id: max(0.0, timestamp - seen_at)
                                   for person_id, seen_at in last_seen.items()
                                   if person_id in self._items and person_id not in self._visible_ids}
        self._refresh_absence_warnings()

    def _refresh_absence_warnings(self):
        threshold = self.absence_threshold.value()
        self._absence_warnings = {person_id: elapsed for person_id, elapsed in self._absence_durations.items()
                                  if elapsed >= threshold}
        self.absence_count.setText(str(len(self._absence_warnings)))
        self.absence_card.setToolTip(f"화면에 {threshold}초 이상 보이지 않는 등록 인원")
        warning = bool(self._absence_warnings)
        if self.absence_card.property("warning") != warning:
            self.absence_card.setProperty("warning", warning)
            for widget in (self.absence_card, self.absence_count):
                widget.style().unpolish(widget)
                widget.style().polish(widget)
                widget.update()
        self._update_list_labels()

    def _select_id(self, person_id):
        item = self._items.get(person_id)
        if item is not None:
            self.id_list.setCurrentItem(item)
            self.id_list.scrollToItem(item)

    def _select_raw(self, raw_id):
        record = self._raw_reports.get(raw_id)
        if record is None:
            return
        person_id = record.current.person_id
        if person_id is None:
            self.id_list.setCurrentRow(-1)
        else:
            self._select_id(person_id)
        self._diagnostic_raw_id = raw_id
        self._sync_name_from_raw()
        self._sync_video_selection()
        self.diagnostic_section.header.setChecked(True)
        self._refresh_diagnostics()
        self.diagnostic_table.scrollToTop()

    def _diagnose_selected_identity(self):
        self._diagnostic_raw_id = None
        self._sync_video_selection()
        self._refresh_diagnostics()
        self.diagnostic_table.scrollToTop()

    def _selection_changed(self, current, previous):
        self._set_name_target(current.data(Qt.ItemDataRole.UserRole) if current else None)
        self._diagnose_selected_identity()

    def _set_name_target(self, person_id):
        self._selected_id = person_id
        self._set_name_controls()
        self.name_edit.setText(self._aliases.get(person_id, ""))

    def _set_name_controls(self, *, checking=False):
        enabled = self._selected_id is not None and not checking
        title = f"P{self._selected_id:03d} 이름 변경" if enabled else "표시 이름"
        self.name_title.setText("번호 확인 중" if checking else title)
        self.name_panel.setVisible(enabled or checking)
        self.name_edit.setEnabled(enabled)
        self.save_button.setEnabled(enabled)

    def _sync_name_from_raw(self):
        if self._diagnostic_raw_id not in self._visible_raw_ids:
            return
        person_id = self._raw_reports[self._diagnostic_raw_id].current.person_id
        if person_id is None:
            self._set_name_controls(checking=True)
        elif person_id != self._selected_id:
            with QSignalBlocker(self.id_list):
                self._select_id(person_id)
            self._set_name_target(person_id)
        elif not self.name_edit.isEnabled():
            self._set_name_controls()

    def _save_name(self):
        if not self.name_edit.isEnabled() or self._selected_id not in self._items:
            return
        alias = " ".join(self.name_edit.text().split())
        if alias:
            self._aliases[self._selected_id] = alias
        else:
            self._aliases.pop(self._selected_id, None)
        self.name_edit.setText(alias)
        self._update_list_labels()
        self.video.set_tracks(self._draw_tracks())
        self._refresh_diagnostics()

    def _confirm_reset(self):
        dialog = _ResetDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        if self._reset_callback() is False:
            return
        self.clear_session()

    def clear_session(self):
        self._aliases.clear()
        self._items.clear()
        self._visible_ids.clear()
        self._visible_tracks.clear()
        self._diagnostic_raw_id = None
        self._visible_raw_ids.clear()
        self._raw_reports.clear()
        self._identity_reports.clear()
        self.diagnostic_source.setCurrentIndex(0)
        self._frame_times.clear()
        self.id_list.clear()
        self._absence_durations.clear()
        self._refresh_absence_warnings()
        self._selection_changed(None, None)
        self.visible_count.setText("0")
        self.cumulative_count.setText("0")
        self.status.setText(f"{self._status_prefix}   ·   0.0 FPS")
        self.video.clear()
