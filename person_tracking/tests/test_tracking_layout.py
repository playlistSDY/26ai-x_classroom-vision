"""Ensure diagnostic footer text stays below its table on every Qt style."""

import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication, QLabel, QStyleFactory
from tracking_window import TrackingWindow


class TrackingLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_scrolled_list_ends_at_last_person(self):
        window = TrackingWindow(0, "CPU", [(100, 170, 220)], lambda: True)
        try:
            window.show_frame(np.zeros((50, 50, 3), dtype=np.uint8), [], [], range(1, 19))
            window.people_section.header.click()
            window.show()
            self.app.processEvents()
            window.id_list.scrollToBottom()
            self.app.processEvents()
            last_row = window.id_list.visualItemRect(window.id_list.item(17))
            gap = window.id_list.viewport().rect().bottom() - last_row.bottom()
            self.assertGreaterEqual(gap, 0)
            self.assertLessEqual(gap, 2)
        finally:
            window.close()

    def test_diagnostic_footer_never_overlaps_table(self):
        original_style = self.app.style().objectName()
        try:
            for style in QStyleFactory.keys():
                self.app.setStyle(style)
                window = TrackingWindow(0, "CPU", [(100, 170, 220)], lambda: True)
                try:
                    window.diagnostic_section.header.click()
                    window.show()
                    for size in ((730, 515), (1190, 730)):
                        with self.subTest(style=style, size=size):
                            window.resize(*size)
                            self.app.processEvents()
                            panel = window.diagnostics
                            legend = next(label for label in panel.findChildren(QLabel)
                                          if label.text() == "—  점수 없음")
                            legend_y = legend.mapTo(panel, QPoint(0, 0)).y()
                            face_y = window.diagnostic_face_status.mapTo(panel, QPoint(0, 0)).y()
                            self.assertGreater(legend_y, window.diagnostic_table.geometry().bottom() + 5)
                            self.assertGreaterEqual(face_y, legend_y + legend.height() + 8)
                            self.assertLessEqual(face_y + window.diagnostic_face_status.height(),
                                                 panel.height() - 12)
                finally:
                    window.close()
        finally:
            self.app.setStyle(original_style)


if __name__ == "__main__":
    unittest.main()
