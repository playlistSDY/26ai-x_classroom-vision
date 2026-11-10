"""Filtering rendered boxes through the actual identity list."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from tracking_window import TrackingWindow


class TrackingSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = TrackingWindow(0, "CPU", [(100, 170, 220)], lambda: True)
        self.show_people([1, 2, None])
        self.window.people_section.header.click()
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        self.window.close()

    def show_people(self, ids):
        observations = [SimpleNamespace(box=(i * 50, 10, i * 50 + 40, 80))
                        for i in range(len(ids))]
        assignments = [SimpleNamespace(person_id=person_id, raw_id=i + 10,
                                       status="tracked", diagnostic=None)
                       for i, person_id in enumerate(ids)]
        self.window.show_frame(np.zeros((100, 200, 3), dtype=np.uint8),
                               observations, assignments, [1, 2, 3])

    def click_person(self, row):
        widget = self.window.id_list
        point = widget.visualItemRect(widget.item(row)).center()
        QTest.mouseClick(widget.viewport(), Qt.MouseButton.LeftButton, pos=point)

    def test_select_and_repeat_click_restore_all_boxes(self):
        self.assertEqual(len(self.window.video._tracks), 3)
        self.click_person(0)
        self.assertEqual(len(self.window.video._tracks), 1)
        self.assertEqual(self.window.video._tracks[0][1], 10)
        self.assertEqual(self.window.visible_count.text(), "3")
        self.assertEqual(self.window.cumulative_count.text(), "3")
        self.click_person(0)
        self.assertIsNone(self.window._selected_id)
        self.assertEqual(len(self.window.video._tracks), 3)

    def test_switch_person_and_blank_click(self):
        self.click_person(0)
        self.click_person(1)
        self.assertEqual(self.window.video._tracks[0][1], 11)
        widget = self.window.id_list
        widget.setFixedHeight(180)
        self.app.processEvents()
        QTest.mouseClick(widget.viewport(), Qt.MouseButton.LeftButton,
                         pos=QPoint(8, widget.viewport().height() - 5))
        self.assertEqual(len(self.window.video._tracks), 3)

    def test_filter_persists_across_frames_and_reappearance(self):
        self.click_person(0)
        self.show_people([2, None])
        self.assertEqual(self.window.video._tracks, [])
        self.show_people([2, 1, None])
        self.assertEqual(len(self.window.video._tracks), 1)
        self.assertEqual(self.window.video._tracks[0][1], 11)

    def test_reset_clears_filter(self):
        self.click_person(0)
        self.window.clear_session()
        self.show_people([1, 2, None])
        self.assertIsNone(self.window._selected_id)
        self.assertEqual(len(self.window.video._tracks), 3)


if __name__ == "__main__":
    unittest.main()
