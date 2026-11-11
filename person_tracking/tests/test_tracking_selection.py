"""Highlighting selected identities while retaining all rendered boxes."""

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
        self.window = TrackingWindow(0, "CPU", [(100, 170, 220), (220, 130, 90)], lambda: True)
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

    def box_colors(self):
        return {track[1]: track[3].name() for track in self.window.video._tracks}

    def test_select_and_repeat_click_restore_colors(self):
        original = self.box_colors()
        self.assertEqual(len(original), 3)
        self.click_person(0)
        self.assertEqual(self.box_colors(), {10: original[10], 11: "#868c96", 12: "#868c96"})
        self.assertEqual(self.window.visible_count.text(), "3")
        self.assertEqual(self.window.cumulative_count.text(), "3")
        self.click_person(0)
        self.assertIsNone(self.window._selected_id)
        self.assertEqual(self.box_colors(), original)

    def test_switch_person_and_blank_click(self):
        original = self.box_colors()
        self.click_person(0)
        self.click_person(1)
        self.assertEqual(self.box_colors(), {10: "#868c96", 11: original[11], 12: "#868c96"})
        widget = self.window.id_list
        widget.setFixedHeight(180)
        self.app.processEvents()
        QTest.mouseClick(widget.viewport(), Qt.MouseButton.LeftButton,
                         pos=QPoint(8, widget.viewport().height() - 5))
        self.assertEqual(self.box_colors(), original)

    def test_highlight_persists_across_frames_and_reappearance(self):
        self.click_person(0)
        self.show_people([2, None])
        self.assertEqual(self.box_colors(), {10: "#868c96", 11: "#868c96"})
        self.show_people([2, 1, None])
        self.assertEqual(len(self.window.video._tracks), 3)
        self.assertEqual(self.box_colors()[11], self.window._colors[0].name())
        self.assertEqual(self.box_colors()[10], "#868c96")
        self.assertEqual(self.box_colors()[12], "#868c96")

    def test_reset_clears_highlight(self):
        original = self.box_colors()
        self.click_person(0)
        self.window.clear_session()
        self.show_people([1, 2, None])
        self.assertIsNone(self.window._selected_id)
        self.assertEqual(self.box_colors(), original)


if __name__ == "__main__":
    unittest.main()
