"""Absence counts and state transitions in the actual sidebar."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PySide6.QtWidgets import QApplication
from tracking_window import TrackingWindow


class TrackingAbsenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = TrackingWindow(0, "CPU", [(100, 170, 220)], lambda: True)
        self.show_people([1])

    def tearDown(self):
        self.window.close()

    def show_people(self, ids):
        observations = [SimpleNamespace(box=(10, 10, 30, 50)) for _ in ids]
        assignments = [SimpleNamespace(person_id=i, raw_id=i, status="tracked", diagnostic=None)
                       for i in ids]
        self.window.show_frame(np.zeros((60, 60, 3), dtype=np.uint8), observations,
                               assignments, [1, 2, 3])

    def test_threshold_and_live_setting_changes(self):
        self.window.update_absences(100, {1: 100, 2: 40.1, 3: 40})
        self.assertEqual(self.window.absence_count.text(), "1")
        self.assertEqual(set(self.window._absence_warnings), {3})
        self.assertIn("확인 필요 (01:00)", self.window._items[3].text())
        self.window.absence_threshold.setValue(90)
        self.assertEqual(self.window.absence_count.text(), "0")
        self.assertFalse(self.window.absence_card.property("warning"))

    def test_reappearance_clears_warning(self):
        self.window.update_absences(100, {1: 100, 2: 0})
        self.show_people([1, 2])
        self.window.update_absences(110, {1: 110, 2: 110})
        self.assertEqual(self.window.absence_count.text(), "0")
        self.assertTrue(self.window._items[2].text().endswith("화면 안"))

    def test_reset_clears_count_and_timers(self):
        self.window.update_absences(100, {1: 100, 2: 0, 3: 0})
        self.window.clear_session()
        self.assertEqual(self.window.absence_count.text(), "0")
        self.assertFalse(self.window._absence_durations)
        self.assertFalse(self.window._absence_warnings)

    def test_unregistered_ids_are_not_counted(self):
        self.window.update_absences(100, {1: 100, 99: 0})
        self.assertEqual(self.window.absence_count.text(), "0")


if __name__ == "__main__":
    unittest.main()
