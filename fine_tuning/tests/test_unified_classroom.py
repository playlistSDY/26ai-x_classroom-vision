"""Safeguards for a reviewed, source-preserving dataset export."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'yolo'))
from prepare_unified_classroom import prepare, reviewed_records, validate_boxes


class UnifiedClassroomTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inherited = self.root / 'inherited'
        self.shared = self.root / 'shared'
        self.output = self.root / 'output'
        for split in ('train', 'valid', 'test'):
            for source in (self.inherited, self.shared):
                (source / split / 'images').mkdir(parents=True)
                (source / split / 'labels').mkdir()
                (source / split / 'images' / 'frame.jpg').write_bytes(f'{source.name}-{split}'.encode())
                (source / split / 'labels' / 'frame.txt').write_text(
                    '0 0.5 0.5 0.4 0.4\n' if source == self.inherited else '1 0.5 0.5 0.9 0.9\n')
        (self.inherited / 'preparation.json').write_text('{}')
        self.ledger = self.root / 'decisions.json'
        self.records = {
            split: {'path': f'{split}/images/frame.jpg', 'status': 'accept', 'reason': None,
                    'origin': 'manual_visual_review', 'boxes': [[0.3, 0.2, 0.6, 0.5]]}
            for split in ('train', 'valid', 'test')}
        self.save()

    def save(self):
        self.ledger.write_text(json.dumps(self.records))

    def build(self):
        return prepare(self.inherited, self.shared, self.ledger, self.output)

    def test_export_uses_reviewed_boxes_and_preserves_source_labels_and_splits(self):
        raw = (self.shared / 'train/labels/frame.txt').read_bytes()
        result = self.build()
        self.assertEqual((self.shared / 'train/labels/frame.txt').read_bytes(), raw)
        self.assertEqual(len(result['records']), 6)
        exported = (self.output / 'train/labels/shared__frame.txt').read_text().split()
        self.assertEqual(exported[0], '0')
        self.assertAlmostEqual(float(exported[3]), 0.3)
        self.assertEqual(result['splits']['test']['shared']['images'], 1)
        self.assertEqual(json.loads((self.output / 'review-decisions.json').read_text()), self.records)

    def test_excluded_image_is_not_exported_as_background(self):
        self.records['train'].update(status='exclude', reason='group_box', boxes=None)
        self.save()
        result = self.build()
        self.assertFalse((self.output / 'train/images/shared__frame.jpg').exists())
        self.assertEqual(len(result['excluded']), 1)

    def test_reviewed_background_requires_explicit_empty_list(self):
        self.records['train']['boxes'] = []
        self.save()
        result = self.build()
        self.assertEqual((self.output / 'train/labels/shared__frame.txt').read_text(), '')
        self.assertEqual(result['splits']['train']['shared']['backgrounds'], 1)
        with self.assertRaises(ValueError):
            validate_boxes(None)

    def test_missing_pending_or_proposal_only_review_is_rejected(self):
        for change in ('missing', 'pending', 'proposal'):
            with self.subTest(change=change):
                original = json.loads(json.dumps(self.records))
                if change == 'missing':
                    del self.records['train']
                elif change == 'pending':
                    self.records['train']['status'] = 'pending'
                else:
                    self.records['train']['origin'] = 'model_prediction'
                self.save()
                with self.assertRaises(ValueError):
                    list(reviewed_records(self.shared, self.ledger))
                self.records = original

    def test_nonfinite_inverted_out_of_frame_and_duplicate_boxes_are_rejected(self):
        for boxes in ([[0.5, 0.2, 0.3, 0.7]], [[0, 0, 1.1, 1]],
                      [[0, 0, float('nan'), 1]], [[0, 0, 1, 1], [0, 0, 1, 1]]):
            with self.subTest(boxes=boxes), self.assertRaises(ValueError):
                validate_boxes(boxes)

    def test_duplicate_images_across_splits_fail_without_partial_output(self):
        (self.shared / 'test/images/frame.jpg').write_bytes((self.shared / 'train/images/frame.jpg').read_bytes())
        with self.assertRaisesRegex(ValueError, 'Exact image duplicate'):
            self.build()
        self.assertFalse(self.output.exists())

    def test_existing_output_is_protected(self):
        self.output.mkdir()
        marker = self.output / 'keep.txt'
        marker.write_text('keep')
        with self.assertRaises(ValueError):
            self.build()
        self.assertEqual(marker.read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
