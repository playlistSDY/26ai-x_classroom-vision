"""Validate and train the shared ClassRoom v1 export without merging classes."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import traceback

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/shared-classroom-v1'
RAW = DATA / 'raw'
CONTROL = ROOT / 'runs/shared-classroom-v1-20261118'
OUTPUT = ROOT / 'runs/yolo/shared-classroom-v1'
BASELINE = Path.home() / '.cache/aix-classroom-tracking/yolo26n.pt'
CONFIG = ROOT / 'configs/shared-classroom-v1-train.yaml'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare():
    original = yaml.safe_load((RAW / 'data.yaml').read_text())
    assert original['names'] == ['Student', 'with-student']
    hashes = defaultdict(list)
    splits = {}
    for split in ('train', 'valid', 'test'):
        images = sorted((RAW / split / 'images').glob('*'))
        labels = RAW / split / 'labels'
        assert images and {p.stem for p in images} == {p.stem for p in labels.glob('*.txt')}
        counts = Counter()
        backgrounds = 0
        for image in images:
            hashes[digest(image)].append(str(image.relative_to(RAW)))
            rows = (labels / (image.stem + '.txt')).read_text().splitlines()
            backgrounds += not rows
            for row in rows:
                fields = row.split()
                assert len(fields) == 5, image
                c, x, y, w, h = map(float, fields)
                assert all(math.isfinite(v) for v in (c, x, y, w, h)), image
                assert c in (0, 1) and 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1, image
                counts[int(c)] += 1
        splits[split] = {'images': len(images), 'boxes_by_class': dict(counts), 'background_images': backgrounds}
    duplicates = [v for v in hashes.values() if len(v) > 1]
    assert not duplicates, duplicates
    manifest = {'source_url': 'https://app.roboflow.com/buenos-dias/classroom-o4dnp-k83mj/1',
                'source_metadata': original.get('roboflow'),
                'archive_sha256': digest(DATA / 'ClassRoom.v1i.yolo26.zip'),
                'names': original['names'], 'splits': splits,
                'exact_duplicate_groups': duplicates,
                'annotation_policy': 'Original two classes and boxes preserved; no person remapping',
                'limitation': 'Adjacent classroom frames occur across splits; scene independence is not established'}
    data = {'path': str(RAW), 'train': 'train/images', 'val': 'valid/images',
            'test': 'test/images', 'names': original['names'], 'nc': 2}
    (CONTROL / 'dataset.yaml').write_text(yaml.safe_dump(data))
    (CONTROL / 'preparation.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    if OUTPUT.exists() or CONTROL.exists():
        raise FileExistsError('Existing experiment is protected; select a new run name')
    CONTROL.mkdir(parents=True)
    state = {'status': 'preparing', 'started_at': datetime.now(timezone.utc).isoformat()}
    def save():
        pending = CONTROL / '.status.json.tmp'
        pending.write_text(json.dumps(state, indent=2) + '\n')
        pending.replace(CONTROL / 'status.json')
    try:
        save()
        state['dataset'] = prepare()
        state['baseline_sha256'] = digest(BASELINE)
        config = yaml.safe_load(CONFIG.read_text())
        state['train_config'] = config
        import torch
        import ultralytics
        from ultralytics import YOLO
        state['environment'] = {'torch': torch.__version__, 'ultralytics': ultralytics.__version__,
                                'gpu': torch.cuda.get_device_name(0)}
        state['status'] = 'training'
        save()
        model = YOLO(str(BASELINE), task='detect')
        model.train(data=str(CONTROL / 'dataset.yaml'), device=0,
                    project=str(OUTPUT.parent), name=OUTPUT.name, **config)
        del model
        torch.cuda.empty_cache()
        best = OUTPUT / 'weights/best.pt'
        state.update(status='evaluating', best_weights=str(best), trained_sha256=digest(best))
        save()
        model = YOLO(str(best))
        result = model.val(data=str(CONTROL / 'dataset.yaml'), split='test', device=0,
                           imgsz=640, batch=4, workers=2, plots=True,
                           project=str(CONTROL / 'evaluation'), name='test', exist_ok=False)
        state['test_metrics'] = {k: float(v) for k, v in result.results_dict.items()}
        state['per_class'] = {str(i): {'name': model.names[i],
                                      'precision': float(result.box.p[j]),
                                      'recall': float(result.box.r[j]),
                                      'map50': float(result.box.ap50[j]),
                                      'map50_95': float(result.box.ap[j])}
                              for j, i in enumerate(result.box.ap_class_index)}
        state['speed_ms'] = result.speed
        state['evaluation_scope'] = 'Original two-class detection on 46 exported test images, not person identity or tracking evaluation'
        state.update(status='complete', finished_at=datetime.now(timezone.utc).isoformat())
        save()
        print('EXPERIMENT_COMPLETE', json.dumps(state['test_metrics']), flush=True)
    except BaseException as error:
        state.update(status='failed', error=str(error))
        save()
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
