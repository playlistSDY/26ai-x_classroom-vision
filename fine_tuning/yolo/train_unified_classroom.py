"""Train the reviewed unified dataset and compare source-specific person tests."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import yaml

ROOT = Path(__file__).resolve().parents[2]
FINE = ROOT / 'fine_tuning'
CONTROL = FINE / 'runs/upper-body-unified-v2-20261119'
OUTPUT = FINE / 'runs/yolo/upper-body-unified-v2'
DATA = FINE / 'data/upper-body-unified-v2'
BASELINE = Path.home() / '.cache/aix-classroom-tracking/yolo26n.pt'
PREVIOUS = FINE / 'runs/yolo/classroom-dodzk-person/weights/best.pt'


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    if OUTPUT.exists() or CONTROL.exists():
        raise FileExistsError('Existing experiment is protected; choose a new experiment name')
    CONTROL.mkdir(parents=True)
    started = time.monotonic()
    state = {'status': 'starting', 'started_at': datetime.now(timezone.utc).isoformat(),
             'dataset_manifest_sha256': sha256(DATA / 'preparation.json'),
             'review_ledger_sha256': sha256(DATA / 'review-decisions.json'),
             'train_config': 'configs/upper-body-unified-v2-train.yaml',
             'train_output': str(OUTPUT), 'imgsz': 640,
             'baseline_sha256': sha256(BASELINE), 'previous_person_sha256': sha256(PREVIOUS),
             'test_metrics': {},
             'limitations': ['Original same-room adjacent-frame splits are retained',
                             'Shared labels include model-assisted visual review, not an independent gold standard',
                             'Detection evaluation only; ID tracking and real installation video not evaluated']}
    def save():
        state['elapsed_seconds'] = time.monotonic() - started
        pending = CONTROL / '.status.tmp'
        pending.write_text(json.dumps(state, indent=2) + '\n')
        pending.replace(CONTROL / 'status.json')
    try:
        state['status'] = 'training'
        save()
        subprocess.run([sys.executable, '-u', 'fine_tuning/yolo/run.py', 'train',
                        '--config', state['train_config'], '--device', '0'], cwd=ROOT, check=True)
        best = OUTPUT / 'weights/best.pt'
        state.update(status='evaluating', best_weights=str(best), trained_sha256=sha256(best))
        save()
        import torch
        import ultralytics
        from ultralytics import YOLO
        state['environment'] = {'torch': torch.__version__, 'ultralytics': ultralytics.__version__,
                                'gpu': torch.cuda.get_device_name(0)}
        manifest = json.loads((DATA / 'preparation.json').read_text())
        evaluations = {}
        for source in ('dodzk', 'shared'):
            selected = [r for r in manifest['records'] if r['split'] == 'test' and r['source'] == source]
            images = CONTROL / f'{source}-test.txt'
            images.write_text(''.join(str(DATA / r['output_image']) + '\n' for r in selected))
            data_file = CONTROL / f'{source}-test.yaml'
            data_file.write_text(yaml.safe_dump({'path': str(DATA), 'train': str(DATA / 'train/images'),
                                                 'val': str(images), 'test': str(images), 'names': {0: 'person'}}))
            evaluations[source] = (data_file, len(selected), sum(r['boxes'] for r in selected))
        for name, weights in [('baseline', BASELINE), ('previous_person', PREVIOUS), ('unified', best)]:
            model = YOLO(str(weights), task='detect')
            for source, (data_file, count, boxes) in evaluations.items():
                state.update(evaluation_model=name, evaluation_source=source)
                save()
                result = model.val(data=str(data_file), split='test', device=0, classes=[0],
                                   imgsz=640, batch=4, workers=2, plots=True,
                                   project=str(CONTROL / 'evaluation'), name=f'{name}-{source}', exist_ok=False)
                state['test_metrics'].setdefault(source, {})[name] = {
                    'images': count, 'boxes': boxes,
                    **{k: float(v) for k, v in result.results_dict.items()},
                    'speed_ms': {k: float(v) for k, v in result.speed.items()}}
                save()
            del model
            torch.cuda.empty_cache()
        checkpoint = FINE / 'checkpoints/upper-body-unified-v2-best.pt'
        checkpoint.parent.mkdir(exist_ok=True)
        with best.open('rb') as src, checkpoint.open('xb') as dst:
            import shutil
            shutil.copyfileobj(src, dst)
        if sha256(checkpoint) != state['trained_sha256']:
            raise ValueError('Checkpoint copy hash mismatch')
        state.update(status='complete', checkpoint=str(checkpoint), finished_at=datetime.now(timezone.utc).isoformat())
        save()
        print('EXPERIMENT_COMPLETE', json.dumps(state['test_metrics']), flush=True)
    except BaseException as error:
        state.update(status='failed', error=str(error))
        save()
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
