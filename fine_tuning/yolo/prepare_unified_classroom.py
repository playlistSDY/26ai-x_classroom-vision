"""Build a person-only upper-body dataset from inherited and reviewed labels.

Never promote detector proposals to ground truth: the second source requires an
explicit, complete review ledger. All source files and split assignments survive.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import yaml

from run import ROOT, IMAGE_SUFFIXES, check_split

POLICY = {
    'version': 'visible-upper-body-v1',
    'class': '0: person',
    'target': 'One visible head and shoulders/torso belonging to one physical person',
    'extent': 'Head and visible upper torso; stop at desk occlusion or waist; never infer hidden limbs',
    'exclude': 'Desk/group boxes, face-only boxes, detached hands/arms, duplicate boxes',
    'ambiguity': 'Exclude the whole image when target people or boundaries cannot be verified',
    'face': 'Face evidence remains a separate YuNet/SFace inference path, not an additional count',
}


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_boxes(boxes):
    if not isinstance(boxes, list):
        raise ValueError('Accepted records need an explicit box list (empty only for reviewed backgrounds)')
    for box in boxes:
        if (not isinstance(box, list) or len(box) != 4
                or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box)):
            raise ValueError('Expected four finite normalized xyxy coordinates')
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError(f'Invalid normalized box: {box}')
    if len({tuple(box) for box in boxes}) != len(boxes):
        raise ValueError('Duplicate box in review ledger')
    return boxes


def read_yolo(label):
    boxes = []
    for row in label.read_text().splitlines():
        fields = row.split()
        if len(fields) != 5:
            raise ValueError(f'Expected YOLO detect labels: {label}')
        c, x, y, w, h = map(float, fields)
        if c != 0:
            raise ValueError(f'Inherited labels must already be person-only: {label}')
        boxes.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
    return validate_boxes(boxes)


def reviewed_records(source, ledger):
    records = json.loads(ledger.read_text())
    if not isinstance(records, dict):
        raise ValueError('Review ledger must be a mapping')
    images = {p.relative_to(source).as_posix(): p for p in source.glob('*/images/*')
              if p.suffix.lower() in IMAGE_SUFFIXES}
    paths = [r.get('path') for r in records.values()]
    if len(paths) != len(set(paths)) or set(paths) != set(images):
        raise ValueError('Every source image needs exactly one explicit review decision')
    for record in records.values():
        path = record['path']
        if record.get('status') == 'accept':
            if record.get('reason'):
                raise ValueError(f'Accepted record has an exclusion reason: {path}')
            if record.get('origin') not in {'manual_visual_review', 'model_assisted_visually_reviewed',
                                            'original_boxes_visually_reviewed'}:
                raise ValueError(f'Unreviewed proposals cannot be exported: {path}')
            validate_boxes(record.get('boxes'))
        elif record.get('status') == 'exclude':
            if not record.get('reason'):
                raise ValueError(f'Exclusion reason required: {path}')
        else:
            raise ValueError(f'Pending or unknown decision cannot be exported: {path}')
        yield images[path], record


def prepare(inherited, shared, ledger, output):
    inherited, shared, ledger, output = [p.expanduser().resolve() for p in (inherited, shared, ledger, output)]
    if output.exists():
        raise ValueError(f'Existing output is protected: {output}')
    if any(output == source or source in output.parents for source in (inherited, shared)):
        raise ValueError('Output must be outside source folders')
    manifest = {'policy': POLICY, 'review_ledger_sha256': sha256(ledger),
                'inherited_manifest_sha256': sha256(inherited / 'preparation.json'),
                'review_scope': {'dodzk': 'Prior class8/9 + uncovered-head filter inherited; 12-image visual spot check on 2026-11-19, not full relabeling',
                                 'shared': '450-image visual screening, accepted non-original boxes inspected again; conservative image exclusions'},
                'limitations': ['Adjacent frames and the same rooms occur across original splits; no scene-independent accuracy claim',
                                'Visual screening and model-assisted review are not an independently annotated gold standard',
                                'Excluded images were not silently made background images'],
                'records': [], 'excluded': [], 'splits': {}}
    items = []
    for split in ('train', 'valid', 'test'):
        images = sorted(p for p in (inherited / split / 'images').glob('*') if p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise ValueError(f'Missing inherited split: {split}')
        for image in images:
            label = inherited / split / 'labels' / (image.stem + '.txt')
            items.append(('dodzk', split, image, read_yolo(label), 'inherited_preparation', sha256(label)))
    for image, record in reviewed_records(shared, ledger):
        split = image.relative_to(shared).parts[0]
        if split not in ('train', 'valid', 'test'):
            raise ValueError(f'Unknown split: {split}')
        if record['status'] == 'exclude':
            manifest['excluded'].append({'source': 'shared', **record})
        else:
            label = shared / split / 'labels' / (image.stem + '.txt')
            items.append(('shared', split, image, record['boxes'], record['origin'], sha256(label)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.unified-prepare-', dir=output.parent) as temporary:
        stage = Path(temporary)
        hashes = {}
        counts = {}
        for source, split, image, boxes, origin, label_sha in items:
            digest = sha256(image)
            if digest in hashes:
                raise ValueError(f'Exact image duplicate: {image} and {hashes[digest]}')
            hashes[digest] = str(image)
            name = source + '__' + image.name
            target = stage / split / 'images' / name
            label = stage / split / 'labels' / (Path(name).stem + '.txt')
            target.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(image, target)
            except OSError:
                shutil.copy2(image, target)
            lines = []
            for x1, y1, x2, y2 in boxes:
                lines.append(f'0 {(x1+x2)/2:.9f} {(y1+y2)/2:.9f} {x2-x1:.9f} {y2-y1:.9f}')
            label.write_text('\n'.join(lines) + ('\n' if lines else ''))
            counter = counts.setdefault(split, {}).setdefault(source, Counter(images=0, boxes=0, backgrounds=0))
            counter.update(images=1, boxes=len(boxes), backgrounds=int(not boxes))
            manifest['records'].append({'source': source, 'source_image': str(image),
                                        'output_image': f'{split}/images/{name}', 'split': split,
                                        'image_sha256': digest, 'source_label_sha256': label_sha,
                                        'output_label_sha256': sha256(label), 'boxes': len(boxes), 'annotation_origin': origin})
        data = {'path': str(output), 'train': 'train/images', 'val': 'valid/images',
                'test': 'test/images', 'names': {0: 'person'}}
        for split in ('train', 'val', 'test'):
            check_split({**data, 'path': str(stage)}, split)
        manifest['splits'] = {split: {source: dict(n) for source, n in per_source.items()}
                              for split, per_source in counts.items()}
        manifest['exact_duplicate_groups'] = 0
        (stage / 'data.yaml').write_text(yaml.safe_dump(data, sort_keys=False))
        (stage / 'review-decisions.json').write_bytes(ledger.read_bytes())
        (stage / 'preparation.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        if output.exists():
            raise ValueError(f'Existing output is protected: {output}')
        stage.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inherited', type=Path, default=ROOT / 'data/roboflow-classroom-dodzk/person')
    parser.add_argument('--shared', type=Path, default=ROOT / 'data/shared-classroom-v1/raw')
    parser.add_argument('--ledger', type=Path, default=ROOT / 'data/upper-body-unified-v1-review/decisions.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/upper-body-unified-v2')
    args = parser.parse_args()
    try:
        result = prepare(args.inherited, args.shared, args.ledger, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'Preparation failed: {error}\n')
    print(json.dumps({'splits': result['splits'], 'excluded': len(result['excluded'])}, indent=2))


if __name__ == '__main__':
    main()
