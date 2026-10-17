"""Register 3D/teacher pairs with explicit research or production provenance.

Research mode removes the rights/QA *gate* for exploratory experiments, but
never marks unreviewed targets as production-approved. It does not fetch or
redistribute teacher weights or generated images.
"""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image
from .data import load_manifest


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def ingest(original, targets, rights=None, review=None, out=None,
           min_style=4, min_structure=4, mode='production', teacher_model_id=None):
    if mode not in ('research', 'production'):
        raise ValueError('mode must be research or production')
    if out is None:
        raise ValueError('out manifest path required')
    root = Path(original).resolve().parent
    targets = Path(targets)
    legal = json.loads(Path(rights).read_text()) if rights is not None else {}
    if mode == 'production':
        if not legal.get('outputs_can_train_independent_model'):
            raise PermissionError('training rights for generated outputs not confirmed')
        if not legal.get('rights_reference') or not legal.get('teacher_model_id'):
            raise ValueError('rights reference and teacher model ID are mandatory')
        if not review:
            raise ValueError('production mode needs independent quality review')
    model_id = teacher_model_id or legal.get('teacher_model_id') or 'unidentified_research_teacher'
    if mode == 'production' and model_id != legal['teacher_model_id']:
        raise ValueError('teacher ID must match production rights declaration')
    rows = load_manifest(original)
    reviews = ({r['id']: r for r in (json.loads(x) for x in Path(review).read_text().splitlines() if x.strip())}
               if review else {})
    staged, excluded = [], []
    for row in rows:
        sample_id = f"{row['scene']}_{int(row['frame']):04d}"
        image = targets / f'{sample_id}.png'
        qa = reviews.get(sample_id)
        if not image.is_file():
            excluded.append({'id': sample_id, 'reason': 'missing_target'})
            continue
        qa_valid = bool(qa and qa.get('reviewed') and qa.get('style', 0) >= min_style
                        and qa.get('structure', 0) >= min_structure)
        if mode == 'production' and not qa_valid:
            excluded.append({'id': sample_id, 'reason': 'failed_qa'})
            continue
        with Image.open(root / row['source']) as src, Image.open(image) as dst:
            if src.size != dst.size:
                excluded.append({'id': sample_id, 'reason': 'resolution_mismatch'})
                continue
        r = dict(row)
        # Data manifests can live in a new folder; make all source references absolute
        # so relocating a manifest cannot quietly break G-buffer lookup.
        for key in ('source', 'depth', 'normal', 'semantic', 'line', 'motion_to_prev', 'motion_valid'):
            if key in r:
                r[key] = str((root / r[key]).resolve())
        r['target'] = str(image.resolve())
        r['target_origin'] = ('reviewed_teacher_output' if mode == 'production' else
                              'research_teacher_reviewed' if qa_valid else 'research_teacher_unreviewed')
        r['target_sha256'] = sha256(image)
        r['teacher_model_id'] = model_id
        r['usage_mode'] = mode
        if mode == 'production':
            r['rights_reference'] = legal['rights_reference']
        if qa:
            r['quality_review'] = qa
        staged.append(r)
    if not staged:
        raise RuntimeError('zero accepted targets: check target naming, image sizes, and QA')
    scene_names = sorted({r['scene'] for r in staged})
    if len(scene_names) < 3:
        raise RuntimeError('Need >=3 independent scene groups for train/val/test')
    ordered = sorted(scene_names, key=lambda v: hashlib.sha256(v.encode()).digest())
    nval = max(1, len(ordered) // 10)
    ntest = max(1, len(ordered) // 10)
    groups = {name: ('val' if i < nval else 'test' if i < nval + ntest else 'train')
              for i, name in enumerate(ordered)}
    for row in staged:
        row['split'] = groups[row['scene']]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in staged))
    print(json.dumps({'mode': mode, 'accepted': len(staged), 'rejected': len(excluded),
                      'output': str(out), 'teacher_model_id': model_id}, indent=2))
    return {'accepted': len(staged), 'rejected': excluded}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--original', required=True)
    p.add_argument('--targets', required=True)
    p.add_argument('--rights')
    p.add_argument('--review')
    p.add_argument('--out', required=True)
    p.add_argument('--mode', choices=['research', 'production'], default='production')
    p.add_argument('--teacher-id')
    args = p.parse_args()
    ingest(args.original, args.targets, args.rights, args.review, args.out,
           mode=args.mode, teacher_model_id=args.teacher_id)


if __name__ == '__main__':
    main()
