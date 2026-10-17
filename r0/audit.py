"""Preflight checks for scene leakage, shape, target provenance, legal metadata."""
import argparse,json
from pathlib import Path
from .data import load_manifest

def audit(path,quality=False):
    rows=load_manifest(path);split_scenes={};origins={}
    for r in rows:
        scene=r.get('scene');split=r.get('split')
        if scene is None or split not in ('train','val','test'):
            raise ValueError('scene and split are mandatory')
        split_scenes.setdefault(split,set()).add(scene)
        origins[r['target_origin']]=origins.get(r['target_origin'],0)+1
        if quality:
            if r.get('usage_mode') == 'research' or r['target_origin'] != 'reviewed_teacher_output':
                raise RuntimeError('quality dataset includes research or unreviewed target')
            if not r.get('teacher_model_id') or not r.get('rights_reference'):
                raise RuntimeError('teacher provenance missing')
    if quality:
        for s1,scenes1 in split_scenes.items():
            for s2,scenes2 in split_scenes.items():
                if s1<s2 and scenes1&scenes2:
                    raise RuntimeError(f'leakage between {s1} and {s2}: {scenes1&scenes2}')
    return {'rows':len(rows),'splits':{k:len(v) for k,v in split_scenes.items()},
            'origins':origins,'quality_gate_passed':quality}

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--quality',action='store_true')
    a=p.parse_args();print(json.dumps(audit(a.manifest,a.quality),indent=2))
if __name__=='__main__':main()
