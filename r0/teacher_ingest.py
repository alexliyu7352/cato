"""Safely register style-target pairs after rights AND independent QA review.

Expects pre-rendered source G-buffers and external teacher outputs.
Does NOT call a paid API, download weights or pretend to review aesthetics.
"""
import argparse,json,hashlib
from pathlib import Path
from PIL import Image
from .data import load_manifest

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def ingest(original,targets,rights,review,out,min_style=4,min_structure=4):
    root=Path(original).parent;targets=Path(targets)
    legal=json.loads(Path(rights).read_text())
    if not legal.get('outputs_can_train_independent_model'):
        raise PermissionError('training rights for generated outputs not confirmed')
    if not legal.get('rights_reference') or not legal.get('teacher_model_id'):
        raise ValueError('rights reference and teacher model ID are mandatory')
    rows=load_manifest(original)
    reviews={r['id']:r for r in (json.loads(x) for x in Path(review).read_text().splitlines() if x.strip())}
    staged=[];excluded=[]
    for row in rows:
        # Named after original scene/frame; never lose correspondence to G-buffers.
        id=f"{row['scene']}_{int(row['frame']):04d}"
        image=targets/f'{id}.png'
        qa=reviews.get(id)
        if not image.is_file() or not qa or not qa.get('reviewed') or qa.get('style',0)<min_style or qa.get('structure',0)<min_structure:
            excluded.append(id);continue
        with Image.open(root/row['source']) as src, Image.open(image) as dst:
            if src.size!=dst.size:excluded.append(id);continue
        r=dict(row)
        # External absolute path is deliberately stored for reproducibility.
        # Dataset loader resolves this with pathlib (absolute paths override root).
        r['target']=str(image.resolve());r['target_origin']='reviewed_teacher_output'
        r['target_sha256']=sha256(image)
        r['teacher_model_id']=legal['teacher_model_id'];r['rights_reference']=legal['rights_reference']
        r['quality_review']=qa
        staged.append(r)
    if not staged:raise RuntimeError('zero accepted targets: inspect QA or rights')
    # Scene-group holdout. Adjacent frames of a shot may NEVER leak into validation.
    scene_names=sorted({r['scene'] for r in staged})
    if len(scene_names)<3:raise RuntimeError('Need >=3 independent scene groups for train/val/test')
    ordered=sorted(scene_names,key=lambda v:hashlib.sha256(v.encode('utf-8')).digest())
    nval=max(1,len(ordered)//10);ntest=max(1,len(ordered)//10)
    groups={name:('val' if i<nval else 'test' if i<nval+ntest else 'train') for i,name in enumerate(ordered)}
    for row in staged:row['split']=groups[row['scene']]
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(''.join(json.dumps(x)+'\n' for x in staged))
    print(json.dumps({'accepted':len(staged),'rejected':len(excluded),'output':str(out)},indent=2))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--original',required=True);p.add_argument('--targets',required=True)
    p.add_argument('--rights',required=True);p.add_argument('--review',required=True);p.add_argument('--out',required=True)
    args=p.parse_args()
    ingest(args.original,args.targets,args.rights,args.review,args.out)
if __name__=='__main__':main()
