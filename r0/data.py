"""Paired data manifest; teacher provenance and split gates are enforced."""
from pathlib import Path
import json,random
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from .model import CHANNELS
from .npr import pre_stylized_asset_render

REQUIRED=('source','target','depth','normal','semantic','line')

def load_manifest(path):
    path=Path(path)
    items=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not items:raise ValueError('empty dataset')
    for i,r in enumerate(items):
        missing=set(REQUIRED)-r.keys()
        if missing:raise ValueError(f'row {i}: missing {missing}')
        if 'target_origin' not in r:raise ValueError(f'row {i}: missing target provenance')
        for k in REQUIRED:
            if not (path.parent/r[k]).is_file():raise FileNotFoundError(path.parent/r[k])
    return items

def load_example(root,row,route='A'):
    root=Path(root)
    img=lambda name:np.asarray(Image.open(root/row[name]).convert('RGB'),dtype=np.float32)/255
    source=img('source')
    target=img('target')
    sem=np.asarray(Image.open(root/row['semantic']).convert('L'),dtype=np.int64)
    sem=np.clip(sem,0,6)
    line=np.asarray(Image.open(root/row['line']).convert('L'),dtype=np.float32)/255
    depth=np.load(root/row['depth']).astype(np.float32)
    n=np.load(root/row['normal']).astype(np.float32)
    if source.shape[:2]!=sem.shape or n.shape!=(*sem.shape,3):raise ValueError('invalid G-buffer geometry')
    valid=depth<.999;d=np.zeros_like(depth)
    if valid.any():
        lo,hi=np.percentile(depth[valid],[2,98]);d[valid]=1-np.clip((depth[valid]-lo)/max(hi-lo,1e-6),0,1)
    if route=='B':source=pre_stylized_asset_render(np.uint8(np.clip(source*255,0,255)),sem,depth).astype(np.float32)/255
    # No integer semantic labels passed to the network; they are categorical.
    onehot=np.eye(7,dtype=np.float32)[sem]
    cond=np.concatenate([source,d[...,None],n,onehot,line[...,None]],axis=-1)
    assert cond.shape[-1]==CHANNELS
    return (torch.from_numpy(cond.copy()).permute(2,0,1),
            torch.from_numpy(target.copy()).permute(2,0,1))

class PairedDataset(Dataset):
    def __init__(self,manifest,split='train',route='A',patch=0,allow_smoke=False):
        self.root=Path(manifest).parent;self.route=route;self.patch=patch
        data=load_manifest(manifest)
        self.rows=[r for r in data if r.get('split','train')==split]
        if not self.rows:raise ValueError(f'no split: {split}')
        smoke=[r for r in self.rows if r['target_origin']=='synthetic_npr_smoke_only']
        if smoke and not allow_smoke:raise PermissionError('smoke-only synthetic targets: specify --allow-smoke; not a Ghibli quality dataset')
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        inp,tgt=load_example(self.root,self.rows[i],self.route)
        if self.patch:
            h,w=inp.shape[-2:];p=self.patch
            if h<p or w<p:raise ValueError('patch larger than image')
            y=random.randint(0,h-p);x=random.randint(0,w-p)
            inp=inp[:,y:y+p,x:x+p];tgt=tgt[:,y:y+p,x:x+p]
        return inp,tgt

class TemporalPairs(Dataset):
    """Same-scene (t-1,t) with known current->previous motion from 3D."""
    def __init__(self,manifest,route='A',patch=0,allow_smoke=False):
        self.root=Path(manifest).parent;self.route=route;self.patch=patch
        rows=load_manifest(manifest)
        if not allow_smoke and any(r['target_origin']=='synthetic_npr_smoke_only' for r in rows):
            raise PermissionError('synthetic target cannot be used for style quality training')
        lookup={(r['scene'],r['frame']):r for r in rows}
        self.pairs=[(r,lookup[(r['scene'],r['frame']-1)]) for r in rows
                    if r.get('split')=='train' and (r['scene'],r['frame']-1) in lookup]
        if not self.pairs:raise ValueError('no temporal pairs')
    def __len__(self):return len(self.pairs)
    def __getitem__(self,i):
        row,prev=self.pairs[i]
        x,y=load_example(self.root,row,self.route);px,py=load_example(self.root,prev,self.route)
        flow=torch.from_numpy(np.load(self.root/row['motion_to_prev']).copy()).permute(2,0,1)
        valid=torch.from_numpy(np.asarray(Image.open(self.root/row['motion_valid']).convert('L'),dtype=np.float32).copy()[None]/255)
        if self.patch:
            h,w=x.shape[-2:];p=self.patch
            yy=random.randint(0,h-p);xx=random.randint(0,w-p)
            x=x[:,yy:yy+p,xx:xx+p];y=y[:,yy:yy+p,xx:xx+p]
            px=px[:,yy:yy+p,xx:xx+p];py=py[:,yy:yy+p,xx:xx+p]
            flow=flow[:,yy:yy+p,xx:xx+p];valid=valid[:,yy:yy+p,xx:xx+p]
        return x,y,px,py,flow,valid
