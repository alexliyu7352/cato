"""Motion-warped residual metrics; technical only, not aesthetic evaluation."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .data import load_manifest,load_example
from .model import build_generator
from .train import warp_previous_to_current

def score(manifest,models):
    torch.set_num_threads(3);root=Path(manifest).parent;rows=load_manifest(manifest)
    lookup={(r['scene'],r['frame']):r for r in rows}
    results={}
    for route,ck in models.items():
        saved=torch.load(ck,map_location='cpu',weights_only=True)
        net=build_generator(saved['route'],saved['base']).eval()
        net.load_state_dict(saved['state_dict'])
        pred={};true={}
        with torch.no_grad():
            for r in rows:
                x,y=load_example(root,r,route)
                pred[(r['scene'],r['frame'])]=net(x[None]).cpu()
                true[(r['scene'],r['frame'])]=y[None]
        errors=[]
        for r in rows:
            k=(r['scene'],r['frame']);prev=(r['scene'],r['frame']-1)
            if prev not in pred:continue
            flow=torch.tensor(np.load(root/r['motion_to_prev']),dtype=torch.float32).permute(2,0,1)[None]
            valid=torch.tensor(np.array(Image.open(root/r['motion_valid']).convert('L'),dtype=np.float32))[None,None]/255
            reference_delta=true[k]-warp_previous_to_current(true[prev],flow,valid)
            model_delta=pred[k]-warp_previous_to_current(pred[prev],flow,valid)
            err=((model_delta-reference_delta).abs()*valid).sum()/valid.sum().clamp_min(1)/3
            errors.append(float(err))
        results[route]={'pairs':len(errors),'temporal_residual_mae':round(float(np.mean(errors)),6) if errors else None}
    return results

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',default='samples/smoke/manifest.jsonl')
    p.add_argument('--model-a');p.add_argument('--model-b');p.add_argument('--out',default='results/temporal.json')
    a=p.parse_args();models={k:v for k,v in [('A',a.model_a),('B',a.model_b)] if v}
    data=score(a.manifest,models);Path(a.out).write_text(json.dumps(data,indent=2));print(json.dumps(data,indent=2))
if __name__=='__main__':main()
