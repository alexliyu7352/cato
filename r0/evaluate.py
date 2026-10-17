"""Real model inference and technical metrics (not aesthetic scores)."""
import argparse,json,time,statistics,platform
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from .data import PairedDataset,load_example
from .model import build_generator,parameter_count
from .npr import stylize

def eval_one(model,x,y):
    with torch.inference_mode():out=model(x.unsqueeze(0)).squeeze(0).cpu().clamp(0,1)
    return out,float((out-y).abs().mean())

def image(t):return Image.fromarray(np.uint8(np.clip(t.permute(1,2,0).numpy()*255,0,255)))

def evaluate(manifest,models,out,split='val',allow_smoke=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    root=Path(manifest).parent
    ds=PairedDataset(manifest,split=split,allow_smoke=allow_smoke)
    loaded={};params={}
    for route,path in models.items():
        ckpt=torch.load(path,map_location='cpu',weights_only=True)
        model=build_generator(ckpt['route'],ckpt['base']);model.load_state_dict(ckpt['state_dict'])
        model.eval();loaded[route]=model;params[route]=parameter_count(model)
    metrics=[];tiles=[]
    for i,row in enumerate(ds.rows):
        x,y=load_example(root,row,'A');n,w,h=0,x.shape[-1],x.shape[-2]
        im=[image(x[:3]),image(y)];m={'scene':row['scene'],'frame':row['frame'],'origin':row['target_origin']}
        for route in ['A','B']:
            if route in loaded:
                x_r,_=load_example(root,row,route)
                output,mae=eval_one(loaded[route],x_r,y)
                m[route+'_mae']=round(mae,5);im.append(image(output))
            else:im.append(Image.new('RGB',(w,h),'white'))
        tiles.append(im);metrics.append(m)
    w,h=tiles[0][0].size;sheet=Image.new('RGB',(w*4,(h+25)*len(tiles)),(255,255,255))
    d=ImageDraw.Draw(sheet)
    for i,(ims,row) in enumerate(zip(tiles,metrics)):
        y=i*(h+25)
        for j,im in enumerate(ims):sheet.paste(im,(j*w,y+25))
        d.text((4,y+4),f"{row['scene']} t={row['frame']}   original  |  synthetic-only target  |  model A  |  model B",fill=(10,10,10))
    sheet.save(out/'comparison.png')
    report={'verified_origin':'technical_smoke_no_ghibli_teacher' if allow_smoke else 'user_provided',
            'models_parameters':params,'frames':metrics,
            'mean_absolute_error':{r:round(statistics.mean(m[r+'_mae'] for m in metrics),5) for r in loaded}}
    (out/'evaluation.json').write_text(json.dumps(report,indent=2))
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',default='samples/smoke/manifest.jsonl')
    p.add_argument('--model-a');p.add_argument('--model-b');p.add_argument('--out',default='results/eval')
    p.add_argument('--split',default='val');p.add_argument('--allow-smoke',action='store_true')
    a=p.parse_args();models=dict((k,v) for k,v in [('A',a.model_a),('B',a.model_b)] if v)
    if not models:raise ValueError('require --model-a and/or --model-b')
    torch.set_num_threads(3)
    print(json.dumps(evaluate(a.manifest,models,a.out,a.split,a.allow_smoke)['mean_absolute_error'],indent=2))
if __name__=='__main__':main()
