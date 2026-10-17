"""Create an auditable visual smoke-test video from REAL model predictions."""
import argparse,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from .data import load_manifest,load_example
from .model import build_generator

def load_checkpoint(path):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    m=build_generator(ck['route'],ck['base']);m.load_state_dict(ck['state_dict']);return m.eval()

def img(x):
    return Image.fromarray(np.uint8(np.clip(x.permute(1,2,0).numpy()*255,0,255)))

def make(manifest,a,b,out,fps=3):
    torch.set_num_threads(3)
    root=Path(manifest).parent;rows=load_manifest(manifest)
    ma=load_checkpoint(a);mb=load_checkpoint(b)
    out=Path(out);out.mkdir(parents=True,exist_ok=True);frame_dir=out/'frames';frame_dir.mkdir(exist_ok=True)
    for i,row in enumerate(rows):
        xa,y=load_example(root,row,'A');xb,_=load_example(root,row,'B')
        with torch.inference_mode():ya=ma(xa[None]).cpu().squeeze();yb=mb(xb[None]).cpu().squeeze()
        imgs=[img(xa[:3]),img(y),img(ya),img(yb)]
        w,h=imgs[0].size
        sheet=Image.new('RGB',(4*w,h+58),(245,246,246));d=ImageDraw.Draw(sheet)
        labels=['3D SOURCE','FAKE NPR LABEL','STUDENT A','STUDENT B']
        for j,im in enumerate(imgs):
            sheet.paste(im,(w*j,25));d.text((j*w+7,5),labels[j],fill=(20,24,26))
        d.text((8,h+34),f"ENGINEERING SMOKE ONLY - NO GHIBLI TEACHER | scene={row['scene']}, frame={row['frame']}",fill=(160,35,35))
        sheet.save(frame_dir/f'{i:04d}.png')
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate',str(fps),'-i',str(frame_dir/'%04d.png'),'-c:v','libx264','-pix_fmt','yuv420p','-crf','20',str(out/'R0_engineering_demo.mp4')]
    subprocess.run(cmd,check=True)
    print(out/'R0_engineering_demo.mp4')

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',default='samples/smoke/manifest.jsonl')
    p.add_argument('--a',default='results/longer_smoke/route_A.pt');p.add_argument('--b',default='results/longer_smoke/route_B.pt')
    p.add_argument('--out',default='results/demo');p.add_argument('--fps',type=int,default=3)
    c=p.parse_args();make(c.manifest,c.a,c.b,c.out,c.fps)
if __name__=='__main__':main()
