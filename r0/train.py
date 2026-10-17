"""Fully scripted paired supervision; optional PatchGAN + temporal loss.

IMPORTANT: samples/smoke contains mathematical NPR targets, *not* Ghibli art.
"""
import argparse,json,random,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .data import PairedDataset,TemporalPairs
from .model import build_generator,parameter_count,PatchDiscriminator

def gradient_loss(pred,target):
    return (pred[:,:,:,1:]-pred[:,:,:,:-1] - (target[:,:,:,1:]-target[:,:,:,:-1])).abs().mean()+\
           (pred[:,:,1:,:]-pred[:,:,:-1,:] - (target[:,:,1:,:]-target[:,:,:-1,:])).abs().mean()

def warp_previous_to_current(previous,flow,valid):
    """flow(y,x) = displacement CURRENT -> PREVIOUS in pixel units."""
    b,c,h,w=previous.shape
    yy,xx=torch.meshgrid(torch.arange(h,device=previous.device),torch.arange(w,device=previous.device),indexing='ij')
    xy=torch.stack([xx,yy],-1)[None].expand(b,-1,-1,-1).to(flow.dtype)+flow.permute(0,2,3,1)
    grid=torch.stack([(xy[...,0]+.5)/w*2-1,(xy[...,1]+.5)/h*2-1],-1)
    out=F.grid_sample(previous,grid,mode='bilinear',align_corners=False,padding_mode='border')
    return out*valid

def train(args):
    torch.set_num_threads(args.threads)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=torch.device(args.device)
    ds=(TemporalPairs(args.manifest,route=args.route,patch=args.patch,allow_smoke=args.allow_smoke)
        if args.temporal_weight else PairedDataset(args.manifest,route=args.route,patch=args.patch,allow_smoke=args.allow_smoke))
    loader=DataLoader(ds,batch_size=args.batch,shuffle=True,num_workers=0,drop_last=True)
    model=build_generator(args.route,args.base).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr,betas=(.5,.999))
    disc=PatchDiscriminator().to(device) if args.adv_weight>0 else None
    opt_d=torch.optim.Adam(disc.parameters(),lr=args.lr*.5,betas=(.5,.999)) if disc else None
    step=0;losses=[];since=time.perf_counter()
    while step<args.steps:
        for item in loader:
            if args.temporal_weight:
                x,y,px,py,flow,valid=item
                px=px.to(device);py=py.to(device);flow=flow.to(device);valid=valid.to(device)
            else:x,y=item
            x=x.to(device);y=y.to(device)
            if disc:
                disc.train();opt_d.zero_grad(set_to_none=True)
                with torch.no_grad():fake=model(x)
                loss_d=F.relu(1-disc(x,y)).mean()+F.relu(1+disc(x,fake)).mean()
                loss_d.backward();opt_d.step()
            opt.zero_grad(set_to_none=True)
            pred=model(x)
            l1=F.l1_loss(pred,y);grad=gradient_loss(pred,y)
            loss=l1+args.edge_weight*grad
            if args.temporal_weight:
                prev_pred=model(px)
                # Compare the *change* to target change; preserve intentional motion.
                warped_pred=warp_previous_to_current(prev_pred,flow,valid)
                warped_ref=warp_previous_to_current(py,flow,valid)
                temporal=((pred-warped_pred)-(y-warped_ref)).abs()*valid
                loss=loss+args.temporal_weight*temporal.sum()/valid.sum().clamp_min(1)/3
            if disc:loss=loss-args.adv_weight*disc(x,pred).mean()
            if not torch.isfinite(loss):raise RuntimeError('nonfinite train loss')
            loss.backward();opt.step()
            losses.append({'step':step+1,'l1':float(l1.item()),'edge':float(grad.item()),'total':float(loss.item())})
            step+=1
            if step%max(1,args.steps//5)==0:print(f'{args.route} step {step}/{args.steps}: loss {loss.item():.5f}',flush=True)
            if step>=args.steps:break
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':model.cpu().state_dict(),'route':args.route,'base':model.stem.net[0].out_channels,
                'train_origin':['paired_supervised'], 'steps':step,'input_channels':15},root/f'route_{args.route}.pt')
    with (root/f'route_{args.route}_train.json').open('w') as f:
        json.dump({'route':args.route,'params':parameter_count(model),'steps':step,'device':str(device),
                   'elapsed_sec':round(time.perf_counter()-since,2),'allow_smoke':args.allow_smoke,'losses':losses},f,indent=2)
    print('saved model and manifest',root)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',default='samples/smoke/manifest.jsonl');p.add_argument('--out',default='results')
    p.add_argument('--route',choices=['A','B'],default='A');p.add_argument('--base',type=int)
    p.add_argument('--steps',type=int,default=50);p.add_argument('--batch',type=int,default=1)
    p.add_argument('--patch',type=int,default=128);p.add_argument('--lr',type=float,default=0.0002)
    p.add_argument('--edge-weight',type=float,default=.25);p.add_argument('--adv-weight',type=float,default=0)
    p.add_argument('--temporal-weight',type=float,default=0)
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=3)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--allow-smoke',action='store_true')
    args=p.parse_args()
    if args.steps<1:raise ValueError('steps must be >0')
    train(args)
if __name__=='__main__':main()
