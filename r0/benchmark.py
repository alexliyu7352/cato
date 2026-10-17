"""Measure inference on current hardware, never extrapolate to 1080p."""
import argparse,platform,statistics,time,json
from pathlib import Path
import torch
from .model import build_generator,parameter_count

def benchmark(route='A',base=None,size=(128,128),reps=4,threads=3):
    torch.set_num_threads(threads)
    model=build_generator(route,base).eval()
    x=torch.rand(1,15,size[0],size[1]); x[:,7:14]=0;x[:,7]=1
    with torch.inference_mode():
        for _ in range(2):y=model(x)
        ts=[]
        for _ in range(reps):
            st=time.perf_counter(); y=model(x);ts.append((time.perf_counter()-st)*1000)
    return {'route':route,'parameters':parameter_count(model),'fp16_weights_MiB':round(parameter_count(model)*2/2**20,2),
            'shape_h_w':list(size),'cpu_median_ms':round(statistics.median(ts),2),'trials_ms':[round(v,2) for v in ts],
            'torch':torch.__version__,'platform':platform.platform(),'threads':threads,'cuda_available':torch.cuda.is_available()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--height',type=int,default=128);p.add_argument('--width',type=int,default=128)
    p.add_argument('--reps',type=int,default=3);p.add_argument('--threads',type=int,default=3)
    p.add_argument('--out',default='results/benchmark.json');a=p.parse_args()
    data=[benchmark(route,size=(a.height,a.width),reps=a.reps,threads=a.threads) for route in ['A','B']]
    dst=Path(a.out);dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(json.dumps(data,indent=2))
    print(json.dumps(data,indent=2))
if __name__=='__main__':main()
