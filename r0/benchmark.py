"""Measure inference on the actual CPU/GPU and resolution (no extrapolation)."""
import argparse
import json
import platform
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from .model import build_generator, parameter_count


def benchmark(route='A', base=None, size=(128, 128), reps=4, threads=3,
              device='cpu', precision='fp32'):
    if precision not in ('fp32', 'fp16'):
        raise ValueError('precision must be fp32/fp16')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable; choose a GPU runtime in Colab')
    if device == 'cpu' and precision != 'fp32':
        raise ValueError('CPU baseline uses fp32 only')
    torch.set_num_threads(threads)
    dev = torch.device(device)
    model = build_generator(route, base).to(dev).eval()
    x = torch.rand(1, 15, size[0], size[1], device=dev)
    x[:, 7:14] = 0
    x[:, 7] = 1
    is_cuda = dev.type == 'cuda'
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(dev)

    def run_one():
        with torch.inference_mode():
            with (torch.autocast(device_type='cuda', dtype=torch.float16)
                  if is_cuda and precision == 'fp16' else nullcontext()):
                return model(x)

    for _ in range(2):
        _ = run_one()
    if is_cuda:
        torch.cuda.synchronize(dev)
    ts = []
    for _ in range(reps):
        start = time.perf_counter()
        _ = run_one()
        if is_cuda:
            torch.cuda.synchronize(dev)
        ts.append((time.perf_counter() - start) * 1000)
    median = round(statistics.median(ts), 2)
    return {
        'route': route, 'parameters': parameter_count(model),
        'fp16_weights_MiB': round(parameter_count(model) * 2 / 2**20, 2),
        'shape_h_w': list(size), 'device': str(dev), 'precision': precision,
        'latency_median_ms': median,
        'cpu_median_ms': median if not is_cuda else None,
        'trials_ms': [round(v, 2) for v in ts],
        'peak_cuda_memory_mib': round(torch.cuda.max_memory_allocated(dev) / 2**20, 1) if is_cuda else None,
        'gpu_name': torch.cuda.get_device_name(dev) if is_cuda else None,
        'torch': torch.__version__, 'platform': platform.platform(),
        'threads': threads, 'cuda_available': torch.cuda.is_available(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--height', type=int, default=128)
    p.add_argument('--width', type=int, default=128)
    p.add_argument('--reps', type=int, default=3)
    p.add_argument('--threads', type=int, default=3)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32')
    p.add_argument('--out', default='results/benchmark.json')
    a = p.parse_args()
    data = [benchmark(route, size=(a.height, a.width), reps=a.reps,
                      threads=a.threads, device=a.device, precision=a.precision)
            for route in ('A', 'B')]
    dst = Path(a.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(data, indent=2))
    print(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()
