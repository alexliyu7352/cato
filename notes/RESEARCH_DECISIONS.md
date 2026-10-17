# R0 Architecture decisions (2026-10-17)

## Research anchors, original primary sources

1. **E²GAN (ICML 2024)**: https://proceedings.mlr.press/v235/gong24g.html
   - Small ResNet/Transformer generator and offline paired-data concept.
   - NOT copied line-for-line; this project's generator is an independent simplified implementation.
   - Its 256² benchmarks DO NOT establish 1080p animation speed or fidelity.
2. **GBGST (IEEE Transactions on Games)**: https://github.com/ioannouE/GBGST
   - G-buffer inputs and temporal coherence are more relevant to a renderer than generic photographs.
3. **StructuReiser (EGSR 2025)**: https://github.com/radimspetlik/structureiser
   - Strong structural losses and keyframe consistency; training cost and per-sequence focus argue against deployment as-is.
4. **SPADE (CVPR 2019)**: https://github.com/NVlabs/SPADE
   - Feature modulation from discrete semantic classes; this project implements a small affine conditioner.
5. **GAN Compression (CVPR 2020)**: https://github.com/mit-han-lab/gan-compression
   - Optional phase, only AFTER a good uncompressed model is demonstrated.
6. **StyleMVD**: https://github.com/PeterZs/StyleMVD
   - Offline multi-view asset stylization inspiration; not reproduced in this smoke scaffold.
7. **Blender EXR passes**: https://docs.blender.org/manual/en/4.5/render/layers/passes.html
   - Depth, normal, vector, cryptomatte. Multilayer EXR producer in `scripts/blender_capture.py` is not yet tested.

## R0 routes

- **Route A**: 11.78M param RGB+depth+normal+semantic+line -> image neural generator. Constant style; no text encoder, no diffusion weight.
- **Route B**: 6.20M param same generator, but with deterministic prerendered flat material palette on RGB first; reuses geometry, stylizes only residual image. This is a **pipeline hypothesis**, not the full optimized UV/asset baking approach.

Both routes use a small pooled attention stage to bound quadratic complexity. Spatial semantics modulate residual blocks. The 3D smoke asset is deliberately synthetic and does not demonstrate aesthetic adequacy. The trained weights supplied are merely a three-gradient-step integration check, and should NEVER be taken as a visual-quality release.

## Teacher model shortlist, NOT used for training yet

| Candidate | URL | Main risk | Status |
|---|---|---|---|
| nitrosocke/Ghibli-Diffusion | https://huggingface.co/nitrosocke/Ghibli-Diffusion | trained on Ghibli film images; intellectual property provenance + image-edit control | inspected; not downloaded |
| KappaNeuro/studio-ghibli-style | https://huggingface.co/KappaNeuro/studio-ghibli-style | SDXL adapter; model card `license: other` | inspected; do not use as training source until rights confirmed |
| Owen777/Kontext-Style-Loras | https://huggingface.co/Owen777/Kontext-Style-Loras | requires FLUX.1-Kontext-dev, noncommercial restrictions | inspected; not downloaded |

These models are **not assumed to allow student training**. See `scripts/teacher_rights_template.json`. Rights chain includes both adapter and foundation model, plus image content and the right to train downstream models from outputs.

## Validation gates before any quality claim

- Same input scene/camera/pose for all teachers; reject geometry-changing labels.
- At least two licensed teacher candidates, with matching fixed input and input structure evaluation.
- Split by scenes and assets, NEVER by adjacent frames. The smoke dataset intentionally uses adjacent frames just to exercise data loading; its validation score does not establish generalization.
- Evaluate full-res line quality, backgrounds, occlusion, temporal consistency, and unseen assets.
- Instrument inference resolution, device, actual GPU, latency, peak RAM/VRAM and frame throughput.
- Two dedicated original art-style references, no training on proprietary film stills unless licensed.
