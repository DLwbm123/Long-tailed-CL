# Existing runtime dependencies

No new packages or model downloads. Reuse the historical torch/timm/torchvision/safetensors/Pillow/NumPy runtime.

- `core_depth.mapped`: `../depth_residual/core_depth.py`, original transform definitions from source19bf801.
- `core_attribution.weights`: `../information_attribution/core_attribution.py`, equal class/group/within-group weighting.
- `run_images.Images`: alias of `../run.py` for the locked RGB/224 bicubic/mean-std .5 image pipeline.
- Runtime configuration supplies private reference-run, benchmark configuration, output and neutral entry locations. No paths or data are hard-coded in public configuration.

The global feature cache is the original locked APART pointwise double-branch output, not a substitute bare-ViT embedding. The unchanged original checkpoint produces block6 CLS only. This is an explicitly privileged full-history audit, not a retirement-compliant learner or an exact reproduction of QUILT.
