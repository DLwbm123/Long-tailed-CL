# Dependencies

The neutral launcher exposes the existing image loader from the existing SCALE-Med image-loading entry (`scale_med/run.py`) as `run_images` and adds the existing `depth_residual/core_depth.py` and `information_attribution/core_attribution.py` helpers to the module path. The entry uses the existing timm/PyTorch runtime. Its private config contains the approved cache, base training-row bundle, benchmark/encoder configuration, neutral interpreter/entry and output paths. No dependency was added. Private inputs and projections are excluded from public delivery.
