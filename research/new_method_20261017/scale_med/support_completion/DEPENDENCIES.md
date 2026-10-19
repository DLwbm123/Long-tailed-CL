# Dependencies

The entry reuses `depth_residual/core_depth.py` for fixed coordinates, mapping and state accounting, and `information_attribution/core_attribution.py` for weights and normalization. Geometry and head solves in this entry run on CPU. The private launcher exposes these helpers on the module path and supplies the approved frozen feature cache, neutral interpreter/entry and output directory. No new package is required; data, group identities and private head/transform assets are intentionally excluded from public delivery.
