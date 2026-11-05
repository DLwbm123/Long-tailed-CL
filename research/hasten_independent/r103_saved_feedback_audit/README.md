# R103 saved-only feedback audit

Scientific source ec5502cf886e7e453e98609f9fcd6dde0df6d5fd. Read REPORT_ZH.md, ANALYSIS_PROTOCOL.md and LIMITATIONS.md first. Public files contain only code/aggregates, no private identity or prediction arrays.

On the authorized CPU host with the existing PyTorch environment, set CUDA_VISIBLE_DEVICES empty and OMP/MKL/OPENBLAS_NUM_THREADS=2, restrict compute affinity to2CPUs. Use the controlled SOURCE root and a NEW analysis OUTPUT directory. Run:

    python audit.py --source "$SOURCE" --output "$OUTPUT"
    python verify.py --source "$SOURCE" --output "$OUTPUT"
    python metadata.py --source "$SOURCE" --output "$OUTPUT"
    python report.py --output "$OUTPUT"

These are statistics/verification commands, not training launchers. Source must be the trusted project's saved Tensor dictionaries/manifests; no model/checkpoint/image/TEST reads or model imports are needed. The original execution additionally performed a second verification after adding weighted-matrix and denominator coverage; both costs are retained. COSTS.json records that actual execution, not a runtime prediction for other hosts.

Full information coverage is partial because old-policy/action and row router probabilities were not saved. Do not change this to an unconditional passed status or recover missing outputs through new forwards. No new performance experiment or scheduled monitor. Complete and stop.
