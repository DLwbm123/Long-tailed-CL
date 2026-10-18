# Three frozen spaces, training-only quota feasibility

Read PROTOCOL.md. Dependencies: existing depth_residual/PyTorch extraction runtime, SciPy/NumPy, and isolated PySCIPOpt6.0.0 (SCIP10). No user-chosen big-M or weight bound. Solver infeasible without an independently checked exact certificate is unresolved.

Private RUN_CONFIG has output, reference_run, benchmark_config and source_commit. Add this folder, ../depth_residual, ../information_attribution and .. to PYTHONPATH; isolated solver package path is also needed. coordinate.py uses neutral /tmp/q580 symlinks to avoid identifying process arguments. MODE=prepare reconstructs frozen training features and writes QUOTA_SEAL.json, then exits. Inspect and publicly deliver aggregate quotas before creating DISPATCH_SEAL.json and running MODE=solve. Solver child processes consume fixed budgets sequentially and generate public_results automatically on completion. No test scoring follows.

All features, row identities, witnesses, raw constraints/logs and runtime configs remain private. report.generate exports only aggregate receipts and a Chinese report. This is an offline diagnostic with pooled historical TRAINING inputs, not a legal no-old-input learning algorithm and not a19MB deployment comparison. No parameter/source changes, quota relaxation, retries or budget transfer after dispatch.
