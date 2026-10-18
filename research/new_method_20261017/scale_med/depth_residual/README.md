# Fixed-budget depth residual diagnostic

See PROTOCOL.md for the frozen scientific contract. Dependencies are the existing scale_med environment (PyTorch, timm, safetensors, NumPy, torchvision, Pillow). Add this folder, ../information_attribution and .. to PYTHONPATH. RUN_CONFIG names a private JSON with output, benchmark_config and source_commit. Existing benchmark_config supplies transfer_config, prior_rows, output (audited test cache/rows), and ISIC image roots; no private paths are released here.

Run run_depth.py in three separate interpreters with PHASE=prepare,fit,evaluate, sequentially. The provided coordinator uses neutral runtime symlinks installed by the host-specific launcher, never project-identifying arguments. Create output/inputs first on the approved mounted storage. Runtime configurations, images, features, heads, row bundles and logs stay private. After EXIT.json reports0, report_depth.py RUN_OUTPUT PUBLIC_REPORT_DIRECTORY exports aggregate receipts and a Chinese report. Report arguments can be passed through a neutral stdin launcher on hosts with process-argument privacy requirements.

The S control is a specified 500-step group-lasso feature selector followed by ridge, not a reproduction claim for Head2Toe. No code adapts projections after the initial stage or retries based on evaluation scores.
