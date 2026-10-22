# r01 execution

Read PROTOCOL.md first. This is an independent HASTEN route, not a module of the earlier method. It executes the pinned official model/training/hierarchy code with the explicit corrections in `compat.py`; the upstream checkout is not modified.

Supply a private JSON config through `RUN_CONFIG` with `output`, `official` (pinned upstream source directory), `support_plan` (existing support-only group split), `image_root`, `clip_cache`, `python` and `entry`. The plan contains `rows` (relative_path, label, identity_component) and `folds[0]` train/test row indices. No query dataset is accepted. Provide `RUN_SOURCE` pointing to these scripts. Place `{"hk":["an endoscopic image of {}."]}` in the execution working directory's `knowledge/virtual_templates.json`.

Use `control.py` from a neutral entry path; it starts `entry.py`, preserves the first execution deadline across retries, and independently limits time/RSS/allocated storage. Run on one CUDA GPU. Runtime is the existing PyTorch 2.1.2 / OpenCLIP 2.26.1 environment, differing from the official requirements 2.6.0 / 2.32.0; this is recorded rather than silently called an exact reproduction. No dependency upgrades are required for this bounded development run.

`runner.py` first checks synthetic statistics, sampling, actual parameter projection and the official parent-cache lifecycle. The first two real stages count towards the same 36-stage experiment. Stage completion requires exact saved/restored prediction agreement; actual checkpoints and per-example responses remain private. Public JSON contains aggregate metrics, mechanism checks and runtime receipts only.

On failure, retain the original deadline and all failure receipts. Do not blindly rerun completed stages or reset costs. A recovery may use the saved state; the initial runner deliberately stops on errors rather than automatically changing parameters.

Obtain upstream from https://github.com/LAMDA-CL/NeurIPS2026-HASTEN at commit `063f295d08991d4da5b1f46a61bfcd279a94dbc6` through the required proxy. Upstream is referenced, not bundled. No new data or model license is granted here.
