# Corrected server deployment, 2026-09-15

The user corrected the rental server destination. The incomplete migration on the incorrect host was stopped and its task-specific directory removed after checking paths and processes. Original local data and my-gpu artifacts remain intact.

The new target has one RTX 4090 D (24 GB), Ubuntu 22.04, and a 49 GiB ext4 data volume. The intended experiment remains ISIC2019-LT split1, fixed cleaned manifests, seeds 1993/1994/1995, B/C, 10 epochs per session and one sealed batch test. Historical A100 P1 and budget reports in this directory remain historical evidence; their budget does not describe the new GPU.

## Storage correction

The old NFS `du` block count was approximately twice the checkpoint file length. The initial 80 GB expansion suggestion was therefore excessive. Actual P1 checkpoint lengths are 756,282,654 to 765,923,766 bytes. Budgeting 0.80 GB each for 18 session checkpoints, 6 retained recovery files, 8 P1 files and one atomic temporary write gives 26.4 GB. Original candidate images use 9,541,488,783 bytes; locked weights use 346,284,714 bytes. With code and the small virtual environment, the expected total is about 37 GB, below the approximately 52 GB usable volume. No checkpoint fields, precision, training semantics or retention were reduced.

## Software acceptance

The server-provided torch 2.12.1+cu130 and torchvision 0.27.1+cu130 are used in a separate environment with timm 1.0.15, numpy 1.26.4, Pillow 10.4.0 and easydict 1.13. Exact versions are in new_server_engineering.json. The CPU suite passed in both old and new environments, including graph cleaning, expanded embedding, optimizer/scheduler equivalence, metrics, checkpoint rejection and head-norm denominator edge cases. The approved model SHA256 was verified on the target.

At this deployment checkpoint, image transfer and new-hardware GPU P1/budget acceptance are pending. No formal trajectory or model test prediction has been executed on this target. Runtime protocol will be locked only after those gates pass. Prior BLOCKED reports are preserved.
