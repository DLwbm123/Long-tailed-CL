# R1 resources and memory

Hardware: NVIDIA GeForce RTX 4090 D, 24,564 MiB. One evaluation worker. torch 2.12.1+cu130; CUDA 13.0; FP32, AMP off, matmul TF32 off, cuDNN TF32 on. Exact versions are in ENVIRONMENT_R1.json.

E1 measured process section: 221.829s; E2: 230.734s; sum: 7.543 minutes, below 120 minutes. These conservative process intervals include checkpoint IO and CPU work while the evaluation process is running; they are not GPU kernel active time or a utilization integral. No training epoch or optimizer step was executed.

Peak PyTorch allocation: 1290.704 MiB; peak reserved: 1492.000 MiB. Main-process peak RSS: 3424.547 MiB; this does not sum all DataLoader child RSS. Bootstrap CPU time: 0.256s, below the 30-minute estimate limit.

## Inference and continued-increment state

All byte counts below describe tensor payloads unless explicitly named as on-disk files. They exclude Python object/allocator overhead. Inference states include every backbone, adapter, head and buffer; actual process peaks are reported above. Statistical memory is not claimed anonymous or privacy-safe.

| Method | Inference state MiB | Final continued-increment statistics MiB |
|---|---:|---:|
| C | 700.873 | 0.094 |
| H | 700.873 | 0.094 |
| K | 700.873 | 0.094 |
| RA | 327.343 | 36.047 |
| RD | 330.325 | 36.047 |

For RA/RD, continued-increment sufficient statistics are n_c, s_c and Q_c in float64; the reconstructed stage classifiers are W in float64 without bias. R1 retains small W files and references the original feature caches; Q is transient working memory, not a newly saved 37 MiB copy per order.

## Full archival dependency

| Asset | On-disk MiB | Interpretation |
|---|---:|---|
| C, all nine session checkpoints | 6519.341 | Original full network, optimizer, statistical memory, RNG and metadata |
| Shared C-S0 parents, three total | 2191.268 | Required immutable parents for both H and K; count shared files once |
| H, six head-delta checkpoints | 1.582 | Additionally requires the full parent set above; not the total method footprint |
| K, six head-delta checkpoints | 1.584 | Additionally requires the full parent set above; not the total method footprint |
| Locked AugReg original weights | 330.243 | Existing file reused; no download/copy of weights for R1 |
| Locked DINOv2 original weights | 330.301 | Existing file reused; no download/copy of weights for R1 |

Original two-encoder train/val feature caches total 111.696 MiB, plus 20.851 MiB private association metadata. These are pre-existing referenced assets. R1 adds two small 764×768 FP32 holdout feature caches (about 4.477 MiB payload) and private logits/analytic classifiers. No full neural parent checkpoint is duplicated.

## Fixed-layout validation latency

First 48 sample-ID-sorted full-eight-class val images, decoded tensor resident on GPU. Five warmups then 20 measured forwards; CUDA synchronization around timings. Fixed batch48, no batch search. C/H/K use seed1993 S2. Ridge rows measure encoder feature extraction including normalization; the float64 ridge score multiplication is separately excluded from these encoder timings.

| Method | Median forward ms | Mean forward ms |
|---|---:|---:|
| C | 218.901 | 218.898 |
| H | 220.158 | 221.011 |
| K | 221.115 | 221.150 |
| RA | 53.106 | 52.960 |
| RD | 73.987 | 74.286 |

Latency excludes image decoding, disk IO, checkpoint restore, host transfer and bootstrap; it is not end-to-end clinical throughput. The same decoded batch remains cached across the 25 calls.

## Storage and historical costs

New R1 files: 44.984 MiB (<1 GiB); final free bytes: 2,336,395,264 (>512 MiB). The only removed historical-directory files were four installed-dependency wheel downloads, totaling 16,635,502 bytes. Versioned installed dependencies were checked first. Original data, weights, checkpoint/resume states, results and negative-result evidence remain intact. No server rental or expansion occurred.

Historical costs are not included in R1 inference time. V2 reported 3.386 hours training wall time for its complete B/C six-trajectory, 180-epoch matrix; that aggregate cannot be attributed to C alone. V6-H reported 0.866 summed GPU-process hours for its three 20-epoch incremental trajectories; V9-K reported 0.874 hours for the same-sized incremental matrix. Both depend on the prior C-S0 training. V3 including its DINO supplement reported 4.105 GPU-process hours for the larger P0–P2 program, not solely the frozen baselines. Pretraining cost is outside these measurements. Sources: preserved V2 AUDIT_AND_CONCLUSIONS.md, V6 COMPLETION_REPORT.md, V9 COMPLETION_REPORT.md and V3 COMPLETION_REPORT.md.

GPU workers have exited after the fixed evaluation. Hourly monitoring remains paused. No further experiments are launched.
