# R125: seen-category interface qualification — completed

The fixed five-stage, five-arm replay-free generative experiment finished normally. All four RL arms passed the first new-task startup diagnostics. The proposed TARS_RAPO passed 4/9 original efficacy gates: real retention gains over TARS/GRPO, but no net post-stage macro/tail gain over RaPO, insufficient head protection, and sparse tiny-class transfer. All original metrics and negative results are retained.

| Method | Mean post-stage macro % | Mean post-stage tail % | Mean classwise forgetting pp |
|---|---:|---:|---:|
| GRPO | 16.1930 | 1.7262 | 35.5672 |
| TARS | 17.0042 | 1.7262 | 35.1009 |
| RaPO | 22.9985 | 4.7727 | 16.7780 |
| TARS_RAPO | 22.9707 | 3.5823 | 15.1628 |
| TARS_CE | 14.7715 | 4.2888 | 40.7386 |

Read [REPORT_ZH.md](REPORT_ZH.md), [MECHANISM_ANALYSIS_ZH.md](MECHANISM_ANALYSIS_ZH.md), [FINAL_VERIFICATION.json](FINAL_VERIFICATION.json), [INTERFACE_QUALIFICATION.json](INTERFACE_QUALIFICATION.json), and the per-class/count/cost files. `finalize_saved.run(cfg)` checks the saved private experiment in place; `saved_analysis.run(cfg)` needs only the safe aggregate outputs. Neither regenerates responses. Postprocessing fixed a raw-CSV versus frozen-Engine hyphen-to-space name comparison; the failed source/record and costs are preserved. Scientific worker source and frozen protocol were unchanged.

All methods shared cumulative arrived category names, a fresh generic first-task SFT, current-only TRAIN access and isolated seen-class VAL after all stage models were sealed. The vocabulary interface also changed the common supervised start; its contribution is not RL gain. CE used fewer forwards and no rollout KL. One development order and tiny denominators are not independent clinical evidence or an innovation claim. Source and safe aggregates only; no private images, predictions, indices, features, weights or credentials.
