# R117 tailness-proportional query allocation

Prospective RL mechanism diagnostic: transfer TARS effort into query allocation and importance-correct both the surrogate and reference KL. Four matched-query arms test full correction, omitted KL correction, ordinary RL and weighted CE. Natural-query256-stepTARS is a frozen reused anchor. No effect result yet; static development only. [Protocol](PROTOCOL.md).

[Saved-prediction transition audit](SAVED_ANALYSIS_ZH.md) separates rescued/harmed validation rows and query exposure from independent TRAIN components. CPU-only `saved_diagnostics.run(cfg)` reads saved artifacts; run again after final completion to include all four R117 endpoints and recoverable reward totals. Exact per-class all-wrong/all-right/mixed group counts were not retained by the frozen worker.
