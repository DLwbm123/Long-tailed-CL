# R117 tailness-proportional query allocation

Completed and independently checked from saved artifacts: **macro50.0162%, tail19.3029%, head78.1701%**. Versus commonSFT: **+2.9268/+9.3117/−2.9260pp** respectively. Eight of nine original gates passed; head protection failed. All tiny VAL recalls remain zero. Static single-development-seed evidence, not independent confirmation or continual learning.

[Protocol](PROTOCOL.md) · [Final report](REPORT_ZH.md) · [Class counts](CLASS_RESULTS.csv) · [Verification](FINAL_VERIFICATION.json)

The primary fully corrects both PG and KL under q=p*w/Z and exceeds same-query ordinaryGRPO and weightedCE on macro/tail at this endpoint. CE has fewer forwards and no KL; the naturalTARS256 anchor is reused, not a new replicate. Four previously undiscovered tiny classes still generated zero correct responses despite increased exposure.

[Saved transition audit](SAVED_ANALYSIS_ZH.md) and `SAVED_DIAGNOSTICS.json` distinguish rescue/harm, repeated queries and independent TRAIN components. CPU-only `saved_diagnostics.run(cfg)` now includes completed R117 endpoints and reward totals. Exact per-class all-wrong/all-right/mixed group compositions were not retained by the frozen worker. Private images, identities, predictions, features and weights are excluded.
