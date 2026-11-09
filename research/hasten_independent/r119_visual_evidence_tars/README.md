# R119 fixed visual evidence for TARS RL

Prospectively frozen single448x448 intervention, originalSFT parameters, unchanged TARS weights and query tape. Compare same-resolution TARS-RL/ordinaryRL/weightedCE plus the unchanged SFT parameters at448, separating resolution-only readout effects from incremental RL. No new support SFT or resolution sweep. [Protocol](PROTOCOL.md).

R118 made formerly-zero TRAIN success discoverable but did not reliably generalize to tiny VAL and damaged head. More visual detail is a falsifiable candidate, not a proven failure cause. Fixed queried TRAIN compact cache plus deferred VAL cache require3.412GB; all prior roots and local copies remain in the cumulative<10GB guard. Static development evidence, not a final novel method or continual/clinical confirmation.
