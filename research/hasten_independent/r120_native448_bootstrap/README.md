# R120 native448 common bootstrap and incremental RL

One frozen native448 condition after R119's direct-resolution-switch failure. From generic task-adapter-zero initialization, reproduce the original two-epoch SFT schedule at448; derive the new fixed TRAIN TARS table, then compare matched TARS-RL/ordinaryRL/weightedCE from the same checkpoint and reference. Separate SFT contribution from RL increment. [Protocol](PROTOCOL.md).

Full TRAIN features live only in7.15GBRAM, reusing1556 existing compact rows and encoding5265 missing rows. Existing VAL features reused after all endpoints seal. No new feature disk cache, resolution grid or SFT hyperparameter search. Actual native admission required before science; this is a diagnostic, not mature innovation or clinical/continual validation.
