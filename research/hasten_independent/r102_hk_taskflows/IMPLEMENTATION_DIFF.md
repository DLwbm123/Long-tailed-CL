# Necessary differences from R101 ab67c59

- New class arrival5+5+5+4+4 instead of13+2+2+2+2+2; independent Ordered/stratified0, seed74002. Remaining order1/2 are definitions only.
- Original-label/stable-ID binding from real TRAIN and immutable component roles; tiny masks map original5/6/19, never hard-code new1/2/12. Tail threshold250 unchanged.
- Five-stage dynamic scheduling and age denominator4 instead of5. All fixed constants describing six-stage endpoint/update counts changed to20/17920 per flow, aggregate40/35840. Hyperparameters/architecture/loss/optimizer/carry unchanged.
- Early Ordered empty tail diagnostics/metrics use null; reward's existing empty protected branch remains finite. Mean tail excludes null with denominator3 vs5. New/old/head/tail empties null.
- Old supervised teachers/LoRA/features explicitly disallowed. Fresh R67 recipe implemented with generic frozen first10 prefixes and native last2+Q/V LoRA, parity checked; no extra algorithm, all teacher/cache cost charged. Detached opposite-component prototypes, natural count-priorCE+generic drift and10epochs unchanged.
- One two-worker controller enforces shared original24h/RAM/storage and40endpoint global F/VAL seal. Two threads/CPU cores each give totalCPU4, no scientific change. All new files remain data disk; no historical mutation.
- No inherited six-stage efficacy gate: full fixed trial, all paired deltas, no posthoc success definition. T/A0 retain original strong reference semantics under freshly reconstructed current-stream teacher. No standard-GRPO upgrade or subset-selection mechanism is introduced.
- The provided generator's optional whole-file SHA computation was replaced by exact label/count/stable-ID checks under the user's low-cost verification rule; task-generation algorithm and order seeds unchanged. Names are from prior checked dataset configuration, not inferred from counts.
