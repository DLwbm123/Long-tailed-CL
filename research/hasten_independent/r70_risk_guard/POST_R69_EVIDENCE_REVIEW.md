# Post-R69 evidence review: supervised-risk direction protection

Saved-only evidence, no new model evaluation. R67/R68/R69 sampling primary candidates remain below the common SFT starting point. R69 fixed supports do not improve sampled GRPO (mean+.017638/final-.101532/tail0pp vsdynamic). Fixed-exact has a local tail+0.324315pp vsSFT but mean-1.199584/final-.060230pp; retain positive secondary evidence without changing primary. Raw policy alignment previously failed; more nonzero groups did not yield better generalization.

All six R69 TRAIN macro/tail changes, percentagepoints fromsavedcurves, by stage0..5:
dynamic_sample: (+.139,0),(+.235,-.508),(+1.796,+3.116),(-2.285,-3.587),(-.298,+.622),(-6.910,-11.026).
fixed_sample: (+.005,+.475),(+1.141,+1.457),(+1.318,+2.411),(-1.934,-3.053),(-1.444,-1.564),(-5.869,-9.527).
dynamic_exact: (+1.558,+3.338),(+2.307,+4.671),(+.717,+1.317),(-1.466,-2.447),(+2.648,+5.391),(-1.638,-2.529).
fixed_exact: (+1.477,+3.338),(+1.839,+4),(+.783,+1.188),(-.308,-.456),(+3.012,+6.158),(-.453,-.385).
TRAIN crossfold is not independent holdout and BA is not CE; these numbers do not prove gradient conflicts. They motivate directly constraining the local supervised-risk direction rather than assume group sparsity/prototype drift is the sole cause. KL alone constrains distributions on average, not each proposed optimizer step's alignment to supervised classification risk. Pure continued CE also performed poorly, so simply adding more SFT is not the proposed mechanism.

R70 hypothesis: discard only the component of each actual AdamW-proposed LoRA parameter update that would increase current-batch natural supervisedCE to first order. This preserves a nonconflicting GRPO direction without adding a tuned auxiliaryCE coefficient, shrinkingLR, or choosing stoppingepoch. Compare originalunprotected/protected sampled and exact targets, alloriginaldynamic support/naturalpolicy. Every arm computes the same CE guide backward for exposure and cost; controls discard it. No current evidence proves conflicts occur; record rate/magnitude and both TRAIN andVAL outcomes, and rejectmechanism if no benefit. Finite-step or changing-support losses may stillincrease; this is not global safety or generalization guarantee. AdamWmoments remainunprojected and this limitation is disclosed. Not a revival of queued R66.
