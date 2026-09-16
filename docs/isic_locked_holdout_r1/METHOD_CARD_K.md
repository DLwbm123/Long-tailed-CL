# K: frozen candidate, R1 evaluation only

K is the existing V9 model trained at `47fe3b6c180bd4e45ce83a779a71f844398cbca7`; the delivery reference is `a59156d2d2b0b2b4096e6f91e224e660374bdf10`. R1 adds zero neural epochs and zero optimizer steps.

Each of the three paired orders inherits its own full V2-C S0 network, freezes all feature parameters and buffers, and updates only the original main/few classifier in the already completed incremental training. Real CE uses all seen classes in the original three-term objective, preserving reductions, division by three, pool assignment, pull constraint and theta. Old synthetic CE uses the raw main+few sum, four features per old class, cap 48 and coefficients 2 at S1 and 3 at S2.

K changes H's synthetic standard deviation to `sqrt(max(stored_var, 0) + 1e-6)`, without the upper variance cap. It uses diagonal variance, without full covariance, a new temperature or a new module. It does not include I's old-row freezing, J's three-head replay, or head normalization. R1 performs no synthesis or statistics update.

The main predictor is the uncalibrated main+few logit sum. A compact head delta requires its exact full C-S0 parent. The delta size alone is not the method's storage cost. Numeric labels are retained without unverified disease-name mapping.
