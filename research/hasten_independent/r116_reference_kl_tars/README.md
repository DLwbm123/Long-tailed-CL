# R116 fixed-reference KL TARS diagnostic

Prospective successor to R115's TRAIN-label-loss guard failing to preserve VAL head accuracy. It constrains full-vocabulary conditional distributions on a fixed class-balanced SFT head reference bank, comparing projected and scalar-shrunk TARS with a new same-update-count unconstrained control. Static development only; no effect result yet. [Protocol](PROTOCOL.md).
