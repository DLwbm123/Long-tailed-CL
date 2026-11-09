# R119 fixed448 visual evidence diagnostic

Completed and saved-verified. Original224-trained SFT at448 falls to **macro43.1622/head77.4869/tail5.7170%**. TARS-RL recovers to **47.2370/76.6180/15.1851%**, but same448 CE reachesmacro48.8520/tail19.1578%. **Six of ten original gates pass; RL-over-CE and end-to-end macro/head gates fail.** TinyRL remains0; CEtiny5%isoneUC2-3rowoutoffour.

[Protocol](PROTOCOL.md) · [Report](REPORT_ZH.md) · [Verification](FINAL_VERIFICATION.json) · [Class counts](CLASS_RESULTS.csv)

This tests direct resolution switching of224-trained parameters, not a native448 SFT. It does not establish that high resolution is useless or that input detail caused previous failures. All negative controls/costs preserved; no resolution sweep. Private images/features/indices/weights/per-example predictions excluded.
