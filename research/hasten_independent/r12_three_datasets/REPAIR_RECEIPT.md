# R12 template registration repair

Attempt 1 exited before the first ISIC optimizer update with `KeyError: 'ISIC'` in the upstream hierarchy manager. The runtime virtual-template registry had retained only the NIH key. ISIC stage-0 feature encoding had completed; those cached features were preserved.

`campaign.py` now registers each dataset's already frozen `parent_template` under its exact dataset key before starting its worker. This does not change the registered scientific template, hierarchy, labels, learning parameters or evaluation protocol. Both ISIC and HK receive the same configuration-driven fix.

Attempt 2 resumed using the original `FIRST_ACTUAL` start/deadline and the 8 GB campaign cap. The first attempt's exit receipt and traceback remain preserved. ISIC was verified performing optimizer updates and subsequently encoding stage 1. Cached stage 0 was reused. Main/child process command lines and GPU process display use neutral launcher names.
