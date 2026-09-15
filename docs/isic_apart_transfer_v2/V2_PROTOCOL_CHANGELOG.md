# V2 protocol changelog

Reference source: 7a089e4b30a3efa6e2943aa4b99c1f5e60652b16; preserved V1 audit: 45800adb6a14258723d25314d230033491e87411. New isolated branch: exp/apart-isic-lt-transfer-v2. Original source-only workspace and five historical CSV line-ending differences are untouched.

User-approved changes: deterministic lesion/content cleaning; capacity-adapted raw-frequency embedding; one new fixed ImageNet weight and its .5 normalization; explicit legacy effective optimizer profile. APART losses and ConCM-lite definition remain unchanged. No Full Dynamic, new method module, sweep, split regeneration or clinical label remapping.

P1 uses train-only samples. Formal scope remains two training branches, three paired seeds (1993/1994/1995), three sessions (4+2+2), ten epochs per session, one post-lock batch test with four variants. Results remain NOT_RUN until verified artifacts exist.
