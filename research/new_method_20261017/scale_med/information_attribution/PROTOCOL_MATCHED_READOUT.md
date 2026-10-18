# Fixed matched-regularization attribution follow-up

2026-10-18. Authorized in the user's four-hour autonomous window, ending11:54:26UTC. This is an exploratory follow-up motivated by a real confound, not a replacement for the completed seven-arm test or a performance repair.

Original HK G0/G1 selected alpha=.0001; M/N/S/K selected.01. ISIC all selected.01. Consequently HK M-G1 changes both information and penalty. Freeze a single common alpha=.01: the already support-selected N reference penalty. No candidate search, extra seed, dimension or test-based selection. Refit only HK G0 and G1 from sealed base+new sufficient statistics with exactly their original normalization, weights, intercept and objective. All other original heads and all ISIC results remain as previously reported. Never access old support images or per-image caches; no new fitting examples. Original B remains frozen.

Score HK1271test rows in the identical order. Reuse gB test cache, extract same frozen raw CLS only. Before accepting new metrics, reproduce existing G0/G1 scores at their original saved heads within1e-5 and identical predictions; do not relax tolerances on failure. Seal the two matched heads before scoring. Test is already exposed and this follow-up is exploratory.

Report original versus matched G0/G1, B and M. Four prespecified paired comparisons: G0matched-G0original, G1matched-G1original, G1matched-G0matched, M-G1matched. All/old/new BA,perclass/confusion,micro,CE descriptive,rescue/harm; paired within-class exact-identity-group bootstrap10000seed74003. No simultaneous or blind confirmation claim. No more settings based on results. Explain any old/new tradeoff and small-class sensitivity.

Stop after this diagnostic. The first-round second-order primary gate remains failed whatever happens here. A changed conclusion in the secondary chain means selection sensitivity, not evidence that local second-order information wins. Publish aggregate evidence, costs and analysis; protect features, model weights, identities and private paths. Cost excludes original statistics/cache creation and does not represent end-to-end latency.
