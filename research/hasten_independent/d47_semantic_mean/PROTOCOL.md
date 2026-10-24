# D47: TRAIN crossfit semantic-prior mean estimation

Frozen before first execution. R46 covariance shrinkage failed BOTH accuracy gates despite D45 density support. This distinct hypothesis concerns uncertainty of a rare class image mean, borrowing the fixed pretrained class text direction with an offset estimated from other classes. No covariance tuning or combination with R46.

Only original stage-zero TRAIN caches of NIH, ISIC and HK, exact manifest IDs and original leaf prompts. NIH image and text use the existing H-unit normalization. Per class odd/even manifest rows form two exchanged fit/holdout folds; no random draw, optimizer, neural forward, image encoding, VAL or TEST. This performs statistical estimation and TRAIN classification, not zero fitting or patient-independent validation.

For each fit class c, raw mean m_c and unit text t_c, use prior p_c=t_c+mean_{j!=c}(m_j-t_j). Let v_c be unbiased total sample variance, u_c=v_c/n_c, w_c=min(1,u_c/(||m_c-p_c||²+1e-12)). Candidate mean is (1-w_c)m_c+w_c p_c. If a fold has one fit row, use the equal mean of valid other-class total variances. Do not normalize the estimated mean itself; normalize only for native cosine NCM scoring. Covariance is not changed. Own-class image mean is excluded from offset fitting. Fixed rule is a heuristic, with no optimality claim.

Compare all-class native cosine NCM heldout TRAIN class counts with original sample means. Publish all 54 class×fold records, including weight and mean MSE; tail uses original TRAIN count<250. Support requires NIH macro and tail gains each >=1pp in both folds; ISIC/HK mean over two folds macro and tail each decline <=1pp. No subsequent threshold, weight, prompt, subset, seed or fusion scan. Failure rejects this fixed rule; success supports only independent training feasibility, not final accuracy.

If supported, the independent training hypothesis changes only Gaussian replay locations, preserving original covariance/statistics/NCM/projection and all other training. No combination with failed R46. This diagnostic score is not a proposed inference replacement.

Budget: original first preflight to final exit 600s, retries share clock; 300,000,000 bytes including 16MiB guard reserve, CPU2400s, RSS8GiB, file64MiB, CPU4. Shared caches read only. Independent neutral controller and worker; target ext4 and write/read probe, no historical deletion. Short completion before process capture will be disclosed.
