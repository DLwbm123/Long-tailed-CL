# V2 data cleaning audit

Status: V2_DATA_SUPPORT_PASS. Original images and CSVs were not edited. V1 remains BLOCKED with 0/6 training trajectories and zero model test predictions.

Protocol: `isic19_fopro_s1_documented_lesion_disjoint_v2`. Candidates are exactly the original split1 train/val/test union. Identity components use verified namespaced lesion IDs or equal file SHA256, with transitive closure. Conflict components are quarantined before unknown-lesion exclusions. Original-component priority is test > val > train even when its highest-priority record is later excluded. Equal-content survivors retain only lexical-minimum image ID. No split reassignment, quota filling or long-tail resampling.

Original candidates: 24645; retained: 19777. Train/val/test: 18718/295/764.

| Finding ID | Train images | Val images | Test images | Train lesions | Val lesions | Test lesions | Retention |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10529 | 44 | 86 | 6552 | 44 | 86 | 82.7883% |
| 1 | 3263 | 37 | 86 | 1018 | 36 | 84 | 74.8784% |
| 2 | 2835 | 46 | 100 | 1168 | 45 | 97 | 89.7081% |
| 3 | 1306 | 40 | 92 | 799 | 37 | 91 | 74.2002% |
| 4 | 458 | 41 | 100 | 199 | 38 | 87 | 69.0888% |
| 5 | 256 | 37 | 100 | 144 | 34 | 84 | 62.5796% |
| 6 | 44 | 26 | 100 | 33 | 24 | 78 | 67.1937% |
| 7 | 27 | 24 | 100 | 23 | 20 | 70 | 63.1799% |

Excluded: 4 label-conflict, 2791 unknown-lesion, 2059 lower split priority, 14 duplicate-content records. Each original record has exactly one disposition in the private audit.

Training image imbalance: 389.962963:1; lesion imbalance: 284.869565:1. The old 0.01 suffix does not denote the actual post-cleaning ratio. Frozen groups by (training count, finding ID): tail [7,6], mid [5,4,3,2], head [1,0].

All 12 cross-split intersections (image ID, file SHA256, lesion ID, component ID) are empty. All classes exceed minimum support; none has fewer than 20 independent test components. Retained lesion coverage is 100%. No pixel/perceptual deduplication was added.

V1 decoded and hashed all original images. V2 reused that evidence after validating CSV identity and metadata SHA and checking unchanged image mtimes/nonzero sizes. This is not a fresh bytewise image verification. Remote transport and train readability are reported in P1.

Limitations: documented lesion isolation is not patient isolation (UNKNOWN). Excluding missing lesion IDs changes cohort coverage and may create selection bias. Development exposure is UNKNOWN. Disease names remain SEMANTIC_MAP_UNVERIFIED; finding IDs are retained. New ImageNet provenance is declared but target-image pretraining overlap is not independently disproven. This is an internal migration experiment, not external clinical validation.

Private: per-image manifests, disposition/component/lesion IDs, images, weights, predictions and checkpoints. Public: this aggregate audit, class counts, protocol hashes and method code.
