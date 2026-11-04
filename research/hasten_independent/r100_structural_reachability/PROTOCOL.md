# R100: structural action reachability and selection transfer

Prospective mechanism check, followed by conditional RL only if every admission gate passes. No new optimizer or image encoder in this first check. R98/R99 are closed and unchanged. This is a bounded structural diagnosis, not a claim of a novel method or of RL benefit.

## Identity, information and clock

Seed74002; all six original stages. Reuse the R97 frozen original R67 teacher features, original teacher class prototypes and immutable R91 S/B/F roles, with ordered manifest/component checks. Teacher already saw all TRAIN and F has historical development exposure: F here is a module audit, not independent confirmation. VAL/TEST are disabled in this check. No relabeling, resampling tiny cases, new role split, or endpoint search. Original FIRST includes preparation, implementation, engineering, checks, conditional fit, analysis and delivery;24h never resets. All large/new files on the data disk;8GB science output limit with512MiB reserve and strict<50GB/server. Worker28GB/group29GB, CPU4/400000CPU seconds/file1GiB, GPU reserved<12GiB; native device identity and sufficient free memory required. Precheck worker hard limit1800s; conditional training requires fresh complete-plan measured budget within the same original clock, otherwise NOT_ADMITTED_BUDGET. No monitor.

## Fixed action construction and deployment

Baseline KEEP is the original strong teacher, not SUP0 and not the newer R97 SUP. Its full-history information budget remains disclosed and identical for all arms. KEEP returns the exact same score columns. No new strong representation training.

Per class retain at most20 S examples, with at most one example per recorded identity component. Order/ties use feature-vector lexicographic order, not class/sample IDs. Start at the point nearest the class centroid, then farthest-first by minimum squared Euclidean distance. No B/F label enters construction. Classes require at least4 distinct S components for SPLIT. Rank anchor pairs by decreasing squared distance. Assign each bank example to its nearer anchor; each cluster must contain at least2 distinct components. Retain the first two different unordered valid partitions; no candidate search using query accuracy. Two normalized cluster means form a SPLIT action. A tiny class with insufficient evidence has no own split action.

Raw class score is20*cos. SPLIT score is logmeanexp of its two raw scores, temperature1, residual amplitude1. KEEP is an explicit bypass, so no numerical reconstruction error. LME normalization removes the simple log(K) term but does not guarantee absence of score or capacity bias. All methods use the same banks/partitions/temperature. No CORRECT/builder is added and no R91 fit is resumed.

For each stage freeze four disjoint two-class blocks, greedily choosing the closest teacher-prototype pair among classes that have two valid SPLIT candidates. Ties use prototype vectors. If fewer than8 classes qualify, stop NOT_ADMITTED_STRUCTURE_COVERAGE; do not change blocks or evidence requirements. For each block enumerate exactly9 complete joint plans (KEEP/SPLIT1/SPLIT2 per class), scoring all arrived classes on the entire B and entire F query sets. Other classes remain teacher KEEP. Thus at most2 extra prototypes, within total capacity ceil(1.5C). This is local joint reachability, not a bound on every larger structural policy.

Compare: NOOP; fixed SPLIT1 on both classes; uniform expectation over the9 plans; best plan on B; best plan on F (diagnostic oracle only). B selection maximizes hard reward below; ties prefer fewer modifications, then predetermined plan order. Never select using F. Oracle is the best single complete plan, not per-query selection. Report reachable oracle benefits and harms, actual prediction changes, and B-selected F benefit versus all fixed references. Limited-candidate failure rejects this candidate bank, not all multimode representations.

## Reward and admission gates (fractions internally, percentage points reported)

R = delta macro accuracy - max(0,-delta old macro accuracy) - max(0,-delta tail macro accuracy), each relative to the exact teacher in the same support/query context. Missing old group in stage0 has zero penalty. No margin shaping, group-standard-deviation normalization, or confidence-only credit. All scoring includes full class competition. Old/tail risk coefficients1 and tolerance0 are frozen, not tuned after B/F.

Aggregate the four blocks equally within each stage, then six stages equally. Gates conjunctive:

A reachability: B-selected mean macro gain>=0.5pp, mean tail gain>=0, and at least4/6 stages with both nonnegative. There must be genuine B rescue exposures.

B transfer: the exact B-selected plans have F mean macro gain>=0.3pp, mean tail gain>=0 and at least4/6 stages both nonnegative. Their F mean macro/tail differences against BOTH the fixed SPLIT1 plan and uniform expectation must be nonnegative (1e-8pp tolerance). No additional harm exposure for F teacher-correct tiny classes1/2/12. These repeated contexts are not independent patients or repeats.

No gate is softened, positive block chosen, candidates replaced or old endpoint rescanned following failure. Negative scientific gate closes this experiment without actor training. All aggregate signs, denominators, costs and failure scope are delivered.

## Conditional RL after all checks pass

Before any optimizer, freeze the complete fit config and measured total runtime under the same FIRST. Keep teacher/candidate constructor frozen, keep the same banks and LME scoring at fit and deployment. Fit a shared structure policy, with a fixed deterministic decoder, no label/F access in policy state. RL samples complete feasible plans, G8, hard reward and leave-one-out baseline; constant groups produce zero advantage without resampling. Do not reuse independent-class counterfactual gradients for dependent sequential choices without replaying their legal suffixes.

Retain strong NOOP, same-capacity fixed structure, B-greedy, same-state/network full-information supervised CE selector, full expected hard reward when enumeration is feasible, and state/credit-mismatch control. Same initialization/data/capacity/budget and deterministic deployment rule. Keep supervised training in the main arm; quantify expected-policy versus deployed-plan gaps. All fitted endpoints sealed before final VAL once, TEST0; existing F is audit/development, not selection. No actor is allowed if the two admission gates fail. This conditional section specifies obligations, not an already admitted or running fit.
