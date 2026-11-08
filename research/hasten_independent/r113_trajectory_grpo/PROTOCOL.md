# R113: online-prefix trajectory GRPO, development experiment

This is a new prospective experiment. R112 remains C. Its AUDIT oracle is not
training data, a deployable selector, or independent confirmation. The existing
TRAIN manifest has 6821 distinct image components; clinical patient independence
is unknown. Some classes have only three components. All R113 scores are therefore
development evidence, including outer scores from the previously analyzed AUDIT.
No VAL/TEST or new independent-population claim is opened.

## Hypothesis and arms

LONG_GRPO learns the value of memory choices across three genuinely adaptive
decisions within an already arrived prefix. It is evaluated in an actual five-stage
continuous stream. Its own model, optimizer and memory persist between real stages.
The pseudo-arrivals used for policy training are not real unseen future stages.
No R112 saved outcome table is stitched into a new trajectory.

Arms: ZERO, BAL, MYOPIC_GRPO, LONG_GRPO, SUP_SHARED. ZERO/BAL are unchanged
query-free reference rules, with necessary pinned evidence. Both RL arms train
on-policy from their own legally arrived prefix. SUP_SHARED receives exactly
LONG_GRPO's current group state/action descriptions and FIT feedback and imitates
the best complete trajectory (ZERO when none is positive). It has its own real
deployment model/memory. Thus SUP uses the identical feedback without additional
classifier executions; its feedback-generation cost is attributed to the shared
LONG source, not reported as free. No cross-arm historical feature bank is supplied
to SUP, only the transient current group's descriptors/feedback. These records
are discarded after their updates. There is no supervised-success admission gate.

Both existing streams (Ordered and stratified order0), seeds74003/74004 and real
stage sizes5+5+5+4+4 are retained. Every real arm starts from the same generic
classifier seed. Current birth encodings may be shared, but each arm decodes its
own retained history. No adapted R112 weights or old feature caches are loaded.

## Evidence, legal information, and candidates

Reuse R112 role seed947110, fixed hash-selected S-core/FIT/AUDIT globals and full
float32 bytes. Each arm's persistent evidence is <=256 tokens/528568 bytes;
FIT/AUDIT cost2C, leaving support capacity256-2C (210 at23 classes). No hidden
historical feature cache or new role assignment. Classifier/actor/optimizer
parameters are separately accounted model state, bounded by RAM/GPU/disk guards.
Multiple comparison arms are physically charged in the experiment's disk/RAM.

Keep the original ZERO/BAL plus risk1/2/4, cover4 and random2 edits, deduplicated
to <=12. Cores, parent coupling and anchor quotas remain protected. The R109
14-dimensional class and 10-dimensional action descriptors use S only, including
one S-only virtual classifier step. No learned verifier, reward model or AUDIT
prior. Actor is the existing 64-wide class-equivariant ResidualPolicy with uniform
prior/reference and zero-initialized residual head. Argmax ties select ZERO.
Training samples stochastically; deployment selects before any outcome feedback.

The initial candidate family is equivalence-checked against R112. At every later
trajectory decision, candidates/features are rebuilt from that trajectory's own
retained memory and updated classifier. Future continuation is learned, not fixed
ZERO. The common event/parent RNG couples external learning randomness.

## Policy training

At each real stage and seed, scenarios0/1 of the original arrived-prefix root
construction are used once for each RL arm. Each group executes one ZERO reference
and four on-policy complete H3 trajectories. Prefix classifiers are fit from
generic initialization on only currently accessible evidence. Each trajectory
uses100+200+100+200+100=700 real classifier updates, batch64, original rank16,
cosine head, AdamW lr.001/wd.001, KD T2. Policy-training endpoints evaluate FIT
only; AUDIT never enters these groups, labels, gradients, features or selection.

LONG return is original R=mean_h macro_delta−2*max_h[-oldtail_delta−.005]_+,
relative to the executed ZERO branch. The protected real-seen oldtail set is
birth_count<250 and born before the current real stage. No-tail has no tail
penalty. MYOPIC uses the corresponding per-h immediate reward, centered separately
at each h. LONG uses each whole-trajectory return at all its decisions.

Group-relative advantages use the four on-policy trajectories: (reward−groupmean)
/(population_std+1e−6). ZERO is an absolute reference, not a forced off-policy
GRPO sample. All-negative groups may have positive relative advantages; report
absolute returns and outer ZERO comparisons without calling these success.
Two full-group optimizer epochs, AdamW actor lr1e−4/wd.001, ratio clip.2,
exact categorical KL to the frozen uniform reference weight.01, grad clip1.
SUP uses two AdamW epochs at the same lr/wd, cross-entropy on the highest positive
FIT-return whole trajectory, or the executed ZERO trajectory. No replay or search
over hyperparameters. Actor normalization buffers stay fixed, not fitted on AUDIT.

## Actual stream evaluation and interpretation

Each arm has its own five-stage classifier/memory. Each real arrival uses200
pre-action and100 post-action updates. Learned arms choose using query-free
features and their current policy; no exhaustive outcome FIT argmax is deployed.
Policy updates see only arrived-prefix shadow feedback before the current real
action. Evaluation records the fixed AUDIT globals after each real stage, only
in an isolated scoring path. Future classes are never encoded early.

Report all100 stream×seed×stage×arm rows, absolute macro/tail BA, nonzero actions,
bytes, policy parameter/gradient changes and learning costs. For each stream/seed,
R_real5 is mean of five real-stage macro deltas against continuous ZERO minus twice
the worst evaluable real-stage tail deficit beyond.005. This is a newly specified
real-stream diagnostic, not R112's fixed-root average. A development signal requires
LONG R_real5 exceeding ZERO, BAL, MYOPIC and SUP in every stream/seed block and no
evaluable real-stage oldtail loss below−.005. Otherwise report which claims failed.
Never claim significance or clinical safety from repeated single probes. All
roots/seeds and negative results remain; no retuning from partial outer scores.

## Resource bound, native gate and delivery

One original24h clock covers preparation, checks, native admission and science;
120s stopping reserve. Exactly one full-shape native attempt per stream. Its
synthetic labels/repeated generic S features are engineering fixtures only.
Measure a3331-parent21→22→23 group with four policy trajectories plus ZERO and
all3 decisions. Each worker projects2*(40*group_seconds+71000*group_seconds/3500
+30*proposal_seconds+6821*encode30_seconds/30)+7200. Both projections must fit the
remaining original clock. Resource failure preserves artifacts; it never silently
shortens steps/groups or changes seeds.

Science upper bound212000 classifier optimizer steps per worker (424000 total),
including meta-prefix,3500/group, real streams and virtual steps. Two complete
TRAIN encodings total81852 rows, plus720 native rows. CPU checks are synthetic
engineering only. Preparation CPU is nonzero and may not be completely sampled.
Four total CPU cores, two per worker; workerRSS<28GB/groupRSS<29GB; GPU allocator
<12GiB with64MiB reserve; >=14GiB free GPU and64GiB available host RAM at launch.
Disk<20e9 bytes with512MiB reserve, stricter than the user's50e9 ceiling. One
independent controller guards every2s and writes one30-minute status receipt;
no duplicate monitor/process or automatic retuning. Original assets are protected.
Public delivery includes source/protocol/aggregate results and costs, excluding
private paths, image identities, features, actor/classifier weights and raw data.

GRPO reference: https://arxiv.org/abs/2402.03300 . This is a categorical
trajectory adaptation, not a claim that the original paper validated this task.
