# R111: input readiness before conditional P1b

This protocol is fixed before fresh TRAIN encoding. It does not reopen R110.
Donor seeds: 74003 and 74004, both original streams. All sixteen stage/scenario
slots per seed remain in the denominator; no replacement seeds or favorable roots.
Patient independence remains unverified. VAL and TEST are sealed.

## Necessary information gate

For **each stream and each seed**, require at least four of sixteen scenarios,
at one or more actual middle/late snapshots, simultaneously satisfying:

1. A nonempty protected set of every real-seen old-tail class (TRAIN count <250,
   arrived strictly before the current stage).
2. At least one retained legal component in S, FIT and AUDIT for every protected
   class. Missing support or either query role is unknown. No dropping classes.
3. A shadow root pool strictly larger than 256 tokens and a concrete distinct
   ZERO/BAL anchor or original legal random two-token local edit.

Four scenarios from one snapshot are not four independent patients. This gate
checks availability, not statistical power, accuracy or a method effect.
Failure of necessary input coverage permits skipping expensive prefix/action
construction for that slot; it does not count as a passing no-op.

## Query-free donor equivalence

The frozen R110 uniform ZERO transition depends on raw geometry, token coupling,
capacity and its RNG, not learned scores. Reuse its exact transition and geometry
initialization with inert state embeddings. Verify RANDOM and BALANCED token
equality against the original constructor, using two synthetic classifiers,
two seeds, full/partial views and underfilled/full capacities, on CPU and GPU.
No real classifier is initialized, trained or used for query prediction.

Encode each real current arrival once with the original frozen generic encoder
and six deterministic views, shared between the two seeds only while current.
Each donor advances with its own retained 256-token memory and RNG seed
donor_seed+70000+stage. Discard current features after both donors advance. No
recovery of old images, discarded views, prior donor features or full caches.
Role seed remains 947110; scenario seed 75110+100*stage+scenario; prefix ZERO
compression seeds remain scenario_seed+2000+prefix_phase. Full future H3 risk
coverage follows from the protected old classes being present in each role;
the two pseudo-future classes are current arrivals, outside the old-tail set.

## Cost and next step

At most 81,852 new generic encoder rows across both streams; zero real classifier
forward, optimizer, actor or verifier updates. Synthetic equivalence checks are
engineering cost and separately recorded. One guarded two-worker run uses four
CPU cores and two GPUs, each at most 12 GiB reserved, worker RSS <28e9 bytes,
group RSS <29e9, new disk <20e9 with 512 MiB stopping reserve. Original R111
clock includes A, preparation, readiness and any conditional P1b; hard 24 hours,
120 seconds termination reserve, 400,000 CPU seconds and 1 GiB per file. Guards
run independently of this chat. No automation is created.

Only after this necessary gate passes may the six-control scientific P1b protocol
be frozen and its complete native resource admission run. P1b retains B=256,
H=3, both tapes, both seeds/streams, all slots and original training steps. Do not
shorten the experiment, reduce memory or resample to obtain admission. A failed
input gate stops P1b and requires a separately specified task construction. No
automatic verifier/actor/RL experiment is authorized by this protocol.
