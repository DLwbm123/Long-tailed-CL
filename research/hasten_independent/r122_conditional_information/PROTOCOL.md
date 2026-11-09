# R122: frozen TRAIN-only conditional-information intervention

Frozen before implementation/model calls, 2026-11-09. This is a single bounded diagnostic for the RL/TARS active-evidence mainline, **not new RL training, a successful method, or continual learning**. The minimal fixed-crop GRPO variant remains rejected. No fitting selector from all candidate correctness, no reward-based resampling, no old-run restart, no VAL/TEST access.

## New question and instruments

Can genuinely observed local pixels change a label-independent continuous region proposal so that its subsequently acquired evidence is more useful across disjoint TRAIN components, under the same visual call/token budget? This is an information intervention on a frozen proposal policy, not another R105 all-expert-feedback selector or a test that sampled RL beats exact optimization. A positive diagnostic only licenses further method design, not deployment or an RL-success claim.

Use the existing generic Qwen2.5-VL-3B frozen base as **proposer**, and the chronologically latest pre-RL R121 SEED_SFT as **recognition instrument**. Its choice is not based on best VAL. Both are frozen, no optimizer/SFT/refitting or downloading. The proposer uses generic language weights with task LoRA B=0 through an explicit reference context; recognition uses the saved seed74006 SFT. Same frozen vision. The instrument was fitted on all TRAIN, so this is in-sample measurement of the instrument; component disjointness is not independent patient validation or whole-chain crossfitting.

## Data and fixed component partitions

Only HK23 TRAIN6821 components. CPU generator seed12274006, each original class independently selects min(8,n_c) without replacement in fixed class order. Exactly175 unique components, no rewards used in selection. Alternate selected ranks into A/B within class:88/87 components, all23 classes in each. No fit, parameter or policy selection on A; the identical frozen intervention is measured in both partitions as a direction-consistency check. Original full-TRAIN head>=250/tail<250/tiny<25 definitions stay fixed. Private selected IDs and image/proposal/prediction records remain private.

## Label-independent observation and proposal

All rendered inputs224square,64visualtokens/image. For each source image, compare two proposal policies:

- GLOBAL_REPEAT observes [full image, identical full image], with accurate view2 box metadata(0,0,1,1).
- CENTER_DETAIL observes [full image, center crop(0.25,0.25,0.75,0.75)], with accurate corresponding view2 metadata.

Each uses exactly one deterministic native two-image proposer call, max64 newtokens, same prompt template except truthful view2 coordinates. Prompt asks for a next informative box in original image normalized0..1000; it contains no target label, class frequency, correctness or other policy's action. Text/coordinate metadata changes with the observation, so this is not an isolated causal claim about pixels alone. The repeated-full policy is a matched-information ablation, not a claimed strongest application baseline.

Fixed template: "Image 1 is the full endoscopy image. Image 2 shows the region of Image 1 with normalized coordinates {roi_as_four_three_decimal_fractions}. Using only these views, propose one region of Image 1 to inspect for recognizing the endoscopic finding or anatomical landmark. Return only a JSON array [x1,y1,x2,y2] of four integers normalized from 0 to 1000. The box must lie inside Image 1 and each side must be at least 100 units. Do not name the category."

Parser accepts a sole four-integer JSON array or sole {"bbox":[...]} object, optionally enclosed in one code fence. Reject booleans, nonintegral numbers, coordinates outside0..1000, reversed/empty boxes or width/height<100. Invalid/truncated proposals are retained, count as invalid, and execute the fixed full-image STOP fallback; no prompt/parser adaptation, retry or example dropping to improve validity. Pixel crop uses floor(left/top),ceil(right/bottom), then224resize.

Acquire each proposed crop **after** its proposal, then one greedy max24 response from the frozen recognition instrument using the original category-name-only prompt. Neither policy sees this future result before choosing. Also read fixed full-image and center-only recognition controls; they are cheaper descriptive controls, not equal-compute performance comparisons. Do not feed any counterfactual view/outcome into either proposal. Alternate policy execution order by selected row rank, same fixed parameters throughout.

## Counts and actual cost accounting

Per scientific component:2proposerresponses(2images each),4recognitionresponses(full/center/two acquired crops), total6responses/8visualencoderrows. Across175:1050responses/1400encodings. No new feature cache/checkpoint; raw private records small. Main paired branches each use2proposalimages+1recognitionimage, one proposal+one recognition call, same64/24generation caps. Record actual prefill/response tokens, truncation, timing, valid boxes and changed boxes; matching caps/calls is not a claim of equal actual generated tokens/wall.

Native exactly original TRAINrows0,1 (not chosen by reward): two single-image native/fullforward versus manual recognition parity paths,4encoderrows total; the full six-response/eight-encoding diagnostic matrix on both rows adds12responses/16encodings. Normal total therefore **1062responses/1420TRAINencodings,0optimizerupdates,0VAL/TEST**. Native records are excluded from scientific scores, updates do not exist to discard. The two-image proposal uses native HF generation only; do not claim manual multiimage-cache parity. Require native grids[1,16,16] twice/128image tokens and finite outputs, parser/crop fixtures, instrument/ref immutability and real full-matrix timing fit. Do not gate native on recognition accuracy or proposal benefit; invalid proposals follow the same frozen fallback.

## Frozen diagnostic gates, all reported

Compare CENTER_DETAIL minus GLOBAL_REPEAT using class-balanced component recalls: macro>=+1pp,tail>=+2pp,head>=−1pp; overall non-tiny18 macro>0; non-tiny18 macro>0 in **both** A and B; each proposal policy valid-box fraction>=.90; paired proposal+recognition branches have equal visualencoder counts and input-token totals, and CENTER_DETAIL total generated tokens<=GLOBAL_REPEAT. Report all ten gates/conjunction, all23class numerators/denominators, tiny denominators, both splits and all negative outcomes. Same actual wall time is not established by token counts; report it without claiming benchmark speedup. No use of VAL to choose proposals, thresholds, prompt or crop geometry.

Passing only suggests an input-information opportunity worth further RL-method design. It does not prove a learned controller, multi-step RL necessity, out-of-sample generalization or sufficient novelty over DeepEyes/PixelReasoner. Failure closes this fixed intervention without another prompt/ROI/seed sweep; preserve why it failed, not a blanket verdict on active vision. Oracle rescues, if later computed from the already stored paired outcomes, remain upper bounds and cannot replace the gates.

## Hard resources and completion

Only pro5000physicalGPU1/neutralmain.py argv, no my-gpu/otherprocess intervention. First preparation clock in private state, harddeadline4h later (<24h); all setup/native/retries/children share it. Independent2s controller stops deadline−120sec or cumulative conservative disk>=9GB/RSS>=48GB/freeFS<1GB, escalating process-group termination. All-window new physical strictly<10,000,000,000B: retainedR114–R1215898522624B+local200MBreserve+newroot including HF/Torch/CUDA/Triton/tmp/logs. Expected new<20MB/fullwindow<6.2GB, no model/features copied or downloaded. Filecap256MiB/core0/CPU4/GPUallocator56GiB/startFSfree5GB/hostavailable24GB; actualmount/free/write-readprobe/GPUfree checked before launch. Protect old data/models/failures, no deletion to reset accounting.

Use reliable neutral background controller. Full matrix native forecast=1.5*175/2*measured_two_component_matrix_seconds+600sec reserve must fit originalremainingclock before science. After actual CONTROL_EXIT0, CPU-only saved finalizer checks175selection/88-87split/private component pairing, source initialization/immutability,1062responses/1420encodings/0updates,all23rows per policy,ten gates/costs/fallback geometry and noVAL/TEST; never regenerate outcomes. Publish safe source/aggregates/report by established explicit7897proxy/allowlist/no-force/remoteSHA/anonymous200 workflow. No images/identities/individual proposals or predictions/features/weights/privatepaths/credentials/PDFs. Update the same existing thirty-minute task, no new heartbeat.
