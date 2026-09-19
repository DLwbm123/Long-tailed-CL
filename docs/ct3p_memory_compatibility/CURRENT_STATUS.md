# CT3-P status

COMPLETE_CT3P_MIXED_SIGNAL. Hourly long-term monitoring is ACTIVE (`isic`) under the user's new autonomous monitoring, maintenance and future-experiment authorization. All four R2 phases exited zero; worker and transfer service stopped. Final report and complete aggregate results are available.

The latest ISIC1994 Task3 epoch2 checkpoint passed the ordinary strict model, metadata, code/data hash, teacher, optimizer/scheduler and generator-state checks. It was promoted from the fully written temporary file; the obsolete epoch1 rolling file was removed. All 92 recorded epochs / 1,956 optimizer steps remain intact, with nine archived task checkpoints. No formal epoch was repeated and no new engineering optimizer step was used.

A full-field CPU roundtrip established that pickle protocol4 serializes precisely the same checkpoint values using 521,111,356 bytes instead of 676,882,789. The recovery entry changes storage encoding only; the qualified training/loss/statistics source is unchanged. An executable serialization regression check is included. Original ACK failure and resource-stop evidence remain preserved.

Deleted 91,030,090 bytes of obsolete failed-attempt banks/features after the successful controls had superseded them. Migrated V3 1993_D to `jiangsuiyang:/data_nas/jiangsuiyang/LongTailedCL/v3_q8m8/output_r2/1993_D/`; verified successful stream completion, matching 11-file inventory and known sizes, and readable checkpoint ZIP directories, then removed only the two redundant source checkpoint copies. This released 1,512,585,638 bytes. No bytewise equality is claimed. Runtime disk free is now approximately 3.0 GiB.

Updated measured-throughput admission projects 14,659.477 cumulative GPU seconds (4.072 hours), above the existing 14,400-second limit. On 2026-09-19 the user explicitly instructed “不要限制 GPU 小时，跑完就行”. This supersedes the original GPU residence cap for the fixed CT3-P matrix; GPU_limit_seconds=null, while cumulative accounting, storage/data/CPU guards and fixed training scope remain. The earlier BLOCKED admission and failure receipts remain preserved. Current charged residence is 6,177.804 seconds; remaining fixed work is 28 task-epochs, 4,664 steps, three task checkpoints, full locked online evaluation, four isolated oracle fits and reporting. All recorded test and online old/future fit access remains zero.

Bind this approved resource-only amendment and recovery source/hash, recheck GPU/storage, and run `tools/execute_ct3p_recovery.py` through the neutral entry using the existing bounded transfer service. Preserve original protocol and failure receipts. The current plan's methods, data, seeds and length remain fixed. Follow-up studies may be planned and run autonomously only in separate frozen plans after the complete CT3-P analysis; see AUTONOMOUS_RESEARCH.md.

## Connection update 2026-09-19

The user supplied replacement SSH port 30128 for hb01-ssh.gpuhome.cc. Login succeeded; the original runtime directory, strict recovery receipt and 676,882,789-byte active resume file are present. GPU is RTX 4090 D, 0/24,564 MiB used at this check; disk free is 3,193,294,848 bytes. No worker, approved recovery lock or completion receipt exists. The user subsequently removed this study’s GPU-hour limit; recovery is being prepared. Future monitoring must use port 30128; the obsolete port 30154 is not trusted. This check confirmed file presence/size and existing receipts, not a new full checkpoint integrity test.

## Recovery R2 started 2026-09-19

Source d42b1f1b9ae33408aa90b1b8fc38639b9a283861 binds the explicit no-GPU-hour-limit amendment. RECOVERY_LOCK_R2 is approved and RECOVERY_ADMISSION_R2_APPROVED is PASS; the original BLOCKED admission remains historical evidence. Strict restore passed model/teacher/optimizer/scheduler/RNG checks; ISIC1994 Task3 resumes at epoch3 with zero repeated steps. Neutral driver PID554, worker PID555; startup GPU19,257 MiB and100% utilization. my-gpu archive service PID2360416 passed real NFS mount/write/read probe and uses port30128. Pipeline train→evaluate→oracle→report is detached; future hourly checks must inspect train_r2/evaluate_r2/oracle_r2/report_r2 exit files, PHASE_RECEIPTS_R2, PIPELINE_COMPLETE or failure receipts. Do not launch another worker. All prior consumed time remains charged; no GPU residence ceiling, original CPU/storage/access guards retained.

## Completion 2026-09-19

Verified120 epochs/6620steps/12checkpoints,90/360online rows,4/24offline rows. AllGPU workers and transfer service exited. GPUledger14348.926seconds, with separate conservative process-wall reconciliation in results/DELIVERY_AUDIT.json. No test access; online old/futurefit0; offline oldfit27584. Main ISIC C3−C1 +5.584pp with old/current tradeoff; HK no gain overP. This fixed study is closed with STOP. Next autonomous work, if any, must have its own branch/plan and cannot resume completed CT3-P training.
