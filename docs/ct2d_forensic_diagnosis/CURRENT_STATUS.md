# CT2-D status

COMPLETE_CT2D / STOP. Recovery and CPU report both exited 0. Analysis completed at 2026-09-18 13:49:20 UTC+8; live closeout check at 14:10 confirmed no remaining launcher/worker. No further experiment or periodic monitoring was started.

Completed: 12/12 final models, four fixed pre/post transitions, eight gradient states, 54 oracle metric rows and 837 per-class rows; original 270/2754 rows independently reproduced. Original CT1 assets remain unchanged. New neural training, optimizer steps and all test-access counters are zero. These are offline diagnostics, not legal replay-free CIL candidates.

The interrupted run's resource uncertainty remains explicit: conservative GPU residence upper bound 1.935 h, CPU analytic + engineering reserve + report upper bound 1.546 h. Confirmed analytic solves 79; true total lies in 79–81 because up to two completed solves may have lost their receipts at restart. Persistent diagnostic output was about 46.6 MiB at GPU completion; measured active peak 370.1 MiB; free disk at closeout about 2.4 GiB. No historical asset deletion was needed.

Closeout checks: table coverage, per-class-to-BA consistency, twelve reproduction receipts, original resource ceilings, zero training/test access, STOP, and public aggregate-field scan passed. Public delivery contains source/protocol, aggregate tables, audits and reports; raw images, identity maps, model/features/W and per-sample scores remain private.

Read FINAL_REPORT_ZH.md, BUGS_VS_DESIGN_LIMITS.md and RECOVERY_R1.md for findings, evidence limits and restart recovery details.
