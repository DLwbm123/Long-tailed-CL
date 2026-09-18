# CT3-P status

BLOCKED_RESOURCE, observed 2026-09-18. No active experiment worker and no final evaluation. Controls passed on repair R1; training stopped at ISIC seed1994 Task3 epoch2 during rolling checkpoint save. The fixed matrix is incomplete: 92/120 recorded task-epochs, 1956/6620 optimizer steps, 9/12 archived task checkpoints. Four of six trajectories have both task checkpoints; ISIC1994 has Task2 only and ISIC1995 has not started.

The previous rolling checkpoint and its fully written temporary successor each occupy approximately 646 MiB. With both present, disk free fell to 913,543,168 bytes, below the required 1 GiB; the conservative active-plus-archive measure reached 3,262,427,454 bytes, above 3 GiB. These are engineering storage blockers, not measured scientific failure. The original resource limits remain binding.

Both files were CPU-readable, delta tensors were finite, and optimizer/scheduler/RNG fields were present. The temporary successor contains Task3 epoch2, while the previous committed file contains epoch1. Full strict model restore has not yet been verified. No checkpoint was deleted or promoted; no training was restarted during this status audit.

Recorded cumulative GPU residence: 6,138.312 seconds (1.705 hours), excluding any unrecorded process initialization overhead. Additional CPU inspection: 6.607 seconds. All recorded test access and online old/future fit reads remain zero. No online candidate scores, offline Q11 results, or completed-effect claims are available.

Next technical requirement: resolve rolling-save transient storage within the unchanged caps, verify the latest state under the same lock, and recheck remaining budget before any continuation. Evidence is in `resource_stop_r1/`. Previous failed ACK-race evidence remains in `attempt0/`; scientific settings and source are unchanged.
