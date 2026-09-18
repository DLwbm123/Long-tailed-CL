# CT3-P status

RUNNING after bounded engineering repair R1, dispatched 2026-09-18T19:23:36.811551+08:00. Not complete; formal training was zero steps and zero epochs at restart.

Attempt0 stopped in controls after 376.469 seconds: non-atomic ACK publication exposed an empty JSON file to the reader. The producer now writes a temporary file then atomically renames it. A concurrent fragmented-write regression test passed 50 writes. Training/statistics code and all scientific settings are unchanged. Execution source: `d1eaa599a23679d025659a6d1b45a43996291b86`; repair lock supplements the preserved original protocol lock.

Failure receipts, logs, candidate banks and baseline features remain in the server's `attempt0` directory. Only redundant local copies of original archived parent checkpoints were removed. No historical asset was deleted. Failed GPU residence remains charged: 828.448 seconds before restart; full replay admission is 13,867.741 seconds within the cumulative 14,400-second ceiling. Nine discarded engineering optimizer steps, no additional engineering update.

The detached single-worker driver executes controls, six fixed FD trajectories (120 task-epochs, 6,620 steps, 12 checkpoints), locked online evaluation, four isolated oracle diagnostics, and reports, then STOP. No test access, periodic monitor, or automatic retry. Logs: authorized hb01 `/tmp/p22root/output/`. Final results still require completion verification and publication.
