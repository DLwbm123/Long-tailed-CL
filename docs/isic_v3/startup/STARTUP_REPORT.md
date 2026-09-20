# V3 startup audit

Status at the startup check: RUNNING / P0, not complete. Frozen runtime source: `8ac23f4808e3b3ec9924c89b3b81dd2cd7c9bd69`; the code and public branch match. No merge into main.

The default current-only path produced exactly the original V2 loss, parameter gradients and one-step updated network. All-seen candidates gave nonzero old-head gradients in all three real CE terms and zero future-class CE gradients. Actual inference was independent of supplied class-frequency inputs. Original-backbone native features were identical; zero-adapter relative L2 error was 1.63e-6. The supplied JSON files contain the numerical checks.

The initial conservative projection is 23,460.9 GPU process-seconds (6.52 hours), including a 50% reserve, against 28,800 seconds. Before the complete P2 matrix the pipeline measures its first full formal epoch and rechecks budget and peak memory; that epoch is resumed, not discarded or repeated. Maximum concurrency is two.

The initial launch was stopped before P0 because its disk projection reserved the unavailable DINOv2 weight/cache. That failure and its original scope lock were preserved. Execution revision 2 accounts for the authorized partial-backbone scope: 16,955,367,424 bytes available versus 16,717,192,892 required, including 12 final checkpoints, six resumes, two atomic write buffers, 128 MiB auxiliary space and a further 1 GiB reserve. No historical deletion, retention or precision change occurred.

DINOv2's pinned download returned Network is unreachable on the compute host and a connection timeout on the local fallback. Status is PARTIAL_BACKBONE_COMPARISON; there is no substitute. A and P2 proceed under the approved exception.

The background pipeline performs P0, P1, all six D/E S0 forks after the final gate, aggregate validation reporting, and stops. New model test predictions remain zero. V2 COMPLETE_MIXED_SIGNAL and all original images/manifests/weights/checkpoints/predictions are protected; the five pre-existing dirty CSVs in the historical worktree were not changed. No P3, Joint Training, hyperparameter search, Full Dynamic or monitoring automation was started.

These are startup/engineering artifacts, not performance results or a claim that P0/P1/P2 are finished. Full validation tables and conclusions are produced by the bounded pipeline and will be audited for completed delivery.
