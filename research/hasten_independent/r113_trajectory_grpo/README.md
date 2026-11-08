# Continuous memory-decision GRPO: current method and evidence

Snapshot: 2026-11-08T23:06:50+08:00. R113 revision2, source `fb2327ee0399ee5b2decdf01ae44ea1be38b1ff4`.

Ordered: all5 real stages complete; LONG_GRPO and ZERO both56.52% terminal macro BA,64.29% terminal oldtail BA, R_real5=0. order0: first3 real stages complete,30/40 training groups; latest macro BA73.33% versus ZERO70.00%, but oldtail25.00% versus37.50%. LONG and MYOPIC have identical observed endpoint scores so far. These are development-only fixed-probe scores, not independent confirmation. The full experiment is still running.

[Chinese method/results report](REPORT_ZH.md) · [Snapshot metrics](CURRENT_SUMMARY.json) · [Stage counts](CURRENT_STAGE_RESULTS.csv) · [Seed/arm summary](CURRENT_BLOCK_SUMMARY.csv) · [Saved-only verification](CURRENT_VERIFICATION.json)

The policy makes all3 decisions in arrived-prefix training trajectories, then acts in its own continuous5-stage deployment stream. Arms: ZERO, BAL, MYOPIC_GRPO, LONG_GRPO, SUP_SHARED. FIT trains; AUDIT scores only. Total evidence256 feature tokens including paid probes; no future real-stage encoding. See [protocol](PROTOCOL.md), [configuration](CONFIG.json), and [implementation repair](IMPLEMENTATION_REPAIR.json). Attempt1 failure is preserved separately; no partial outcomes are pooled.
