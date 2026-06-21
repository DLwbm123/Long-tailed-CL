# GPA Reproduction Pack for Codex

This pack contains the markdown files Codex should read **in addition to the GPA paper PDF**.

Recommended repository layout:

```text
repo/
  AGENTS.md
  docs/
    gpa_repro_notes.md
    GPA_PLAN.md
    CODEX_PROMPTS.md
    EXPERIMENT_CHECKLIST.md
    MEDICAL_LT_TRANSFER.md
    papers/
      GPA_ICCV2025.pdf
```

Immediate workflow:

1. Put `AGENTS.md` at the repository root.
2. Put all files under `docs/`.
3. Put the GPA paper PDF under `docs/papers/GPA_ICCV2025.pdf`.
4. Ask Codex to read `AGENTS.md`, `docs/gpa_repro_notes.md`, and `docs/GPA_PLAN.md` first.
5. Start with CIFAR-100-LT + Finetune + GPA smoke tests before attempting full baselines or medical datasets.

Do not ask Codex to implement LUCIR, PODNet, DAT, AIR, or medical transfer in the first patch. The first useful milestone is a minimal, reproducible CIFAR-100-LT pipeline with a working GPA plugin.
