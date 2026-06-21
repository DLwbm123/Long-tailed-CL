# GPA Single-Head Reproduction Scaffold

This workspace contains a single-head GPA reproduction scaffold under `src/gpa_singlehead_repro`. It uses GVAlign / Long-Tailed-CIL for CIFAR-100-LT data construction and evaluation protocol, but it does not use GVAlign's per-task multi-head classifier.

Status: archived as a negative reproduction scaffold. Current results do not reproduce GPA Table 1, so this code should be used only for diagnostics and non-GPA ablations such as prototype imprinting, zero-bias imprinting, old-row freeze controls, and cosine evaluation.

## Scope

- Dataset/protocol source: pass `--gvalign-root`, usually `/Users/bominwang/Desktop/codes/GVAlign` locally or `/dev/shm/wangbomin/GVAlign/code` on server 35.
- Model route: one shared single-head classifier over all seen classes.
- Methods: `finetune` and `finetune_gpa`.
- Exemplar setting: GVAlign protocol, 20 exemplars per class, herding.
- Default run scope: phase0 + phase1 only unless `--max-task 6` is passed.

## Example

```bash
bash scripts/run_cifar100lt_singlehead_phase1.sh \
  --gvalign-root /dev/shm/wangbomin/GVAlign/code \
  --data-root /dev/shm/wangbomin/GVAlign/data \
  --output-root /dev/shm/wangbomin/GPA_singlehead_repro/runs/phase1_finetune_gpa_s0 \
  --method finetune_gpa \
  --epochs 500 \
  --max-task 2 \
  --seed 0 \
  --gpu 0 \
  --num-workers 0
```

See `docs/GPA_PAPER_IMPLEMENTATION_SPEC.md` for method decisions and remaining ambiguities, and `docs/GPA_NEGATIVE_REPRODUCTION_REPORT.md` for the closure decision.
