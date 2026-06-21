# GPA Single-Head Reproduction Scaffold

This workspace is a clean paper-faithful GPA reproduction scaffold. It uses GVAlign / Long-Tailed-CIL for CIFAR-100-LT data construction and evaluation protocol, but it does not use GVAlign's per-task multi-head classifier.

## Scope

- Dataset/protocol source: `/Users/bominwang/Desktop/codes/GVAlign`
- Model route: one shared single-head classifier over all seen classes
- Methods: `finetune` and `finetune_gpa`
- Exemplar setting: GVAlign protocol, 20 exemplars per class, herding
- Default run scope: phase0 + phase1 only unless `--max-task 6` is passed

## Example

```bash
bash scripts/run_cifar100lt_singlehead_phase1.sh \
  --gvalign-root /Users/bominwang/Desktop/codes/GVAlign \
  --data-root /dev/shm/wangbomin/GVAlign/data \
  --output-root /tmp/gpa_singlehead_phase1 \
  --method finetune_gpa \
  --epochs 500 \
  --max-task 2
```

See `docs/GPA_PAPER_IMPLEMENTATION_SPEC.md` for the method decisions and remaining ambiguities.

## Update in this package

The GPA anchor term has been corrected to match the paper-level mechanism: it regularizes classifier rows against normalized moving-average prototypes. The previous scaffold penalized feature centroids against frozen anchors, which is not the GPA objective and can damage both old and new classes.

See `docs/GPA_SINGLEHEAD_IMPLEMENTATION_NOTES.md` for the exact implementation and recommended phase1 gate.
