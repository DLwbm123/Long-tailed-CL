# V2 status and results

**BLOCKED_FORMAL_BUDGET**. Data cleaning, fixed-weight verification and P1 engineering passed. The formal matrix remains **0/6 trajectories, 0/18 session checkpoints and 0 test predictions**. M0/M1/M2/M3 scientific metrics, paired effects and per-class transfer conclusions are NOT_RUN; no values are imputed from smoke accuracy or historical CIFAR results.

The conservative six-trajectory estimate is **47.44 GPU hours**; an explicit cold-start-amortized sensitivity estimate is **32.33 hours**, versus the approved **24 hours**. GPU memory is adequate (observed P1 peak 8.45 GiB on physical GPU3, A100-PCIE-40GB). No experiment was canceled for low performance.

Data: train/val/test **18718/295/764**, image imbalance **389.96:1**, documented-lesion-disjoint with UNKNOWN patient isolation and selection bias from missing lesion IDs. V1 stays BLOCKED, unchanged.

See V2_DATA_CLEANING_AUDIT.md, WEIGHTS_LOCK.json, TRAINING_SEMANTICS_LOCK.json, P1_TECHNICAL_REPORT.md, resolved_formal_configs.json, class_orders.json, budget_audit.json and resource_usage.json. Planned rows are explicitly labeled NOT_RUN. Private manifests, metadata, images, weights and smoke checkpoints are excluded from the repository.

**Minimum pending decision:** whether to increase the six-trajectory budget from 24 to 48 GPU hours while retaining all other approved settings. No formal protocol lock or test release has occurred.
