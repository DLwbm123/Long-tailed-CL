# ISIC APART Transfer V1 — blocked protocol draft

Status: **DRAFT_BLOCKED_NOT_LOCKED**. No full experiment is authorized to start
under the failed P0 gates. [Audit and pending decisions](ISIC_APART_TRANSFER_V1_AUDIT.md).

The original design is retained unchanged: split1/suffix0.01; 8 classes; 4+2+2;
10 epochs/session; last epoch; batch size 48; APART ViT-B/16 adapter pool; no raw
image replay; B without ConCM statistics, C with the reference Stage1 module;
HN applied only by the eventual offline evaluator. M0/M1 share B checkpoints and
M2/M3 share C checkpoints. No test inference before all 18 checkpoints are locked.

| repeat | order/train seed | S0 | S1 | S2 |
|---|---:|---|---|---|
| r1 | 1993 | 4,0,3,7 | 5,6 | 2,1 |
| r2 | 1994 | 3,7,4,5 | 1,0 | 6,2 |
| r3 | 1995 | 1,3,0,6 | 4,5 | 2,7 |

Orders were generated once using independent `np.random.default_rng(order_seed)`
over sorted original numerical IDs. Each repeat's B/C configuration reads the same
order. `lt_list` is computed in that head order from the fixed train CSV.

The intended sequence remains r1-B, r1-C, r2-B, r2-C, r3-B, r3-C. All six rows
are currently BLOCKED_NOT_STARTED. Private `planned_configs/` files record the
unexecuted design; they are not resolved runtime training configurations. The
preflight actual optimizer settings are reported separately.

The predefined main outcomes remain Final_BA, Average_BA, Incremental_Average_BA,
tail_rank2 `[7,6]`, and the six planned paired contrasts with sample SD (`ddof=1`).
No result-based choice or metric change has occurred. All full-training and
four-variant comparison values remain unavailable.
