# Codex Start Prompts

Copy one prompt at a time into Codex. Start from Prompt 1.

---

## Prompt 1 — Repository inspection only

```text
Read AGENTS.md first.

Then read docs/gpa_repro_notes.md and docs/GPA_PLAN.md.

Do not edit code yet.

Inspect the repository and write a concise Plan for implementing the CIFAR-100-LT + Finetune + GPA smoke pipeline. Your Plan must list:
1. files you need to read,
2. files you will create or modify,
3. local validation commands,
4. risks or assumptions.

Do not implement LUCIR, PODNet, DAT, AIR, or medical datasets in this first step.
```

---

## Prompt 2 — Implement CIFAR-100-LT dataset preview

```text
Follow AGENTS.md and docs/GPA_PLAN.md.

Implement only Phase 1: CIFAR-100-LT dataset builder and task split preview.

Requirements:
- rho means N_min / N_max.
- CIFAR-100-LT with rho=0.01 should have max class count about 500 and min class count about 5.
- Support ordered and shuffled class orders.
- Support 50 base classes and 5/10 incremental tasks.
- Save class_counts.json and class_order.json.
- Add a preview command or small test.

Before editing, state your Plan. After editing, run the validation command and summarize the result.
```

---

## Prompt 3 — Implement Finetune smoke baseline

```text
Implement Phase 2 only: Finetune LT-CIL baseline.

Requirements:
- sequential tasks;
- no replay buffer;
- expand classifier for new classes;
- train only on current task data;
- evaluate on all seen classes;
- save metrics.jsonl and an accuracy matrix;
- local smoke mode must finish quickly.

Do not implement GPA yet unless the Finetune baseline already passes.

Start with a Plan, then make the smallest patch, then run a local smoke command.
```

---

## Prompt 4 — Implement GPA plugin

```text
Implement Phase 3: GPA plugin for the existing Finetune baseline.

Requirements:
- add --gpa true/false;
- compute frozen class prototypes before each incremental task using the previous model in eval mode;
- initialize new class classifier weights with normalized prototypes;
- optional bias initialization with --gpa_init_bias;
- add dynamic anchoring loss with --lambda_gpa;
- keep --gpa false behavior identical to baseline;
- log prototype norms and anchor loss.

Start with a Plan. Implement the smallest patch. Run both smoke commands:
1. Finetune without GPA
2. Finetune with GPA

Summarize what changed and whether metrics/logs were produced.
```

---

## Prompt 5 — Add summary script and metrics

```text
Implement Phase 4: result summary.

Requirements:
- final accuracy;
- average incremental accuracy;
- forgetting;
- many/medium/few accuracy;
- head-tail gap;
- output summary.json and summary.csv;
- compare two or more run directories.

Start with a Plan, then implement and run it on the smoke runs.
```

---

## Prompt 6 — Prepare server commands

```text
Do not change core training code unless necessary.

Prepare scripts or configs for full CIFAR-100-LT server reproduction:
- shuffled 5-task Finetune and Finetune+GPA, seeds 0/1/2;
- shuffled 10-task Finetune and Finetune+GPA, seeds 0/1/2;
- optional ordered setting;
- optional lambda_gpa sweep.

Create shell scripts under scripts/ or configs under configs/.
Do not run full experiments locally.
Run only a dry-run or config validation.
```

---

## Prompt 7 — Medical long-tailed transfer, later only

```text
Read docs/MEDICAL_LT_TRANSFER.md.

Do not start medical transfer until CIFAR-100-LT Finetune+GPA is stable.

Create a Plan for adapting the verified GPA pipeline to a medical long-tailed classification dataset. The first medical experiment should preserve the same CIL machinery and add only a dataset adapter plus medical metrics.
```
