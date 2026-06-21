#!/usr/bin/env python3
import argparse
import copy
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gvalign-root", default="/Users/bominwang/Desktop/codes/GVAlign")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset", default="cifar100_lt")
    parser.add_argument("--method", choices=["finetune", "finetune_gpa"], default="finetune_gpa")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-tasks", type=int, default=6)
    parser.add_argument("--max-task", type=int, default=2)
    parser.add_argument("--nc-first-task", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr-factor", type=float, default=10.0)
    parser.add_argument("--schedule-step", type=int, nargs="+", default=[250, 350, 450])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--clipgrad", type=float, default=10000.0)
    parser.add_argument("--num-exemplars-per-class", type=int, default=20)
    parser.add_argument("--exemplar-selection", default="herding")
    parser.add_argument("--lambda-gpa", type=float, default=0.12)
    parser.add_argument("--gpa-anchor-momentum", type=float, default=0.9)
    parser.add_argument("--gpa-anchor-reduction", choices=["sum", "mean"], default="sum")
    parser.add_argument("--gpa-eps", type=float, default=1e-8)
    parser.add_argument("--gpa-nref", choices=["max_current", "mean_current", "first_current"], default="max_current")
    parser.add_argument("--gpa-bias-mode", choices=["paper", "zero", "old_mean", "none"], default="paper")
    parser.add_argument("--eval-mode", choices=["raw", "no_bias", "cosine_eval", "norm_weight_eval"], default="raw")
    parser.add_argument("--eval-calibration", action="store_true", help="Also evaluate raw/no_bias/cosine/norm-weight at the final phase.")
    freeze_group = parser.add_mutually_exclusive_group()
    freeze_group.add_argument("--freeze-old-rows", dest="freeze_old_rows", action="store_true")
    freeze_group.add_argument("--no-freeze-old-rows", dest="freeze_old_rows", action="store_false")
    parser.set_defaults(freeze_old_rows=True)
    restore_group = parser.add_mutually_exclusive_group()
    restore_group.add_argument("--restore-old-rows", dest="restore_old_rows", action="store_true")
    restore_group.add_argument("--no-restore-old-rows", dest="restore_old_rows", action="store_false")
    parser.set_defaults(restore_old_rows=True)
    return parser.parse_args()


def seed_everything(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run():
    args = parse_args()
    import numpy as np
    import torch

    from gpa_singlehead_repro.engine import (
        classifier_stats,
        estimate_prototypes,
        evaluate_seen,
        make_bias_for_mode,
        make_train_loader_with_exemplars,
        old_new_logit_diagnostics,
        save_json,
        task_sizes_until,
        train_one_phase,
        write_csv,
    )
    from gpa_singlehead_repro.gvalign_bridge import (
        build_gvalign_resnet32_backbone,
        load_gvalign_protocol,
        make_gvalign_exemplars,
    )
    from gpa_singlehead_repro.model import SingleHeadCILModel, normalize_rows

    os.makedirs(args.output_root, exist_ok=True)
    seed_everything(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    trn_loader, val_loader, tst_loader, taskcla = load_gvalign_protocol(
        args.gvalign_root,
        args.data_root,
        args.dataset,
        args.num_tasks,
        args.nc_first_task,
        args.batch_size,
        args.num_workers,
    )
    backbone, feature_dim = build_gvalign_resnet32_backbone(args.gvalign_root)
    model = SingleHeadCILModel(backbone, feature_dim).to(device)
    exemplars = make_gvalign_exemplars(
        val_loader[0].dataset.transform,
        trn_loader[0].dataset.class_indices,
        args.num_exemplars_per_class,
        args.exemplar_selection,
    )

    max_task = min(args.max_task, len(taskcla))
    anchors = {}
    acc_rows = []
    summaries = []

    for task_id in range(max_task):
        print("== Phase {} / method={} ==".format(task_id, args.method), flush=True)
        task_sizes = task_sizes_until(taskcla, task_id)
        current_classes = list(range(sum(task_sizes[:-1]), sum(task_sizes)))
        init_weight = None
        init_bias = None
        gpa_init = {}
        before_expand_eval = None
        before_expand_diag = None

        if task_id > 0:
            before_expand_eval = evaluate_seen(model, tst_loader, taskcla, task_id - 1, device, eval_mode=args.eval_mode)
            # The before-expand model has no new rows, so old/new diagnostics are not defined here.
            before_expand_diag = {"old_acc": before_expand_eval["weighted_seen_acc"]}

        # GPA uses a frozen copy from the previous phase for prototype estimation.
        prev_model = copy.deepcopy(model).to(device).eval() if task_id > 0 else None

        if args.method == "finetune_gpa" and task_id > 0:
            prototypes, counts = estimate_prototypes(prev_model, trn_loader[task_id], current_classes, device)
            init_weight = normalize_rows(prototypes)
            old_bias = model.classifier.bias.detach().clone() if model.classifier is not None else None
            init_bias, bias_diag = make_bias_for_mode(args.gpa_bias_mode, counts, args.gpa_nref, args.gpa_eps, old_bias=old_bias)
            for class_id, proto in zip(current_classes, prototypes.detach().cpu()):
                # Store raw EMA prototypes. The anchor loss normalizes them when comparing to W_c.
                anchors[class_id] = proto.clone()
            gpa_init = {
                "current_classes": current_classes,
                "counts": counts.detach().cpu().tolist(),
                "bias_mode": args.gpa_bias_mode,
                "bias_diag": bias_diag,
                "prototype_norm_mean": float(prototypes.norm(dim=1).mean().item()),
                "prototype_norm_min": float(prototypes.norm(dim=1).min().item()),
                "prototype_norm_max": float(prototypes.norm(dim=1).max().item()),
                "new_weight_cos_to_frozen_proto_mean": float(
                    torch.sum(init_weight * normalize_rows(prototypes), dim=1).mean().item()
                ),
                "new_weight_cos_to_frozen_proto_min": float(
                    torch.sum(init_weight * normalize_rows(prototypes), dim=1).min().item()
                ),
            }

        old_dim = model.expand(task_sizes, init_weight, init_bias)
        seen_dim = sum(task_sizes)
        after_init_eval = evaluate_seen(model, tst_loader, taskcla, task_id, device, eval_mode=args.eval_mode)
        after_init_diag = old_new_logit_diagnostics(model, tst_loader, taskcla, task_id, device, eval_mode=args.eval_mode)
        after_init_classifier = classifier_stats(model, old_dim=old_dim, seen_dim=seen_dim)

        phase_loader = make_train_loader_with_exemplars(trn_loader[task_id], exemplars)
        train_stats = train_one_phase(
            model=model,
            loader=phase_loader,
            taskcla=taskcla,
            task_id=task_id,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            lr_factor=args.lr_factor,
            schedule_step=args.schedule_step,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            clipgrad=args.clipgrad,
            method=args.method,
            lambda_gpa=args.lambda_gpa,
            anchors=anchors,
            anchor_momentum=args.gpa_anchor_momentum,
            freeze_old_rows=args.freeze_old_rows,
            restore_old_rows=args.restore_old_rows,
            anchor_reduction=args.gpa_anchor_reduction,
        )
        eval_stats = evaluate_seen(model, tst_loader, taskcla, task_id, device, eval_mode=args.eval_mode)
        after_train_diag = old_new_logit_diagnostics(model, tst_loader, taskcla, task_id, device, eval_mode=args.eval_mode)
        after_train_classifier = classifier_stats(model, old_dim=old_dim, seen_dim=seen_dim)

        # Keep GVAlign's exemplar protocol. Collection happens after the phase has been trained.
        exemplars.collect_exemplars(model, phase_loader, val_loader[task_id].dataset.transform)

        phase_summary = {
            "phase": task_id,
            "num_seen_classes": seen_dim,
            "old_dim": old_dim,
            "eval_mode": args.eval_mode,
            "before_expand_eval": before_expand_eval,
            "before_expand_diag": before_expand_diag,
            "gpa_init": gpa_init,
            "after_init_eval": after_init_eval,
            "after_init_old_new_diag": after_init_diag,
            "after_init_classifier_stats": after_init_classifier,
            "train": train_stats,
            "eval": eval_stats,
            "after_train_old_new_diag": after_train_diag,
            "after_train_classifier_stats": after_train_classifier,
            "num_exemplars_after_phase": len(exemplars),
        }
        summaries.append(phase_summary)
        for eval_task, acc in enumerate(eval_stats["acc_by_task"]):
            acc_rows.append({"phase": task_id, "eval_task": eval_task, "acc": acc})

        phase_dir = os.path.join(args.output_root, "phase{}".format(task_id))
        os.makedirs(phase_dir, exist_ok=True)
        save_json(os.path.join(phase_dir, "summary.json"), phase_summary)

    weighted_seen = [phase["eval"]["weighted_seen_acc"] for phase in summaries]
    final_eval_calibration = {}
    if args.eval_calibration and summaries:
        final_phase = len(summaries) - 1
        for mode in ["raw", "no_bias", "norm_weight_eval", "cosine_eval"]:
            final_eval_calibration[mode] = {
                "eval": evaluate_seen(model, tst_loader, taskcla, final_phase, device, eval_mode=mode),
                "old_new_diag": old_new_logit_diagnostics(model, tst_loader, taskcla, final_phase, device, eval_mode=mode),
            }

    final_summary = {
        "method": args.method,
        "dataset": args.dataset,
        "seed": args.seed,
        "num_tasks": args.num_tasks,
        "max_task": max_task,
        "nc_first_task": args.nc_first_task,
        "num_exemplars_per_class": args.num_exemplars_per_class,
        "exemplar_selection": args.exemplar_selection,
        "single_head": True,
        "gvalign_root": os.path.abspath(args.gvalign_root),
        "data_root": os.path.abspath(args.data_root),
        "class_order": trn_loader[0].dataset.class_indices,
        "taskcla": taskcla,
        "args": vars(args),
        "paper_acc_include_base": float(np.mean(weighted_seen)),
        "paper_acc_exclude_base": float(np.mean(weighted_seen[1:])) if len(weighted_seen) > 1 else None,
        "paper_accT": float(weighted_seen[-1]),
        "eval_calibration_final": final_eval_calibration,
        "phases": summaries,
    }
    save_json(os.path.join(args.output_root, "summary.json"), final_summary)
    write_csv(os.path.join(args.output_root, "acc_matrix.csv"), acc_rows, ["phase", "eval_task", "acc"])
    print(json.dumps({
        "paper_acc_include_base": final_summary["paper_acc_include_base"],
        "paper_acc_exclude_base": final_summary["paper_acc_exclude_base"],
        "paper_accT": final_summary["paper_accT"],
        "summary": os.path.join(args.output_root, "summary.json"),
    }, indent=2))


if __name__ == "__main__":
    run()
