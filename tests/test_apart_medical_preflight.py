"""Run original APART components on CPU without loading images or pretrained weights."""
import ast
import csv
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn, optim


def original_component(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def run():
    root = Path(__file__).resolve().parents[1]
    backbone = root / "third_party/APART/backbone/vision_transformer_adapter_pool_a.py"
    pool = original_component(backbone, "PoolAssigner", {"torch": torch, "nn": nn})()
    split = root / "src/datasets/fopro_splits/ham2019/train_skin1_0.01.csv"
    with split.open(newline="") as handle:
        counts = Counter(int(r["finding"]) for r in csv.DictReader(handle))
    observations = []
    for label, count in sorted(counts.items()):
        try:
            output = pool(torch.zeros(1, 768), torch.tensor([count]))
            assert torch.isfinite(output).all()
            error = None
        except IndexError as exc:
            error = str(exc)
        observations.append({"original_label": label, "train_count": count, "error": error})
        assert bool(error) == (count >= pool.cls_emb.num_embeddings)
    assert sum(row["error"] is not None for row in observations) == 5

    # Execute the two original optimizer methods without instantiating the pretrained ViT.
    tree = ast.parse((root / "third_party/APART/models/apart.py").read_text())
    learner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Learner")
    methods = [n for n in learner.body if isinstance(n, ast.FunctionDef) and n.name in {"get_optimizer", "get_scheduler"}]
    namespace = {"optim": optim}
    exec(compile(ast.Module(body=methods, type_ignores=[]), "original_apart_optimizer_methods", "exec"), namespace)
    model = nn.Module()
    model.pool = nn.Linear(2, 2)
    model.head = nn.Linear(2, 2)
    state = SimpleNamespace(_network=SimpleNamespace(backbone=model), init_lr=0.003, weight_decay=0, min_lr=1e-5,
                            args={"optimizer": "adamw", "scheduler": "constant", "init_cls": 4, "increment": 2, "tuned_epoch": 10})
    first = namespace["get_optimizer"](state)
    scheduler = namespace["get_scheduler"](state, first)
    active = namespace["get_optimizer"](state)
    groups = [{"lr": g["lr"], "weight_decay": g["weight_decay"], "legacy_weight_decay_key": g["weight decay"], "betas": g["betas"]} for g in active.param_groups]
    assert all(g["weight_decay"] == 0.01 for g in active.param_groups)
    assert scheduler.optimizer is first and scheduler.optimizer is not active
    return {
        "status": "BLOCKED_TRAINING_SEMANTICS",
        "test": "original PoolAssigner on all actual split1 class counts",
        "device": "cpu", "torch": torch.__version__,
        "embedding_rows": pool.cls_emb.num_embeddings, "valid_indices": [0, pool.cls_emb.num_embeddings-1],
        "class_results": observations,
        "optimizer_audit": {"mode": "legacy_effective", "groups": groups, "scheduler": type(scheduler).__name__,
                            "scheduler_arg_overwritten_to": state.args["scheduler"], "S0_binding_correct": True,
                            "incremental_reinit_binding_correct": scheduler.optimizer is active},
        "full_pretrained_label_blind_test": "NOT_RUN", "train_batches": 0, "test_predictions": 0,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
