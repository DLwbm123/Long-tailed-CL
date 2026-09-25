"""Minimal, read-only APART and OpenCLIP adapters for CT13."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_digest(items: Any) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(items):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_clip(*, lock: Mapping[str, Any]) -> dict[str, Any]:
    """Load the pinned OpenCLIP file, without any network fallback."""
    import open_clip
    from safetensors.torch import load_file

    weight_path = Path(str(lock["weights_path"])).expanduser()
    if _sha256(weight_path) != str(lock["weights_sha256"]).lower():
        raise ValueError("BLOCKED_CLIP_WEIGHT_SHA256")
    for key, digest_key in (("preprocess_path", "preprocess_sha256"),
                            ("tokenizer_path", "tokenizer_sha256")):
        path = Path(str(lock[key])).expanduser()
        if _sha256(path) != str(lock[digest_key]).lower():
            raise ValueError("BLOCKED_CLIP_" + key.upper() + "_SHA256")
    source_cfg = json.loads(Path(str(lock["preprocess_path"])).read_text())
    model_cfg = source_cfg["model_cfg"]
    if (model_cfg["embed_dim"] != 512 or model_cfg["vision_cfg"] !=
            {"image_size": 224, "layers": 12, "width": 768, "patch_size": 16} or
            model_cfg["text_cfg"] != {"context_length": 77, "vocab_size": 49408,
                                      "width": 512, "heads": 8, "layers": 12}):
        raise ValueError("BLOCKED_CLIP_MODEL_CONFIG")
    preprocess_cfg = source_cfg["preprocess_cfg"]
    device = _device()
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained=None, device=device,
        image_mean=tuple(preprocess_cfg["mean"]), image_std=tuple(preprocess_cfg["std"]),
        image_interpolation=preprocess_cfg["interpolation"],
        image_resize_mode=preprocess_cfg["resize_mode"])
    state = load_file(str(weight_path), device="cpu")
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError("BLOCKED_CLIP_STATE_DICT")
    model.eval()
    model.requires_grad_(False)
    initial_hash = _tensor_digest(model.state_dict().items())
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    official_model = json.loads(Path(str(lock["tokenizer_path"])).read_text())["model"]
    official_vocab = official_model["vocab"]
    special_aliases = {"<|startoftext|>": "<start_of_text>",
                       "<|endoftext|>": "<end_of_text>"}
    if len(official_vocab) != len(tokenizer.encoder) or any(
        tokenizer.encoder.get(special_aliases.get(k, k)) != v
        for k, v in official_vocab.items()
    ):
        raise ValueError("BLOCKED_CLIP_TOKENIZER_VOCAB")
    official_merges = [tuple(item.split(" ") if isinstance(item, str) else item)
                       for item in official_model["merges"]]
    if len(official_merges) != len(tokenizer.bpe_ranks) or any(
        tokenizer.bpe_ranks.get(merge) != i for i, merge in enumerate(official_merges)
    ):
        raise ValueError("BLOCKED_CLIP_TOKENIZER_MERGES")

    @torch.no_grad()
    def forward(batch: Any) -> torch.Tensor:
        return model.encode_image(batch.to(device, non_blocking=True)).float().cpu()

    @torch.no_grad()
    def encode_text(tokens: Any) -> torch.Tensor:
        return model.encode_text(tokens.to(device, non_blocking=True)).float().cpu()

    return {"preprocess": preprocess, "forward": forward,
            "tokenizer": tokenizer, "encode_text": encode_text,
            "state_hash": lambda: _tensor_digest(model.state_dict().items()),
            "initial_state_hash": initial_hash}


def build_apart(*, checkpoint: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    """Restore one normal Task1 parent and expose label-free APART features."""
    apart_root = Path(__file__).resolve().parents[1] / "third_party/APART"
    if str(apart_root) not in sys.path:
        sys.path.insert(0, str(apart_root))
    from utils.inc_net import AdapterVitNet
    from utils.medical_v2 import extend_embedding, WEIGHT_SHA
    from torchvision import transforms

    checkpoint = Path(checkpoint).expanduser()
    expected = str(lock["parent_sha256"]).lower()
    if _sha256(checkpoint) != expected:
        raise ValueError("BLOCKED_F1_PARENT_SHA256")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["dataset"] != lock["dataset"] or payload["seed"] != int(lock["seed"]) or payload["stream"] != "U":
        raise ValueError("BLOCKED_F1_PARENT_IDENTITY")
    if payload["task"] != 0 or payload["known"] != 0 or payload["seen"] != 2 or payload["order"] != list(lock["order"]):
        raise ValueError("BLOCKED_F1_PARENT_TASK1")
    if payload["manifest_sha256"] != lock["manifest_sha256"] or payload["weight_sha256"] != WEIGHT_SHA:
        raise ValueError("BLOCKED_F1_PARENT_SOURCE")
    source_root = Path(str(lock["source_root"]))
    if {k: _sha256(source_root / k) for k in payload["code_sha256"]} != payload["code_sha256"]:
        raise ValueError("BLOCKED_F1_PARENT_CODE")
    args = dict(payload["args"])
    if args["locked_weight_path"] != lock["original_weight_path"]:
        raise ValueError("BLOCKED_F1_PARENT_ARGS")
    args["locked_weight_path"] = str(lock["weight_path"])
    if _sha256(Path(args["locked_weight_path"])) != WEIGHT_SHA:
        raise ValueError("BLOCKED_F1_CORE_SHA256")
    args["device"] = [torch.device("cpu")]
    model = AdapterVitNet(args, True)
    extend_embedding(model.backbone.assigner, int(lock["seed"]))
    nonshared = {"backbone." + k for k in model.backbone.weight_load_audit["allowed_missing"]}
    if sorted(nonshared) != payload["nonshared_keys"] or set(payload["delta"]) != nonshared:
        raise ValueError("BLOCKED_F1_PARENT_DELTA")
    shared = _tensor_digest((k, v) for k, v in model.state_dict().items() if k not in nonshared)
    if shared != payload["shared_sha256"]:
        raise ValueError("BLOCKED_F1_PARENT_SHARED")
    state = model.state_dict()
    state.update(payload["delta"])
    model.load_state_dict(state, strict=True)
    if _tensor_digest(model.state_dict().items()) != payload["network_sha256"]:
        raise ValueError("BLOCKED_F1_PARENT_NETWORK")
    # CT1's approved probe keeps native B=1 GEMM behavior for rank-3 Linear inputs.
    for module in model.modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        native = module.forward
        def forward_linear(x: torch.Tensor, *, module: torch.nn.Linear = module,
                           native: Any = native) -> torch.Tensor:
            if x.ndim != 3 or x.shape[0] == 1:
                return native(x)
            if x.dtype != module.weight.dtype or x.dtype != torch.float32:
                raise ValueError("BLOCKED_PROBE_LINEAR_DTYPE")
            return torch.cat([native(x[i:i + 1]) for i in range(len(x))], dim=0)
        module.forward = forward_linear
    model.backbone.pool.batchwise_prompt = False
    model.backbone.pool_few.batchwise_prompt = False
    model.requires_grad_(False)
    device = _device()
    model.to(device).eval()
    captured: dict[str, torch.Tensor] = {}
    model.original_backbone.register_forward_hook(
        lambda _module, _args, output: captured.update(p=output["pre_logits"])
    )
    preprocess = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((.5,) * 3, (.5,) * 3),
    ])

    @torch.no_grad()
    def forward(batch: Any) -> Any:
        captured.clear()
        raw = model(batch.to(device, non_blocking=True), train=False)
        if "p" not in captured:
            raise ValueError("BLOCKED_FROZEN_ORIGINAL_FEATURE")
        out = {key: value.float().cpu() for key, value in raw.items()
               if isinstance(value, torch.Tensor)}
        out["p"] = captured["p"].float().cpu()
        return out

    return {"preprocess": preprocess, "forward": forward,
            "state_hash": lambda: _tensor_digest(model.state_dict().items()),
            "restore": {"parent_sha256": expected, "shared_sha256": shared,
                        "network_sha256": payload["network_sha256"], "strict": True}}
