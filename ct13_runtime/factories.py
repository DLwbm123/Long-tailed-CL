"""Minimal, read-only APART and OpenCLIP adapters for CT13."""
from __future__ import annotations

import hashlib
import re
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
    device = _device()
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained=None, device=device)
    state = load_file(str(weight_path), device="cpu")
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError("BLOCKED_CLIP_STATE_DICT")
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-16")

    @torch.no_grad()
    def forward(batch: Any) -> torch.Tensor:
        return model.encode_image(batch.to(device, non_blocking=True)).float().cpu()

    @torch.no_grad()
    def encode_text(tokens: Any) -> torch.Tensor:
        return model.encode_text(tokens.to(device, non_blocking=True)).float().cpu()

    return {"preprocess": preprocess, "forward": forward,
            "tokenizer": tokenizer, "encode_text": encode_text}


def build_apart(*, checkpoint: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    """Restore one normal Task1 parent and expose label-free APART features."""
    import torch
    apart_root = Path("/root/rivermind-data/LongTailedCL/v2/q8m7/code/third_party/APART")
    if apart_root.is_dir() and str(apart_root) not in sys.path:
        sys.path.insert(0, str(apart_root))
    from utils.inc_net import AdapterVitNet
    from utils.data import build_transform
    from torchvision import transforms

    checkpoint = Path(checkpoint).expanduser()
    expected = None
    match = re.search(r"ISIC_(\d+)_", checkpoint.name)
    if match and isinstance(lock, Mapping):
        item = lock.get(match.group(1), {})
        expected = item.get("sha256") if isinstance(item, Mapping) else item
    if expected and _sha256(checkpoint) != str(expected).lower():
        raise ValueError("BLOCKED_F1_PARENT_SHA256")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = dict(payload["args"])
    args["device"] = [torch.device("cpu")]
    model = AdapterVitNet(args, True)
    delta = payload["delta"]
    embedding = delta.get("backbone.assigner.cls_emb.weight")
    if embedding is not None:
        from torch import nn
        model.backbone.assigner.cls_emb = nn.Embedding.from_pretrained(
            embedding.detach().clone(), freeze=False)
    state = model.load_state_dict(delta, strict=False)
    if state.unexpected_keys or len(state.missing_keys) != 348:
        raise ValueError("BLOCKED_F1_PARENT_DELTA")
    device = _device()
    model.to(device).eval()
    preprocess = transforms.Compose(build_transform(False, args))

    @torch.no_grad()
    def forward(batch: Any) -> Any:
        raw = model(batch.to(device, non_blocking=True), train=False)
        return {key: value.float().cpu() for key, value in raw.items()
                if isinstance(value, torch.Tensor)}

    return {"preprocess": preprocess, "forward": forward}
