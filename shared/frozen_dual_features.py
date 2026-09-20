"""Pure feature assembly for the GSR-VILA frozen dual representation.

The module deliberately does not import a model package.  Callers provide the
already locked APART logits and CLIP image features, which keeps feature
extraction auditable and makes accidental label dependent forward passes
impossible.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


def _finite_matrix(x: Any, name: str) -> Any:
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(x, torch.Tensor):
        if x.ndim != 2 or not torch.isfinite(x).all():
            raise ValueError(f"BLOCKED_{name.upper()}_SHAPE_OR_FINITE")
        return x
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError(f"BLOCKED_{name.upper()}_SHAPE_OR_FINITE")
    return x


def normalize_rows(x: Any, *, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize rows without silently accepting zero vectors."""
    x = _finite_matrix(x, "feature")
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(x, torch.Tensor):
        norm = torch.linalg.vector_norm(x, dim=1, keepdim=True)
        if bool((norm <= eps).any()):
            raise ValueError("BLOCKED_ZERO_FEATURE_NORM")
        return x / norm
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norm <= eps):
        raise ValueError("BLOCKED_ZERO_FEATURE_NORM")
    return x / norm


@dataclass(frozen=True)
class FeatureBatch:
    """A locked, label-free feature batch."""

    a: np.ndarray
    u: np.ndarray
    h: np.ndarray

    @property
    def n(self) -> int:
        return int(self.h.shape[0])


def build_joint_feature(main_few_logits: Any, clip_visual: Any) -> FeatureBatch:
    """Build ``a``, ``u`` and ``h=[a;u]/sqrt(2)`` exactly once per image."""
    a = normalize_rows(main_few_logits)
    u = normalize_rows(clip_visual)
    if a.shape[0] != u.shape[0]:
        raise ValueError("BLOCKED_DUAL_BATCH_MISMATCH")
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(a, torch.Tensor):
        h = torch.cat((a, u), dim=1) / np.sqrt(2.0)
        finite = bool(torch.isfinite(h).all())
    else:
        h = np.concatenate((a, u), axis=1) / np.sqrt(2.0)
        finite = bool(np.isfinite(h).all())
    if h.shape[1] != a.shape[1] + u.shape[1] or not finite:
        raise ValueError("BLOCKED_JOINT_FEATURE")
    return FeatureBatch(a=a, u=u, h=h)


def _concat_pair(main: Any, few: Any) -> Any:
    """Concatenate the two APART branches before the single L2 normalization."""
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(main, torch.Tensor):
        if not isinstance(few, torch.Tensor) or main.ndim != 2 or few.ndim != 2:
            raise ValueError("BLOCKED_APART_PRELOGITS_TYPE")
        if main.shape[0] != few.shape[0] or main.shape[1] + few.shape[1] != 1536:
            raise ValueError("BLOCKED_APART_PRELOGITS_DIM")
        return torch.cat((main, few), dim=1)
    main, few = np.asarray(main), np.asarray(few)
    if main.ndim != 2 or few.ndim != 2 or main.shape[0] != few.shape[0] or main.shape[1] + few.shape[1] != 1536:
        raise ValueError("BLOCKED_APART_PRELOGITS_DIM")
    return np.concatenate((main, few), axis=1)


def _unwrap_pre_logits(value: Any, *, apart: bool = False) -> Any:
    if isinstance(value, Mapping):
        if apart:
            if "pre_logits" in value and "pre_logits_few" in value:
                return _concat_pair(value["pre_logits"], value["pre_logits_few"])
            if "main_few_pre_logits" in value:
                joined = value["main_few_pre_logits"]
                if getattr(joined, "ndim", None) != 2 or joined.shape[1] != 1536:
                    raise ValueError("BLOCKED_APART_PRELOGITS_DIM")
                return joined
            raise ValueError("BLOCKED_APART_PRELOGITS_CONTRACT")
        for key in ("pre_logits", "features", "image_features"):
            if key in value:
                return value[key]
        raise ValueError("BLOCKED_ENCODER_OUTPUT_KEY")
    if apart and getattr(value, "shape", (None, None))[1] != 1536:
        raise ValueError("BLOCKED_APART_PRELOGITS_DIM")
    return value


def label_free_forward(
    images: Any,
    apart_forward: Callable[[Any], Any],
    clip_forward: Callable[[Any], Any],
) -> FeatureBatch:
    """Run both frozen encoders without passing labels to either forward."""
    return build_joint_feature(_unwrap_pre_logits(apart_forward(images), apart=True),
                               _unwrap_pre_logits(clip_forward(images)))


class FrozenDualFeatureExtractor:
    """Small adapter around locked, externally constructed encoders.

    ``clip_model`` may be an OpenCLIP model or any object exposing
    ``encode_image``.  Dependency loading is intentionally left to the caller
    so tests and protocol qualification never download weights implicitly.
    """

    def __init__(self, apart_model: Callable[[Any], Any], clip_model: Any,
                 *, apart_dim: int = 1536, clip_dim: int = 512):
        if not callable(apart_model) or not callable(getattr(clip_model, "encode_image", None)):
            raise TypeError("BLOCKED_ENCODER_INTERFACE")
        self.apart_model = apart_model
        self.clip_model = clip_model
        self.apart_dim, self.clip_dim = int(apart_dim), int(clip_dim)

    def __call__(self, images: Any) -> FeatureBatch:
        batch = label_free_forward(images, self.apart_model, self.clip_model.encode_image)
        if batch.a.shape[1] != self.apart_dim or batch.u.shape[1] != self.clip_dim:
            raise ValueError("BLOCKED_ENCODER_DIM")
        return batch


def validate_encoder_lock(lock: Mapping[str, Any]) -> None:
    """Require provenance fields before a real extraction is allowed."""
    required = ("model_name", "pretrained", "weights_path", "preprocess_path", "tokenizer_path",
                "weights_sha256", "preprocess_sha256", "tokenizer_sha256")
    missing = [key for key in required if not lock.get(key) or str(lock.get(key)).startswith("REQUIRED_")]
    if missing:
        raise ValueError("BLOCKED_ENCODER_LOCK:" + ",".join(missing))
    if lock["model_name"] != "ViT-B-16" or lock["pretrained"] != "laion400m_e32":
        raise ValueError("BLOCKED_ENCODER_IDENTITY")
    import re
    for key in ("weights_sha256", "preprocess_sha256", "tokenizer_sha256"):
        if not re.fullmatch(r"[0-9a-fA-F]{64}", str(lock[key])):
            raise ValueError("BLOCKED_ENCODER_DIGEST:" + key)
    import hashlib
    for path_key, digest_key in (("weights_path", "weights_sha256"),
                                 ("preprocess_path", "preprocess_sha256"),
                                 ("tokenizer_path", "tokenizer_sha256")):
        path = Path(str(lock[path_key])).expanduser()
        if not path.is_file():
            raise ValueError("BLOCKED_ENCODER_FILE:" + path_key)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest.lower() != str(lock[digest_key]).lower():
            raise ValueError("BLOCKED_ENCODER_SHA256:" + path_key)
