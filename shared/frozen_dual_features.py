"""Pure feature assembly for the GSR-VILA frozen dual representation.

The module deliberately does not import a model package.  Callers provide the
already locked APART logits and CLIP image features, which keeps feature
extraction auditable and makes accidental label dependent forward passes
impossible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np


def _finite_matrix(x: Any, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError(f"BLOCKED_{name.upper()}_SHAPE_OR_FINITE")
    return x


def normalize_rows(x: Any, *, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize rows without silently accepting zero vectors."""
    x = _finite_matrix(x, "feature")
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
    h = np.concatenate((a, u), axis=1) / np.sqrt(2.0)
    if h.shape[1] != a.shape[1] + u.shape[1] or not np.isfinite(h).all():
        raise ValueError("BLOCKED_JOINT_FEATURE")
    return FeatureBatch(a=a, u=u, h=h)


def label_free_forward(
    images: Any,
    apart_forward: Callable[[Any], Any],
    clip_forward: Callable[[Any], Any],
) -> FeatureBatch:
    """Run both frozen encoders without passing labels to either forward."""
    return build_joint_feature(apart_forward(images), clip_forward(images))


class FrozenDualFeatureExtractor:
    """Small adapter around locked, externally constructed encoders.

    ``clip_model`` may be an OpenCLIP model or any object exposing
    ``encode_image``.  Dependency loading is intentionally left to the caller
    so tests and protocol qualification never download weights implicitly.
    """

    def __init__(self, apart_model: Callable[[Any], Any], clip_model: Any):
        if not callable(apart_model) or not callable(getattr(clip_model, "encode_image", None)):
            raise TypeError("BLOCKED_ENCODER_INTERFACE")
        self.apart_model = apart_model
        self.clip_model = clip_model

    def __call__(self, images: Any) -> FeatureBatch:
        return label_free_forward(images, self.apart_model, self.clip_model.encode_image)


def validate_encoder_lock(lock: Mapping[str, Any]) -> None:
    """Require provenance fields before a real extraction is allowed."""
    required = ("model_name", "pretrained", "weights_sha256", "preprocess_sha256")
    missing = [key for key in required if not lock.get(key) or str(lock.get(key)).startswith("REQUIRED_")]
    if missing:
        raise ValueError("BLOCKED_ENCODER_LOCK:" + ",".join(missing))
