"""Independent old-NC/current-dynamic asymmetric anchor assembly."""

from __future__ import annotations

import hashlib
from typing import Mapping, Sequence

import torch
from torch.nn import functional as F


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().contiguous().cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


@torch.no_grad()
def assemble_old_nc_current_dynamic_anchors(
    canonical_nc_anchors: torch.Tensor,
    original_dynamic_anchors: torch.Tensor,
    class_order: Sequence[int],
    current_class_ids: Sequence[int],
) -> tuple[torch.Tensor, dict[str, object]]:
    """Assemble one normalized anchor matrix in global checkpoint class order.

    Old rows always come from canonical NC coordinates. Current-session rows
    always come from the unmodified original Full Dynamic geometry.
    """

    if canonical_nc_anchors.ndim != 2 or original_dynamic_anchors.ndim != 2:
        raise ValueError("anchor inputs must be 2D [classes, dimension]")
    if canonical_nc_anchors.shape != original_dynamic_anchors.shape:
        raise ValueError("NC and dynamic anchor shapes must match")
    if canonical_nc_anchors.shape[0] != len(class_order):
        raise ValueError("class_order length must match anchor rows")
    if len(set(int(value) for value in class_order)) != len(class_order):
        raise ValueError("class_order must contain unique class ids")
    current = {int(value) for value in current_class_ids}
    if not current or not current.issubset({int(value) for value in class_order}):
        raise ValueError("current classes must be a non-empty subset of class_order")

    nc = canonical_nc_anchors.detach()
    dynamic = original_dynamic_anchors.detach()
    assembled = dynamic.clone()
    old_rows = []
    current_rows = []
    for row, class_id in enumerate(class_order):
        if int(class_id) in current:
            current_rows.append(row)
        else:
            assembled[row] = nc[row]
            old_rows.append(row)
    assembled = F.normalize(assembled, dim=1).detach()
    metadata: dict[str, object] = {
        "method": "old_nc_current_dynamic_v1",
        "class_order": [int(value) for value in class_order],
        "current_class_ids": sorted(current),
        "old_rows": old_rows,
        "current_rows": current_rows,
        "shape": list(assembled.shape),
        "dtype": str(assembled.dtype),
        "normalized": True,
        "requires_grad": bool(assembled.requires_grad),
        "semantic_role": "single asymmetric cosine-classifier anchor matrix",
        "old_source": "canonical fixed_nc_geometry rows",
        "current_source": "original compute_dynamic_structure rows",
        "old_anchor_sha256": tensor_sha256(assembled[old_rows]),
        "current_anchor_sha256": tensor_sha256(assembled[current_rows]),
        "assembled_sha256": tensor_sha256(assembled),
    }
    return assembled, metadata


def anchor_sources_by_class(
    class_order: Sequence[int],
    current_class_ids: Sequence[int],
) -> Mapping[int, str]:
    current = {int(value) for value in current_class_ids}
    return {
        int(class_id): ("original_dynamic" if int(class_id) in current else "canonical_nc")
        for class_id in class_order
    }
