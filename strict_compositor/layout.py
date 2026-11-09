from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .errors import ContractError
from .render import placement_footprint


@dataclass(frozen=True)
class Placement:
    product_id: str
    center_x: float
    bottom_y: float
    height_ratio: float
    z_order: int


def placement_homography(source_size: tuple[int, int], canvas_size: tuple[int, int], placement: Placement) -> np.ndarray:
    """Deterministic scale/translate placement for front-facing source crops.

    Perspective transforms may be supplied directly in a job.  This helper is
    used by an upstream solver only after it has verified the support plane.
    """
    source_w, source_h = source_size
    canvas_w, canvas_h = canvas_size
    if not 0 < placement.height_ratio <= 1:
        raise ContractError("height_ratio must be in (0,1]")
    target_h = canvas_h * placement.height_ratio
    scale = target_h / source_h
    x = canvas_w * placement.center_x - source_w * scale / 2
    y = canvas_h * placement.bottom_y - source_h * scale
    return np.array([[scale, 0, x], [0, scale, y], [0, 0, 1]], dtype=np.float64)


def overlap_ratio(alpha_a: np.ndarray, alpha_b: np.ndarray) -> float:
    a, b = alpha_a > 4, alpha_b > 4
    overlap = np.count_nonzero(a & b)
    denominator = max(1, min(np.count_nonzero(a), np.count_nonzero(b)))
    return overlap / denominator


def validate_placement_masks(
    product_masks: list[np.ndarray],
    support: np.ndarray,
    protected: np.ndarray,
    max_overlap: float = 0.005,
    placement_modes: list[str] | None = None,
) -> None:
    support_tolerant = cv2.dilate(support.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    modes = placement_modes or ["upright"] * len(product_masks)
    for mask, mode in zip(product_masks, modes):
        if np.any((mask > 4) & protected):
            raise ContractError("Placement covers a protected region")
        footprint = placement_footprint(mask, mode) > 4
        if not np.any(footprint) or np.any(footprint & ~support_tolerant):
            raise ContractError("Placement contact footprint lies outside support area")
    for index, first in enumerate(product_masks):
        for second in product_masks[index + 1:]:
            if overlap_ratio(first, second) > max_overlap:
                raise ContractError("Product body overlap exceeds permitted threshold")
