from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PixelAudit:
    changed_pixels: int
    max_delta: int
    verdict: str

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def audit_background(background: np.ndarray, result: np.ndarray, authorized_mask: np.ndarray) -> PixelAudit:
    if background.shape != result.shape:
        return PixelAudit(-1, -1, "reject")
    outside = ~authorized_mask.astype(bool)
    delta = np.abs(result.astype(np.int16) - background.astype(np.int16)).max(axis=2)
    values = delta[outside]
    changed = int(np.count_nonzero(values))
    maximum = int(values.max()) if values.size else 0
    return PixelAudit(changed, maximum, "pass" if changed == 0 and maximum == 0 else "reject")
