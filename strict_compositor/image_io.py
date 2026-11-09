from __future__ import annotations

from pathlib import Path
from typing import Final

import cv2
import numpy as np
from PIL import Image, ImageOps

from .errors import ContractError

LOSSLESS_SUFFIXES: Final = {".png", ".tif", ".tiff"}


def load_background(path: str | Path) -> np.ndarray:
    """Return orientation-corrected uint8 RGB, without colour correction."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        return np.asarray(image).copy()


def load_rgba(path: str | Path) -> np.ndarray:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGBA")
        rgba = np.asarray(image).copy()
    if rgba.ndim != 3 or rgba.shape[2] != 4 or not np.any(rgba[..., 3]):
        raise ContractError(f"Product must be a non-empty RGBA image: {path}")
    return rgba


def load_mask(
    path: str | Path,
    size: tuple[int, int],
    *,
    allow_resize: bool = False,
) -> np.ndarray:
    with Image.open(path) as source:
        mask = ImageOps.exif_transpose(source).convert("L")
        if mask.size != size:
            if not allow_resize:
                raise ContractError(
                    f"Mask size {mask.size} does not match background {size}: {path}"
                )
            mask = mask.resize(size, Image.Resampling.NEAREST)
        return np.asarray(mask) > 0


def save_png(path: str | Path, image: np.ndarray) -> None:
    target = Path(path)
    if target.suffix.lower() not in LOSSLESS_SUFFIXES:
        raise ContractError("Strict output must be PNG or TIFF, never JPEG")
    target.parent.mkdir(parents=True, exist_ok=True)
    if image.dtype != np.uint8:
        raise ContractError("Only uint8 image output is supported")
    if image.ndim == 2:
        encoded = image
    elif image.ndim == 3 and image.shape[2] == 3:
        encoded = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        encoded = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    else:
        raise ContractError(f"Unsupported image shape: {image.shape}")
    # OpenCV's filename-based writer can fail on Unicode Windows paths.
    # Encode in memory, then let pathlib perform the filesystem operation.
    success, buffer = cv2.imencode(target.suffix.lower(), encoded)
    if not success:
        raise OSError(f"Could not write {target}")
    target.write_bytes(buffer.tobytes())
