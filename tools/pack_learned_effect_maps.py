from __future__ import annotations

"""Pack ComfyUI/RunningHub effect maps into the strict NPZ contract.

Image encoding:

* shadow: black=0, white=1;
* gain: middle gray=1.0, black=0.82, white=1.18;
* specular: middle gray=0, black=-0.08, white=+0.08.

``.npy`` inputs are treated as already-decoded float arrays.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strict_compositor.learned_effects import (  # noqa: E402
    GAIN_MAX,
    GAIN_MIN,
    SPECULAR_RESIDUAL_LIMIT,
)


def _read_image(path: Path, *, rgb: bool = False) -> tuple[np.ndarray, float]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cannot decode effect map: {path}")
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[..., :3]
    if rgb and image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif not rgb and image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype == np.uint8:
        scale = 255.0
    elif image.dtype == np.uint16:
        scale = 65535.0
    else:
        raise ValueError("effect map images must be 8-bit or 16-bit PNG/TIFF")
    return image.astype(np.float32), scale


def _load_array(path: Path, kind: str) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
        if not np.issubdtype(array.dtype, np.floating):
            raise ValueError(f"{kind} .npy must contain floating-point data")
        array = array.astype(np.float32, copy=False)
    else:
        image, scale = _read_image(path, rgb=kind == "specular_residual")
        normalized = image / scale
        if kind == "shadow_alpha":
            array = normalized
        elif kind == "illumination_gain":
            # Make the exact midpoint an exact neutral 1.0 for both 8/16 bit.
            midpoint = (round(scale / 2.0)) / scale
            below = normalized <= midpoint
            array = np.empty_like(normalized, dtype=np.float32)
            array[below] = 1.0 - (
                (midpoint - normalized[below]) / max(midpoint, 1e-6)
            ) * (1.0 - GAIN_MIN)
            array[~below] = 1.0 + (
                (normalized[~below] - midpoint) / max(1.0 - midpoint, 1e-6)
            ) * (GAIN_MAX - 1.0)
        else:
            midpoint = (round(scale / 2.0)) / scale
            below = normalized <= midpoint
            array = np.empty_like(normalized, dtype=np.float32)
            array[below] = -(
                (midpoint - normalized[below]) / max(midpoint, 1e-6)
            ) * SPECULAR_RESIDUAL_LIMIT
            array[~below] = (
                (normalized[~below] - midpoint) / max(1.0 - midpoint, 1e-6)
            ) * SPECULAR_RESIDUAL_LIMIT
    if not np.isfinite(array).all():
        raise ValueError(f"{kind} contains NaN or infinity")
    if kind in {"shadow_alpha", "illumination_gain"} and array.ndim != 2:
        raise ValueError(f"{kind} must be a single-channel HxW map")
    if kind == "specular_residual" and array.ndim not in {2, 3}:
        raise ValueError("specular_residual must be HxW or HxWx3")
    if kind == "specular_residual" and array.ndim == 3 and array.shape[2] != 3:
        raise ValueError("specular_residual RGB map must have exactly three channels")
    limits = {
        "shadow_alpha": (0.0, 1.0),
        "illumination_gain": (GAIN_MIN, GAIN_MAX),
        "specular_residual": (
            -SPECULAR_RESIDUAL_LIMIT,
            SPECULAR_RESIDUAL_LIMIT,
        ),
    }
    low, high = limits[kind]
    if float(array.min()) < low - 1e-6 or float(array.max()) > high + 1e-6:
        raise ValueError(f"{kind} exceeds [{low},{high}]")
    return array.astype(np.float32, copy=False)


def _sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pack restricted learned shadow/relight maps into NPZ",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--shadow")
    parser.add_argument("--gain")
    parser.add_argument("--specular")
    args = parser.parse_args()
    inputs = {
        "shadow_alpha": args.shadow,
        "illumination_gain": args.gain,
        "specular_residual": args.specular,
    }
    if not any(inputs.values()):
        parser.error("at least one of --shadow, --gain or --specular is required")
    arrays = {
        name: _load_array(Path(path).resolve(), name)
        for name, path in inputs.items()
        if path
    }
    shapes = {value.shape[:2] for value in arrays.values()}
    if len(shapes) != 1:
        raise ValueError(f"all maps must share one canvas size, got {sorted(shapes)}")
    output = Path(args.output).resolve()
    if output.suffix.lower() != ".npz":
        raise ValueError("--output must end in .npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    print(json.dumps({
        "output": str(output),
        "canvas_shape": list(next(iter(shapes))),
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "range": [float(value.min()), float(value.max())],
                "sha256": _sha256(value),
            }
            for name, value in arrays.items()
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
