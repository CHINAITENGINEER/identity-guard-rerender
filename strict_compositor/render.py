"""Deterministic render facade with a recovered Python 3.13 runtime.

The source file was truncated by an interrupted external process while the
already-compiled implementation remained intact.  The frozen runtime keeps the
complete, last-known-good rendering implementation available; this facade
re-exports it and contains the small, reviewable contact-profile correction
that was being developed when the interruption happened.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path
import sys


_RUNTIME_PATH = Path(__file__).with_name("_render_runtime.cpython-313.pyc")
if sys.version_info[:2] != (3, 13):
    raise RuntimeError(
        "The recovered deterministic renderer requires CPython 3.13; "
        f"running {sys.version_info.major}.{sys.version_info.minor}."
    )
if not _RUNTIME_PATH.is_file():
    raise RuntimeError(f"Recovered renderer runtime is missing: {_RUNTIME_PATH}")

_RUNTIME_NAME = f"{__package__}._render_runtime"
_loader = importlib.machinery.SourcelessFileLoader(
    _RUNTIME_NAME,
    str(_RUNTIME_PATH),
)
_spec = importlib.util.spec_from_loader(_RUNTIME_NAME, _loader)
if _spec is None:
    raise RuntimeError(f"Cannot create renderer runtime spec for {_RUNTIME_PATH}")
_runtime = importlib.util.module_from_spec(_spec)
sys.modules[_RUNTIME_NAME] = _runtime
_loader.exec_module(_runtime)

# Preserve the public and package-private API expected by layout.py,
# pipeline.py and the existing tests.  Function objects keep the runtime
# module as their globals, so the corrected function is installed there too.
for _name, _value in vars(_runtime).items():
    if _name not in {
        "__name__",
        "__package__",
        "__loader__",
        "__spec__",
        "__file__",
        "__cached__",
        "__builtins__",
    }:
        globals()[_name] = _value


def contact_profile(alpha: np.ndarray) -> np.ndarray:
    """Return only the silhouette columns that can physically touch a plane.

    The lowest coherent source band determines the real contact columns.  A
    one-row morphological close may fill small *internal* gaps in that band,
    but it must never grow past the raw first or last contact column.  Bottom
    heights are measured only from raw contact columns; filled gap columns are
    interpolated from their real neighbours.  This prevents a high side edge
    (for example the two sides of a house clock) from being promoted to a foot
    after preview-scale morphology.
    """

    binary = alpha >= 32
    profile = np.zeros_like(alpha, np.uint8)
    occupied = np.argwhere(binary)
    if occupied.size == 0:
        return profile

    y0, x0 = occupied.min(axis=0)
    y1, x1 = occupied.max(axis=0)
    object_h = max(1, int(y1 - y0 + 1))
    band_h = max(3, min(8, int(round(object_h * 0.01))))
    band_top = max(int(y0), int(y1) - band_h + 1)
    band = binary[band_top : int(y1) + 1, int(x0) : int(x1) + 1]

    raw_columns = band.any(axis=0).astype(np.uint8)
    raw_xs_local = np.flatnonzero(raw_columns > 0)
    if not len(raw_xs_local):
        return profile

    columns = raw_columns[None, :] * 255
    close_width = max(3, int(round(object_h * 0.012)) | 1)
    columns = cv2.morphologyEx(
        columns,
        cv2.MORPH_CLOSE,
        np.ones((1, close_width), np.uint8),
    )
    raw_first = int(raw_xs_local[0])
    raw_last = int(raw_xs_local[-1])
    columns[0, :raw_first] = 0
    columns[0, raw_last + 1 :] = 0
    xs_local = np.flatnonzero(columns[0] > 0)
    if not len(xs_local):
        return profile

    raw_xs = raw_xs_local + int(x0)
    xs = xs_local + int(x0)
    raw_bottoms = np.empty(len(raw_xs), np.float32)
    for index, x in enumerate(raw_xs):
        occupied_y = np.flatnonzero(binary[:, int(x)])
        if not len(occupied_y):
            raw_bottoms[index] = float(y1)
        else:
            raw_bottoms[index] = float(occupied_y[-1])

    bottoms = np.interp(xs, raw_xs, raw_bottoms)
    for x, bottom in zip(xs, np.rint(bottoms).astype(np.int32)):
        top = max(0, int(bottom) - 1)
        profile[top : min(alpha.shape[0], int(bottom) + 2), int(x)] = 255
    return profile


_runtime.contact_profile = contact_profile
globals()["contact_profile"] = contact_profile
