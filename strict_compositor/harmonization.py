from __future__ import annotations

import cv2
import numpy as np


_LUMA = np.asarray([0.2126, 0.7152, 0.0722], np.float32)


def _alpha_normalized_lowpass(
    rgb: np.ndarray,
    alpha: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Low-pass foreground colour without mixing transparent matte RGB."""
    weight = cv2.GaussianBlur(
        alpha,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REPLICATE,
    )
    numerator = np.stack(
        [
            cv2.GaussianBlur(
                rgb[..., channel] * alpha,
                (0, 0),
                sigmaX=sigma,
                sigmaY=sigma,
                borderType=cv2.BORDER_REPLICATE,
            )
            for channel in range(3)
        ],
        axis=-1,
    )
    return numerator / np.maximum(weight[..., None], 1.0 / 255.0)


def low_frequency_harmonize(
    rgba: np.ndarray,
    rgb_gain: tuple[float, float, float] = (1.0, 1.0, 1.0),
    *,
    chroma_retention: float = 0.90,
    shadow_direction: tuple[float, float] | None = None,
    relight_strength: float = 0.05,
) -> np.ndarray:
    """Adapt only the low-frequency foreground illumination.

    The correction is estimated from an alpha-normalized low-pass image and
    added back to the original RGB.  Consequently text, patterns and texture
    remain the source high-frequency residual; transparent matte colour never
    enters the estimate.  Alpha and fully transparent RGB are bit-exact.

    This is a deterministic photographic correction, not a learned relight.
    """
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("RGBA image must have four channels")
    gain = np.asarray(rgb_gain, np.float32)
    if gain.shape != (3,) or not np.isfinite(gain).all():
        raise ValueError("rgb_gain must contain three finite values")
    gain = np.clip(gain, 0.78, 1.10)
    chroma_retention = float(np.clip(chroma_retention, 0.84, 1.0))

    # Identity is a real rendering contract, not merely the numerical limit
    # of the harmonisation formula.  Running a nominally neutral product
    # through low-pass subtraction/addition still introduces float rounding,
    # and a synthetic key gradient changes photographed highlights even when
    # the caller requested no colour adaptation.  Bypass the operation
    # completely when every authorised low-frequency adjustment is neutral so
    # source colour, gloss and high-frequency texture remain byte-exact.
    if (
        np.array_equal(gain, np.ones(3, np.float32))
        and chroma_retention == 1.0
        and (shadow_direction is None or relight_strength <= 0.0)
    ):
        return rgba.copy()

    alpha_u8 = rgba[..., 3]
    occupied = np.argwhere(alpha_u8 > 4)
    if not occupied.size:
        return rgba.copy()
    y0, x0 = occupied.min(axis=0)
    y1, x1 = occupied.max(axis=0)
    extent = max(int(y1 - y0 + 1), int(x1 - x0 + 1))
    # At the current source-to-canvas transforms this becomes roughly 4--15
    # output pixels: below object-scale lighting, above text/texture scale.
    sigma = float(np.clip(extent * 0.018, 4.0, 48.0))

    rgb = rgba[..., :3].astype(np.float32)
    alpha = alpha_u8.astype(np.float32) / 255.0
    low = _alpha_normalized_lowpass(rgb, alpha, sigma)
    target_low = low * gain.reshape(1, 1, 3)
    target_luma = target_low @ _LUMA
    target_low = (
        target_luma[..., None]
        + (target_low - target_luma[..., None]) * chroma_retention
    )

    if shadow_direction is not None and relight_strength > 0.0:
        yy, xx = np.indices(alpha.shape, dtype=np.float32)
        x_mid, y_mid = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        x_norm = (xx - x_mid) / max(1.0, (x1 - x0) * 0.5)
        y_norm = (yy - y_mid) / max(1.0, (y1 - y0) * 0.5)
        light = -np.asarray(shadow_direction, np.float32)
        normalizer = max(1e-6, float(np.abs(light).sum()))
        projection = np.clip(
            (light[0] * x_norm + light[1] * y_norm) / normalizer,
            -1.0,
            1.0,
        )
        factor = 1.0 + float(np.clip(relight_strength, 0.0, 0.08)) * projection
        target_low *= factor[..., None]

    correction = target_low - low
    correction = cv2.GaussianBlur(
        correction,
        (0, 0),
        sigmaX=max(1.0, sigma * 0.35),
        sigmaY=max(1.0, sigma * 0.35),
        borderType=cv2.BORDER_REPLICATE,
    )
    active = alpha_u8 > 0
    result = rgba.copy()
    corrected = np.clip(np.round(rgb + correction), 0, 255).astype(np.uint8)
    result[..., :3][active] = corrected[active]
    return result
