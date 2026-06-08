from __future__ import annotations

import numpy as np

from mri_recon.evaluation.metrics import compute_reconstruction_metrics
from mri_recon.transforms import center_crop


def channels_to_complex_numpy(array: np.ndarray) -> np.ndarray:
    """Convert [2, H, W] real/imag channels to [H, W] complex array."""
    if array.ndim != 3 or array.shape[0] != 2:
        raise ValueError(f"Expected [2, H, W] array, got shape {array.shape}")

    return array[0] + 1j * array[1]


def target_scaled_magnitude_pair(
    prediction_channels: np.ndarray,
    target_channels: np.ndarray,
    target_shape: tuple[int, int] = (320, 320),
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert complex channel outputs to magnitude images on one target scale.

    This is intended for metrics, not visualization. Unlike per-image min-max
    normalization, the prediction is scaled only by the target magnitude scale.
    """
    prediction_complex = channels_to_complex_numpy(prediction_channels)
    target_complex = channels_to_complex_numpy(target_channels)

    prediction_magnitude = center_crop(
        np.abs(prediction_complex).astype(np.float32),
        target_shape=target_shape,
    )
    target_magnitude = center_crop(
        np.abs(target_complex).astype(np.float32),
        target_shape=target_shape,
    )

    scale = float(target_magnitude.max())
    if scale < eps:
        scale = 1.0

    return prediction_magnitude / scale, target_magnitude / scale


def complex_image_target_scaled_magnitude_pair(
    prediction_complex: np.ndarray,
    target_complex: np.ndarray,
    target_shape: tuple[int, int] = (320, 320),
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert complex images to cropped magnitudes on one target scale."""
    prediction_magnitude = center_crop(
        np.abs(prediction_complex).astype(np.float32),
        target_shape=target_shape,
    )
    target_magnitude = center_crop(
        np.abs(target_complex).astype(np.float32),
        target_shape=target_shape,
    )

    scale = float(target_magnitude.max())
    if scale < eps:
        scale = 1.0

    return prediction_magnitude / scale, target_magnitude / scale


def compute_complex_channel_metrics(
    prediction_channels: np.ndarray,
    target_channels: np.ndarray,
    target_shape: tuple[int, int] = (320, 320),
) -> dict[str, float]:
    """Compute PSNR/SSIM/MAE for complex-channel reconstructions."""
    prediction_magnitude, target_magnitude = target_scaled_magnitude_pair(
        prediction_channels=prediction_channels,
        target_channels=target_channels,
        target_shape=target_shape,
    )

    return compute_reconstruction_metrics(
        prediction=prediction_magnitude,
        target=target_magnitude,
    )


def compute_complex_image_metrics(
    prediction_complex: np.ndarray,
    target_complex: np.ndarray,
    target_shape: tuple[int, int] = (320, 320),
) -> dict[str, float]:
    """Compute PSNR/SSIM/MAE for complex image reconstructions."""
    prediction_magnitude, target_magnitude = complex_image_target_scaled_magnitude_pair(
        prediction_complex=prediction_complex,
        target_complex=target_complex,
        target_shape=target_shape,
    )

    return compute_reconstruction_metrics(
        prediction=prediction_magnitude,
        target=target_magnitude,
    )
