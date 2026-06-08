from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_mae(
    prediction: NDArray[np.floating],
    target: NDArray[np.floating],
) -> float:
    """Compute mean absolute error."""
    return float(np.mean(np.abs(prediction - target)))


def compute_psnr(
    prediction: NDArray[np.floating],
    target: NDArray[np.floating],
    data_range: float = 1.0,
) -> float:
    """Compute peak signal-to-noise ratio."""
    return float(
        peak_signal_noise_ratio(
            target,
            prediction,
            data_range=data_range,
        )
    )


def compute_ssim(
    prediction: NDArray[np.floating],
    target: NDArray[np.floating],
    data_range: float = 1.0,
) -> float:
    """Compute structural similarity index."""
    return float(
        structural_similarity(
            target,
            prediction,
            data_range=data_range,
        )
    )


def compute_reconstruction_metrics(
    prediction: NDArray[np.floating],
    target: NDArray[np.floating],
) -> dict[str, float]:
    """Compute standard MRI reconstruction metrics."""
    return {
        "psnr": compute_psnr(prediction, target),
        "ssim": compute_ssim(prediction, target),
        "mae": compute_mae(prediction, target),
    }
