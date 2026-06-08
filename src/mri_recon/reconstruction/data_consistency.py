from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mri_recon.transforms import fft2c, ifft2c


def expand_mask_for_kspace(
    mask: NDArray[np.bool_],
    kspace_shape: tuple[int, ...],
) -> NDArray[np.bool_]:
    """Expand a 1D sampling mask to match a k-space array shape.

    Args:
        mask: 1D sampling mask with shape [width].
        kspace_shape: Target k-space shape, usually [height, width].

    Returns:
        Broadcastable boolean mask with shape [1, width] for single-coil data.

    Raises:
        ValueError: If the mask shape is incompatible with k-space.
    """
    if mask.ndim != 1:
        raise ValueError(f"Expected 1D mask, got shape {mask.shape}")

    if mask.shape[0] != kspace_shape[-1]:
        raise ValueError(
            f"Mask width {mask.shape[0]} does not match "
            f"k-space width {kspace_shape[-1]}."
        )

    expanded_shape = (1,) * (len(kspace_shape) - 1) + (mask.shape[0],)
    return mask.reshape(expanded_shape)


def apply_hard_data_consistency(
    predicted_image: NDArray[np.complexfloating],
    measured_kspace: NDArray[np.complexfloating],
    mask: NDArray[np.bool_],
) -> NDArray[np.complexfloating]:
    """Apply hard k-space data consistency.

    At sampled k-space locations, replace the predicted k-space values with
    the originally measured k-space values.

    Args:
        predicted_image: Complex-valued predicted image with shape [height, width].
        measured_kspace: Measured undersampled k-space with shape [height, width].
        mask: 1D boolean sampling mask with shape [width].

    Returns:
        Complex-valued data-consistent image with shape [height, width].
    """
    predicted_kspace = fft2c(predicted_image)
    expanded_mask = expand_mask_for_kspace(mask, measured_kspace.shape)

    corrected_kspace = np.where(
        expanded_mask,
        measured_kspace,
        predicted_kspace,
    )

    return ifft2c(corrected_kspace)


def compute_kspace_consistency_error(
    reconstructed_image: NDArray[np.complexfloating],
    measured_kspace: NDArray[np.complexfloating],
    mask: NDArray[np.bool_],
) -> dict[str, float]:
    """Compute k-space consistency error at sampled locations.

    Args:
        reconstructed_image: Complex-valued reconstructed image.
        measured_kspace: Measured undersampled k-space.
        mask: 1D boolean sampling mask.

    Returns:
        Dictionary containing mean and max absolute k-space error at sampled points.
    """
    reconstructed_kspace = fft2c(reconstructed_image)

    expanded_mask = expand_mask_for_kspace(
        mask=mask,
        kspace_shape=measured_kspace.shape,
    )

    full_mask = np.broadcast_to(expanded_mask, measured_kspace.shape)

    sampled_error = np.abs(reconstructed_kspace[full_mask] - measured_kspace[full_mask])

    return {
        "mean_abs_error": float(sampled_error.mean()),
        "max_abs_error": float(sampled_error.max()),
    }
