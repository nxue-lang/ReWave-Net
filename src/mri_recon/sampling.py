from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def create_cartesian_undersampling_mask(
    width: int,
    acceleration: int = 4,
    center_fraction: float = 0.08,
    seed: int = 0,
) -> NDArray[np.bool_]:
    """Create a 1D Cartesian undersampling mask for MRI k-space.

    Args:
        width: Number of k-space columns.
        acceleration: Approximate acceleration factor.
        center_fraction: Fraction of low-frequency center columns to always keep.
        seed: Random seed for reproducibility.

    Returns:
        Boolean mask with shape [width].
    """
    if not 0 < center_fraction < 1:
        raise ValueError(f"center_fraction must be in (0, 1), got {center_fraction}")

    rng = np.random.default_rng(seed)

    num_low_freqs = int(round(width * center_fraction))
    num_low_freqs = max(num_low_freqs, 1)

    mask = np.zeros(width, dtype=bool)

    center = width // 2
    left = center - num_low_freqs // 2
    right = left + num_low_freqs
    mask[left:right] = True

    target_samples = width // acceleration
    remaining_samples = max(target_samples - num_low_freqs, 0)

    candidate_indices = np.concatenate(
        [
            np.arange(0, left),
            np.arange(right, width),
        ]
    )

    if remaining_samples > len(candidate_indices):
        remaining_samples = len(candidate_indices)

    sampled_indices = rng.choice(
        candidate_indices,
        size=remaining_samples,
        replace=False,
    )

    mask[sampled_indices] = True

    return mask


def apply_undersampling_mask(
    kspace: NDArray[np.complexfloating],
    mask: NDArray[np.bool_],
) -> NDArray[np.complexfloating]:
    """Apply a 1D Cartesian undersampling mask to k-space.

    Args:
        kspace: K-space array with spatial width in the last dimension.
        mask: Boolean sampling mask with shape [width].

    Returns:
        Undersampled k-space with missing columns set to zero.
    """
    if mask.ndim != 1:
        raise ValueError(f"Expected 1D mask, got shape {mask.shape}")

    if mask.shape[0] != kspace.shape[-1]:
        raise ValueError(
            f"Mask width {mask.shape[0]} does not match "
            f"k-space width {kspace.shape[-1]}"
        )

    expanded_shape = (1,) * (kspace.ndim - 1) + (mask.shape[0],)
    expanded_mask = mask.reshape(expanded_shape)

    return np.where(expanded_mask, kspace, 0.0)
