from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mri_recon.transforms import ifft2c, normalize_to_unit_range


def root_sum_of_squares(
    coil_images: NDArray[np.complexfloating],
) -> NDArray[np.floating]:
    """Combine multi-coil images using root-sum-of-squares.

    Args:
        coil_images: Complex-valued coil images with shape [num_coils, height, width].

    Returns:
        Coil-combined magnitude image with shape [height, width].
    """
    return np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=0)).astype(np.float32)


def zero_filled_reconstruction(
    kspace_slice: NDArray[np.complexfloating],
    use_root_sum_of_squares: bool = True,
) -> NDArray[np.floating]:
    """Compute a zero-filled reconstruction from one k-space slice.

    Args:
        kspace_slice: Single slice of k-space.
            Expected shape:
            - single-coil: [height, width]
            - multi-coil: [num_coils, height, width]
        use_root_sum_of_squares: Whether to combine multi-coil images using RSS.

    Returns:
        Normalized zero-filled magnitude image.
    """
    if kspace_slice.ndim == 2:
        image = ifft2c(kspace_slice)
        return normalize_to_unit_range(image)

    if kspace_slice.ndim == 3:
        coil_images = ifft2c(kspace_slice)

        if not use_root_sum_of_squares:
            raise ValueError(
                "Multi-coil k-space requires coil combination. "
                "Currently only root-sum-of-squares is supported."
            )

        combined_image = root_sum_of_squares(coil_images)
        return normalize_to_unit_range(combined_image)

    raise ValueError(
        "Expected k-space slice with shape [height, width] or "
        f"[num_coils, height, width], got {kspace_slice.shape}"
    )
