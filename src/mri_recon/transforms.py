from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complexfloating]
FloatArray = NDArray[np.floating]


def fft2c(image: ComplexArray) -> ComplexArray:
    """Apply centered 2D FFT.

    Args:
        image: Complex-valued image array with spatial dimensions in the last two axes.

    Returns:
        Centered 2D Fourier transform of the input image.
    """
    return np.fft.fftshift(
        np.fft.fft2(
            np.fft.ifftshift(image, axes=(-2, -1)),
            norm="ortho",
        ),
        axes=(-2, -1),
    )


def ifft2c(kspace: ComplexArray) -> ComplexArray:
    """Apply centered 2D inverse FFT.

    Args:
        kspace: Complex k-space array with spatial dimensions in the last two axes.

    Returns:
        Complex-valued image-domain array.
    """
    return np.fft.fftshift(
        np.fft.ifft2(
            np.fft.ifftshift(kspace, axes=(-2, -1)),
            norm="ortho",
        ),
        axes=(-2, -1),
    )


def normalize_to_unit_range(image: NDArray) -> FloatArray:
    """Normalize an image to [0, 1] for visualization.

    Args:
        image: Real or complex image array.

    Returns:
        Magnitude image normalized to [0, 1].
    """
    magnitude = np.abs(image).astype(np.float32)
    magnitude = magnitude - magnitude.min()

    max_value = magnitude.max()
    if max_value > 0:
        magnitude = magnitude / max_value

    return magnitude


def center_crop(image: NDArray, target_shape: tuple[int, int]) -> NDArray:
    """Center crop an image to the target spatial shape.

    Args:
        image: Input image with spatial dimensions in the last two axes.
        target_shape: Target shape as (height, width).

    Returns:
        Center-cropped image.
    """
    target_height, target_width = target_shape
    height, width = image.shape[-2:]

    if target_height > height or target_width > width:
        raise ValueError(
            f"Target shape {target_shape} cannot be larger than "
            f"input shape {image.shape}."
        )

    top = (height - target_height) // 2
    left = (width - target_width) // 2

    return image[..., top : top + target_height, left : left + target_width]


def center_pad(
    image: NDArray,
    target_shape: tuple[int, int],
) -> NDArray:
    """Center pad an image to the target spatial shape.

    Args:
        image: Input image with spatial dimensions in the last two axes.
        target_shape: Target shape as (height, width).

    Returns:
        Center-padded image.
    """
    target_height, target_width = target_shape
    height, width = image.shape[-2:]

    if target_height < height or target_width < width:
        raise ValueError(
            f"Target shape {target_shape} cannot be smaller than "
            f"input shape {image.shape}."
        )

    pad_top = (target_height - height) // 2
    pad_bottom = target_height - height - pad_top

    pad_left = (target_width - width) // 2
    pad_right = target_width - width - pad_left

    pad_width = [(0, 0)] * image.ndim
    pad_width[-2] = (pad_top, pad_bottom)
    pad_width[-1] = (pad_left, pad_right)

    return np.pad(image, pad_width=pad_width, mode="constant")
