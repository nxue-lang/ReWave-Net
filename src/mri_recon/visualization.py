from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from numpy.typing import NDArray


def save_grayscale_image(
    image: NDArray,
    output_path: str | Path,
    title: str | None = None,
    dpi: int = 200,
) -> None:
    """Save a single grayscale image.

    Args:
        image: Image array to save.
        output_path: Output file path.
        title: Optional figure title.
        dpi: Output resolution.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap="gray")

    if title is not None:
        plt.title(title)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_image_grid(
    images: list[NDArray],
    titles: list[str],
    output_path: str | Path,
    dpi: int = 200,
) -> None:
    """Save a horizontal grid of grayscale images.

    Args:
        images: List of image arrays.
        titles: List of image titles.
        output_path: Output file path.
        dpi: Output resolution.
    """
    if len(images) != len(titles):
        raise ValueError(
            f"Expected same number of images and titles, got "
            f"{len(images)} images and {len(titles)} titles."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_images = len(images)
    plt.figure(figsize=(5 * num_images, 5))

    for index, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1, num_images, index + 1)
        plt.imshow(image, cmap="gray")
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def save_reconstruction_detail_comparison(
    target: NDArray,
    zero_filled: NDArray,
    reconstruction: NDArray,
    output_path: str | Path,
    method_name: str = "ReWave-Net",
    crop_size: int = 96,
    dpi: int = 200,
) -> None:
    """Save full-image and automatically selected detail-region comparisons."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = target.shape
    crop_size = min(crop_size, height, width)
    gradient_y, gradient_x = np.gradient(target.astype(np.float32))
    detail_score = np.hypot(gradient_x, gradient_y)

    margin_y = max((height - crop_size) // 10, 0)
    margin_x = max((width - crop_size) // 10, 0)
    best_score = float("-inf")
    best_top = max((height - crop_size) // 2, 0)
    best_left = max((width - crop_size) // 2, 0)
    stride = max(crop_size // 8, 1)

    for top in range(margin_y, height - crop_size - margin_y + 1, stride):
        for left in range(margin_x, width - crop_size - margin_x + 1, stride):
            region = detail_score[top : top + crop_size, left : left + crop_size]
            score = float(region.mean())
            if score > best_score:
                best_score = score
                best_top = top
                best_left = left

    error = np.abs(reconstruction - target)
    images = [target, zero_filled, reconstruction, error]
    titles = ["Target", "Zero-filled", method_name, "Absolute Error"]
    error_vmax = max(float(np.quantile(error, 0.995)), 1e-8)

    figure, axes = plt.subplots(2, 4, figsize=(16, 8))
    for column, (image, title) in enumerate(zip(images, titles)):
        vmax = error_vmax if column == 3 else 1.0
        axes[0, column].imshow(image, cmap="gray", vmin=0.0, vmax=vmax)
        axes[0, column].add_patch(
            Rectangle(
                (best_left, best_top),
                crop_size,
                crop_size,
                edgecolor="red",
                facecolor="none",
                linewidth=1.5,
            )
        )
        axes[0, column].set_title(title)
        axes[0, column].axis("off")

        crop = image[
            best_top : best_top + crop_size,
            best_left : best_left + crop_size,
        ]
        axes[1, column].imshow(crop, cmap="gray", vmin=0.0, vmax=vmax)
        axes[1, column].set_title(f"{title} detail")
        axes[1, column].axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
