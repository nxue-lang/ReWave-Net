from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
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
