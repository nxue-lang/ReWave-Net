from __future__ import annotations

from _bootstrap import add_project_src_to_path

add_project_src_to_path()

import argparse
import csv
from pathlib import Path

import numpy as np

from mri_recon.evaluation.complex_metrics import (
    complex_image_target_scaled_magnitude_pair,
)
from mri_recon.evaluation.metrics import compute_reconstruction_metrics
from mri_recon.io import get_middle_slice_index, load_array_from_h5, load_kspace_from_h5
from mri_recon.sampling import (
    apply_undersampling_mask,
    create_cartesian_undersampling_mask,
)
from mri_recon.transforms import ifft2c
from mri_recon.visualization import save_image_grid


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate zero-filled reconstruction against ground truth."
    )

    parser.add_argument(
        "--h5-path",
        type=str,
        default="data/knee_singlecoil_val/file1000000.h5",
        help="Path to a fastMRI single-coil validation HDF5 file.",
    )

    parser.add_argument(
        "--target-key",
        type=str,
        default="reconstruction_esc",
        help="Ground-truth reconstruction key.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory.",
    )

    return parser.parse_args()


def save_metrics_csv(metrics: dict[str, float], output_path: Path) -> None:
    """Save reconstruction metrics to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["method", *metrics.keys()])
        writer.writeheader()
        writer.writerow({"method": "zero_filled", **metrics})


def main() -> None:
    args = parse_args()

    h5_path = Path(args.h5_path)
    output_dir = Path(args.output_dir)
    figure_dir = output_dir / "figures"
    metrics_dir = output_dir / "metrics"
    npz_dir = output_dir / "npz"

    kspace = load_kspace_from_h5(h5_path)
    target = load_array_from_h5(h5_path, key=args.target_key)

    slice_index = get_middle_slice_index(kspace)

    kspace_slice = kspace[slice_index]
    target_slice = target[slice_index].astype(np.float32)

    mask = create_cartesian_undersampling_mask(
        width=kspace_slice.shape[-1],
        acceleration=4,
        center_fraction=0.08,
        seed=0,
    )

    undersampled_kspace_slice = apply_undersampling_mask(
        kspace=kspace_slice,
        mask=mask,
    )

    full_complex = ifft2c(kspace_slice)
    zero_filled_complex = ifft2c(undersampled_kspace_slice)
    zero_filled_normalized, target_normalized = (
        complex_image_target_scaled_magnitude_pair(
            prediction_complex=zero_filled_complex,
            target_complex=full_complex,
            target_shape=target_slice.shape,
        )
    )

    error_map = np.abs(zero_filled_normalized - target_normalized)

    metrics = compute_reconstruction_metrics(
        prediction=zero_filled_normalized,
        target=target_normalized,
    )

    # Keep a crop shape check because fastMRI target is typically 320 x 320.
    zero_filled_cropped_shape = zero_filled_normalized.shape
    if zero_filled_cropped_shape != target_slice.shape:
        raise ValueError(
            f"Metric crop shape {zero_filled_cropped_shape} does not match "
            f"target shape {target_slice.shape}"
        )

    figure_path = figure_dir / "zero_filled_vs_ground_truth.png"
    save_image_grid(
        images=[
            target_normalized,
            zero_filled_normalized,
            error_map,
        ],
        titles=[
            "Ground Truth",
            "Zero-filled",
            "Absolute Error",
        ],
        output_path=figure_path,
    )

    metrics_path = metrics_dir / "zero_filled_metrics.csv"
    save_metrics_csv(metrics, metrics_path)

    npz_dir.mkdir(parents=True, exist_ok=True)
    npz_path = npz_dir / "zero_filled_evaluation.npz"
    np.savez_compressed(
        npz_path,
        target=target_normalized,
        zero_filled=zero_filled_normalized,
        error_map=error_map,
        mask=mask,
        undersampled_kspace=undersampled_kspace_slice,
        slice_index=slice_index,
    )

    print("Zero-filled baseline evaluation completed.")
    print(f"HDF5 file: {h5_path}")
    print(f"K-space shape: {kspace.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Selected slice index: {slice_index}")
    print(f"Zero-filled cropped shape: {zero_filled_cropped_shape}")
    print(f"Target slice shape: {target_normalized.shape}")
    print(f"PSNR: {metrics['psnr']:.4f}")
    print(f"SSIM: {metrics['ssim']:.4f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"Saved figure to: {figure_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved arrays to: {npz_path}")
    print("Undersampling acceleration: 4")
    print(f"Mask sampled columns: {int(mask.sum())} / {mask.shape[0]}")


if __name__ == "__main__":
    main()
