from __future__ import annotations

from _bootstrap import add_project_src_to_path

add_project_src_to_path()

import argparse
from pathlib import Path

import numpy as np

from mri_recon.io import get_middle_slice_index, load_kspace_from_h5, load_mask_from_h5
from mri_recon.reconstruction.data_consistency import (
    apply_hard_data_consistency,
    compute_kspace_consistency_error,
)
from mri_recon.transforms import ifft2c, normalize_to_unit_range
from mri_recon.visualization import save_image_grid


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Check hard data consistency on a single MRI sample."
    )

    parser.add_argument(
        "--h5-path",
        type=str,
        default="data/knee_singlecoil_val/file1000000.h5",
        help="Path to the HDF5 file.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/figures",
        help="Directory for output figures.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    h5_path = Path(args.h5_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kspace = load_kspace_from_h5(h5_path)
    mask = load_mask_from_h5(h5_path)

    slice_index = get_middle_slice_index(kspace)
    measured_kspace = kspace[slice_index]

    zero_filled_complex = ifft2c(measured_kspace)

    # For this sanity check, pretend zero-filled image is the model prediction.
    data_consistent_complex = apply_hard_data_consistency(
        predicted_image=zero_filled_complex,
        measured_kspace=measured_kspace,
        mask=mask,
    )

    consistency_error = compute_kspace_consistency_error(
        reconstructed_image=data_consistent_complex,
        measured_kspace=measured_kspace,
        mask=mask,
    )

    zero_filled_image = normalize_to_unit_range(zero_filled_complex)
    data_consistent_image = normalize_to_unit_range(data_consistent_complex)
    difference_image = normalize_to_unit_range(
        np.abs(data_consistent_complex - zero_filled_complex)
    )

    figure_path = output_dir / "data_consistency_check.png"

    save_image_grid(
        images=[
            zero_filled_image,
            data_consistent_image,
            difference_image,
        ],
        titles=[
            "Zero-filled",
            "After Data Consistency",
            "Difference",
        ],
        output_path=figure_path,
    )

    print("Data consistency check completed.")
    print(f"HDF5 file: {h5_path}")
    print(f"K-space shape: {kspace.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask sampled columns: {int(mask.sum())} / {mask.shape[0]}")
    print(f"Selected slice index: {slice_index}")
    print(f"Mean sampled k-space error: {consistency_error['mean_abs_error']:.6e}")
    print(f"Max sampled k-space error: {consistency_error['max_abs_error']:.6e}")
    print(f"Saved figure to: {figure_path}")


if __name__ == "__main__":
    main()
