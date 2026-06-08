from __future__ import annotations

from _bootstrap import add_project_src_to_path

add_project_src_to_path()

import argparse
from pathlib import Path

import numpy as np

from mri_recon.baselines import zero_filled_reconstruction
from mri_recon.io import get_middle_slice_index, load_kspace_from_h5, load_yaml_config
from mri_recon.visualization import save_grayscale_image


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run zero-filled reconstruction for a single MRI sample."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/single_sample.yaml",
        help="Path to the YAML config file.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    h5_path = Path(config["data"]["h5_path"])
    figure_dir = Path(config["output"]["figure_dir"])
    npz_dir = Path(config["output"]["npz_dir"])

    use_root_sum_of_squares = bool(config["reconstruction"]["use_root_sum_of_squares"])

    kspace = load_kspace_from_h5(h5_path)

    slice_index = config["data"].get("slice_index")
    if slice_index is None:
        slice_index = get_middle_slice_index(kspace)

    kspace_slice = kspace[slice_index]

    zero_filled_image = zero_filled_reconstruction(
        kspace_slice=kspace_slice,
        use_root_sum_of_squares=use_root_sum_of_squares,
    )

    figure_path = figure_dir / "single_sample_zero_filled.png"
    save_grayscale_image(
        image=zero_filled_image,
        output_path=figure_path,
        title="Zero-Filled Reconstruction",
    )

    npz_dir.mkdir(parents=True, exist_ok=True)
    npz_path = npz_dir / "single_sample_zero_filled.npz"
    np.savez_compressed(
        npz_path,
        zero_filled=zero_filled_image,
        slice_index=slice_index,
    )

    print("Single-sample baseline completed.")
    print(f"HDF5 file: {h5_path}")
    print(f"K-space shape: {kspace.shape}")
    print(f"Selected slice index: {slice_index}")
    print(f"Selected k-space slice shape: {kspace_slice.shape}")
    print(f"Zero-filled image shape: {zero_filled_image.shape}")
    print(f"Saved figure to: {figure_path}")
    print(f"Saved array to: {npz_path}")


if __name__ == "__main__":
    main()
