from __future__ import annotations

import argparse
from pathlib import Path

import h5py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect an HDF5 MRI file.")
    parser.add_argument(
        "--h5-path",
        type=str,
        default="data/knee_singlecoil_val/file1000000.h5",
        help="Path to the HDF5 file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    h5_path = Path(args.h5_path)

    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with h5py.File(h5_path, "r") as h5_file:
        print(f"HDF5 file: {h5_path}")
        print("\nKeys:")

        for key in h5_file.keys():
            item = h5_file[key]

            if hasattr(item, "shape"):
                print(f"  {key}: shape={item.shape}, dtype={item.dtype}")
            else:
                print(f"  {key}: {type(item)}")

        print("\nAttributes:")
        for key, value in h5_file.attrs.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
