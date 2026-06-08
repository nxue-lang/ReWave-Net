from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import yaml
from numpy.typing import NDArray


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed YAML configuration as a dictionary.
    """
    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


def load_kspace_from_h5(h5_path: str | Path, key: str = "kspace") -> NDArray:
    """Load k-space data from a fastMRI-style HDF5 file.

    Args:
        h5_path: Path to the HDF5 file.
        key: Dataset key for k-space data.

    Returns:
        K-space array.

    Raises:
        KeyError: If the requested key does not exist.
        FileNotFoundError: If the HDF5 file does not exist.
    """
    h5_path = Path(h5_path)

    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with h5py.File(h5_path, "r") as h5_file:
        available_keys = list(h5_file.keys())

        if key not in h5_file:
            raise KeyError(
                f"Key '{key}' not found in {h5_path}. "
                f"Available keys: {available_keys}"
            )

        kspace = h5_file[key][()]

    return kspace


def get_middle_slice_index(kspace: NDArray) -> int:
    """Return the middle slice index for a k-space volume."""
    if kspace.ndim < 3:
        raise ValueError(f"Expected k-space with slice dimension, got {kspace.shape}")

    return kspace.shape[0] // 2


def load_array_from_h5(h5_path: str | Path, key: str) -> NDArray:
    """Load an array from an HDF5 file.

    Args:
        h5_path: Path to the HDF5 file.
        key: Dataset key.

    Returns:
        Array stored under the given key.

    Raises:
        KeyError: If the requested key does not exist.
        FileNotFoundError: If the HDF5 file does not exist.
    """
    h5_path = Path(h5_path)

    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with h5py.File(h5_path, "r") as h5_file:
        available_keys = list(h5_file.keys())

        if key not in h5_file:
            raise KeyError(
                f"Key '{key}' not found in {h5_path}. "
                f"Available keys: {available_keys}"
            )

        array = h5_file[key][()]

    return array


def load_mask_from_h5(h5_path: str | Path, key: str = "mask") -> NDArray:
    """Load a sampling mask from an HDF5 file.

    Args:
        h5_path: Path to the HDF5 file.
        key: Dataset key for the sampling mask.

    Returns:
        Boolean sampling mask.
    """
    return load_array_from_h5(h5_path=h5_path, key=key).astype(bool)
