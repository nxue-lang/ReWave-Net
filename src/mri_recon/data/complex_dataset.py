from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from mri_recon.io import load_array_from_h5, load_kspace_from_h5
from mri_recon.sampling import (
    apply_undersampling_mask,
    create_cartesian_undersampling_mask,
)
from mri_recon.transforms import ifft2c, normalize_to_unit_range


def complex_to_channels(array: np.ndarray) -> np.ndarray:
    """Convert a complex array [H, W] to a 2-channel real array [2, H, W]."""
    return np.stack([array.real, array.imag], axis=0).astype(np.float32)


def channels_to_complex(array: np.ndarray) -> np.ndarray:
    """Convert a 2-channel real array [2, H, W] to a complex array [H, W]."""
    return array[0] + 1j * array[1]


class FastMRIComplexSingleCoilDataset(Dataset):
    """fastMRI single-coil dataset for full-resolution complex reconstruction.

    Each sample returns:
        input: zero-filled complex image, shape [2, H, W]
        target_complex: fully-sampled complex image, shape [2, H, W]
        target_magnitude: cropped ground-truth magnitude image, shape [1, 320, 320]
        measured_kspace: undersampled normalized k-space, shape [2, H, W]
        mask: sampling mask, shape [W]
    """

    def __init__(
        self,
        h5_path: str | Path,
        target_key: str = "reconstruction_esc",
        acceleration: int = 4,
        center_fraction: float = 0.08,
        slice_indices: list[int] | None = None,
        seed: int = 0,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.target_key = target_key
        self.acceleration = acceleration
        self.center_fraction = center_fraction
        self.seed = seed

        self.kspace = load_kspace_from_h5(self.h5_path)
        self.target = load_array_from_h5(self.h5_path, key=self.target_key).astype(
            np.float32
        )

        if slice_indices is None:
            self.slice_indices = list(range(self.kspace.shape[0]))
        else:
            self.slice_indices = slice_indices

    def __len__(self) -> int:
        return len(self.slice_indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        slice_index = self.slice_indices[index]

        full_kspace = self.kspace[slice_index]
        target_magnitude = self.target[slice_index]

        full_complex_image = ifft2c(full_kspace)

        scale = float(np.max(np.abs(full_complex_image)))
        if scale <= 0:
            scale = 1.0

        full_complex_image = full_complex_image / scale
        normalized_full_kspace = full_kspace / scale

        mask = create_cartesian_undersampling_mask(
            width=full_kspace.shape[-1],
            acceleration=self.acceleration,
            center_fraction=self.center_fraction,
            seed=self.seed,
        )

        undersampled_kspace = apply_undersampling_mask(
            kspace=normalized_full_kspace,
            mask=mask,
        )

        zero_filled_complex = ifft2c(undersampled_kspace)

        target_magnitude = normalize_to_unit_range(target_magnitude).astype(np.float32)

        return {
            "input": torch.from_numpy(complex_to_channels(zero_filled_complex)),
            "target_complex": torch.from_numpy(complex_to_channels(full_complex_image)),
            "target_magnitude": torch.from_numpy(target_magnitude).unsqueeze(0),
            "measured_kspace": torch.from_numpy(
                complex_to_channels(undersampled_kspace)
            ),
            "mask": torch.from_numpy(mask.astype(np.bool_)),
            "slice_index": slice_index,
        }
