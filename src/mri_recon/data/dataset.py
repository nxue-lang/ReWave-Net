from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from mri_recon.baselines import zero_filled_reconstruction
from mri_recon.io import load_array_from_h5, load_kspace_from_h5
from mri_recon.sampling import (
    apply_undersampling_mask,
    create_cartesian_undersampling_mask,
)
from mri_recon.transforms import center_crop, normalize_to_unit_range


class FastMRISingleCoilDataset(Dataset):
    """fastMRI single-coil slice dataset for image-domain reconstruction.

    Each sample is:
        input: zero-filled reconstruction from undersampled k-space
        target: ground-truth reconstruction_esc
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

        kspace_slice = self.kspace[slice_index]
        target_slice = self.target[slice_index]

        mask = create_cartesian_undersampling_mask(
            width=kspace_slice.shape[-1],
            acceleration=self.acceleration,
            center_fraction=self.center_fraction,
            seed=self.seed,
        )

        undersampled_kspace = apply_undersampling_mask(
            kspace=kspace_slice,
            mask=mask,
        )

        zero_filled = zero_filled_reconstruction(undersampled_kspace)
        zero_filled = center_crop(zero_filled, target_shape=target_slice.shape)

        zero_filled = normalize_to_unit_range(zero_filled).astype(np.float32)
        target_slice = normalize_to_unit_range(target_slice).astype(np.float32)

        input_tensor = torch.from_numpy(zero_filled).unsqueeze(0)
        target_tensor = torch.from_numpy(target_slice).unsqueeze(0)

        return {
            "input": input_tensor,
            "target": target_tensor,
            "slice_index": slice_index,
        }
