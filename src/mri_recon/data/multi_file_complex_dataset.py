from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from mri_recon.sampling import (
    apply_undersampling_mask,
    create_cartesian_undersampling_mask,
)
from mri_recon.transforms import center_crop, ifft2c, normalize_to_unit_range


def complex_to_channels(array: np.ndarray) -> np.ndarray:
    return np.stack([array.real, array.imag], axis=0).astype(np.float32)


def channels_to_complex(array: np.ndarray) -> np.ndarray:
    return array[0] + 1j * array[1]


class MultiFileFastMRIComplexSingleCoilDataset(Dataset):
    """Multi-file fastMRI single-coil complex reconstruction dataset.

    Each sample is one slice from one h5 volume.

    Returns:
        input: zero-filled complex image, [2, H, W]
        target_complex: full complex image, [2, H, W]
        target_magnitude: cropped target magnitude, [1, 320, 320]
        measured_kspace: undersampled k-space, [2, H, W]
        mask: sampling mask, [W]
        file_path: source h5 file path
        slice_index: slice index inside that file
    """

    def __init__(
        self,
        h5_paths: list[str | Path],
        acceleration: int = 4,
        center_fraction: float = 0.08,
        use_middle_slices_only: bool = True,
        middle_slice_margin: int = 5,
        mask_seed: int | None = None,
    ) -> None:
        super().__init__()

        self.h5_paths = [Path(path) for path in h5_paths]
        self.acceleration = acceleration
        self.center_fraction = center_fraction
        self.use_middle_slices_only = use_middle_slices_only
        self.middle_slice_margin = middle_slice_margin
        self.mask_seed = mask_seed

        self.samples: list[tuple[Path, int]] = []

        for h5_path in self.h5_paths:
            with h5py.File(h5_path, "r") as hf:
                num_slices = hf["kspace"].shape[0]

            if use_middle_slices_only:
                center = num_slices // 2
                start = max(0, center - middle_slice_margin)
                end = min(num_slices, center + middle_slice_margin + 1)
                slice_indices = list(range(start, end))
            else:
                slice_indices = list(range(num_slices))

            for slice_index in slice_indices:
                self.samples.append((h5_path, slice_index))

    def __len__(self) -> int:
        return len(self.samples)

    def _mask_seed_for_sample(self, h5_path: Path, slice_index: int) -> int:
        if self.mask_seed is None:
            return slice_index

        seed_key = f"{self.mask_seed}:{h5_path.name}:{slice_index}".encode("utf-8")
        digest = hashlib.blake2s(seed_key, digest_size=4).digest()
        return int.from_bytes(digest, byteorder="little", signed=False)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        h5_path, slice_index = self.samples[index]

        with h5py.File(h5_path, "r") as hf:
            full_kspace = hf["kspace"][slice_index]
            target_magnitude = hf["reconstruction_esc"][slice_index]

        full_complex_image = ifft2c(full_kspace)

        scale = np.max(np.abs(full_complex_image)) + 1e-8
        full_complex_image = full_complex_image / scale
        full_kspace = full_kspace / scale

        width = full_kspace.shape[-1]
        mask = create_cartesian_undersampling_mask(
            width=width,
            acceleration=self.acceleration,
            center_fraction=self.center_fraction,
            seed=self._mask_seed_for_sample(h5_path, slice_index),
        )

        measured_kspace = apply_undersampling_mask(full_kspace, mask)
        zero_filled_complex_image = ifft2c(measured_kspace)

        target_magnitude = normalize_to_unit_range(target_magnitude)
        target_magnitude = center_crop(target_magnitude, target_shape=(320, 320))
        target_magnitude = normalize_to_unit_range(target_magnitude).astype(np.float32)

        return {
            "input": torch.from_numpy(complex_to_channels(zero_filled_complex_image)),
            "target_complex": torch.from_numpy(complex_to_channels(full_complex_image)),
            "target_magnitude": torch.from_numpy(target_magnitude[None, ...]),
            "measured_kspace": torch.from_numpy(complex_to_channels(measured_kspace)),
            "mask": torch.from_numpy(mask.astype(bool)),
            "file_path": str(h5_path),
            "slice_index": int(slice_index),
        }
