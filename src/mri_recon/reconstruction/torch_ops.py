from __future__ import annotations

import torch


def channels_to_complex_torch(x: torch.Tensor) -> torch.Tensor:
    """Convert [B, 2, H, W] real/imag channels to [B, H, W] complex tensor."""
    if x.ndim != 4 or x.shape[1] != 2:
        raise ValueError(f"Expected [B, 2, H, W] tensor, got shape {tuple(x.shape)}")

    return torch.complex(x[:, 0], x[:, 1])


def complex_to_channels_torch(x: torch.Tensor) -> torch.Tensor:
    """Convert [B, H, W] complex tensor to [B, 2, H, W] real/imag channels."""
    if x.ndim != 3:
        raise ValueError(
            f"Expected [B, H, W] complex tensor, got shape {tuple(x.shape)}"
        )

    return torch.stack([x.real, x.imag], dim=1)


def fft2c_torch(image: torch.Tensor) -> torch.Tensor:
    """Centered orthonormal 2D FFT for complex tensors with shape [B, H, W]."""
    image = torch.fft.ifftshift(image, dim=(-2, -1))
    kspace = torch.fft.fft2(image, norm="ortho")
    return torch.fft.fftshift(kspace, dim=(-2, -1))


def ifft2c_torch(kspace: torch.Tensor) -> torch.Tensor:
    """Centered orthonormal 2D inverse FFT for complex tensors with shape [B, H, W]."""
    kspace = torch.fft.ifftshift(kspace, dim=(-2, -1))
    image = torch.fft.ifft2(kspace, norm="ortho")
    return torch.fft.fftshift(image, dim=(-2, -1))


def expand_mask_for_torch_kspace(
    mask: torch.Tensor, kspace: torch.Tensor
) -> torch.Tensor:
    """Expand [W] or [B, W] mask to [B, H, W] for single-coil k-space tensors."""
    if kspace.ndim != 3:
        raise ValueError(f"Expected [B, H, W] k-space, got shape {tuple(kspace.shape)}")

    if mask.ndim == 1:
        mask = mask[None, :]
    elif mask.ndim != 2:
        raise ValueError(f"Expected [W] or [B, W] mask, got shape {tuple(mask.shape)}")

    if mask.shape[-1] != kspace.shape[-1]:
        raise ValueError(
            f"Mask width {mask.shape[-1]} does not match "
            f"k-space width {kspace.shape[-1]}"
        )

    if mask.shape[0] == 1 and kspace.shape[0] != 1:
        mask = mask.expand(kspace.shape[0], -1)
    elif mask.shape[0] != kspace.shape[0]:
        raise ValueError(
            f"Mask batch {mask.shape[0]} does not match k-space batch {kspace.shape[0]}"
        )

    return mask.bool()[:, None, :].expand(-1, kspace.shape[-2], -1)


def apply_soft_data_consistency_torch(
    predicted_channels: torch.Tensor,
    measured_kspace_channels: torch.Tensor,
    mask: torch.Tensor,
    dc_weight: torch.Tensor | float,
) -> torch.Tensor:
    """Apply differentiable soft k-space data consistency.

    `dc_weight=0` keeps the predicted k-space unchanged, while `dc_weight=1`
    replaces sampled k-space locations with the measured values.
    """
    predicted_image = channels_to_complex_torch(predicted_channels)
    measured_kspace = channels_to_complex_torch(measured_kspace_channels)
    predicted_kspace = fft2c_torch(predicted_image)

    expanded_mask = expand_mask_for_torch_kspace(mask=mask, kspace=predicted_kspace)
    dc_weight = torch.as_tensor(
        dc_weight,
        dtype=predicted_kspace.real.dtype,
        device=predicted_kspace.device,
    )

    corrected_kspace = predicted_kspace + dc_weight * expanded_mask * (
        measured_kspace - predicted_kspace
    )
    corrected_image = ifft2c_torch(corrected_kspace)

    return complex_to_channels_torch(corrected_image)


def apply_hard_data_consistency_torch(
    predicted_channels: torch.Tensor,
    measured_kspace_channels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply hard k-space data consistency at sampled locations."""
    return apply_soft_data_consistency_torch(
        predicted_channels=predicted_channels,
        measured_kspace_channels=measured_kspace_channels,
        mask=mask,
        dc_weight=1.0,
    )
