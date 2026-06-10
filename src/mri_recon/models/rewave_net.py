from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import nn

from mri_recon.models.complex_unet import ComplexUNet
from mri_recon.models.residual_conditioned_wavelet_unet import (
    ResidualConditionedWaveletComplexUNet,
)
from mri_recon.reconstruction.torch_ops import (
    apply_soft_data_consistency_torch,
    normalized_band_residuals_torch,
)


def _initial_dc_logit(initial_dc_weight: float) -> float:
    if not 0.0 < initial_dc_weight < 1.0:
        raise ValueError(
            f"initial_dc_weight must be in (0, 1), got {initial_dc_weight}"
        )

    return math.log(initial_dc_weight / (1.0 - initial_dc_weight))


class _UnrolledComplexReconBase(nn.Module):
    """Unrolled single-coil complex MRI reconstruction with learned DC weights."""

    def __init__(
        self,
        denoiser_factory: Callable[[], nn.Module],
        num_cascades: int = 5,
        shared_denoiser: bool = True,
        initial_dc_weight: float = 0.1,
        residual_conditioned: bool = False,
    ) -> None:
        super().__init__()

        if num_cascades < 1:
            raise ValueError(f"num_cascades must be >= 1, got {num_cascades}")

        self.num_cascades = num_cascades
        self.shared_denoiser = shared_denoiser
        self.residual_conditioned = residual_conditioned

        num_denoisers = 1 if shared_denoiser else num_cascades
        self.denoisers = nn.ModuleList(
            [denoiser_factory() for _ in range(num_denoisers)]
        )

        initial_logit = _initial_dc_logit(initial_dc_weight)
        self.dc_weight_logits = nn.Parameter(
            torch.full((num_cascades,), float(initial_logit))
        )

    @property
    def dc_weights(self) -> torch.Tensor:
        """Per-cascade soft data-consistency weights in [0, 1]."""
        return torch.sigmoid(self.dc_weight_logits)

    def _denoiser_for_cascade(self, cascade_index: int) -> nn.Module:
        if self.shared_denoiser:
            return self.denoisers[0]

        return self.denoisers[cascade_index]

    def forward(
        self,
        image: torch.Tensor,
        measured_kspace: torch.Tensor,
        mask: torch.Tensor,
        return_intermediates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Run learned regularization and soft DC for each cascade.

        Args:
            image: Zero-filled complex image as [B, 2, H, W].
            measured_kspace: Undersampled measured k-space as [B, 2, H, W].
            mask: Sampling mask as [B, W] or [W].
            return_intermediates: Whether to return cascade outputs.
        """
        current = image
        intermediates = []

        for cascade_index in range(self.num_cascades):
            denoiser = self._denoiser_for_cascade(cascade_index)
            if self.residual_conditioned:
                band_residuals = normalized_band_residuals_torch(
                    predicted_channels=current,
                    measured_kspace_channels=measured_kspace,
                    mask=mask,
                )
                cascade_progress = current.new_full(
                    (current.shape[0], 1),
                    cascade_index / max(self.num_cascades - 1, 1),
                )
                current = denoiser(current, band_residuals, cascade_progress)
            else:
                current = denoiser(current)
            current = apply_soft_data_consistency_torch(
                predicted_channels=current,
                measured_kspace_channels=measured_kspace,
                mask=mask,
                dc_weight=self.dc_weights[cascade_index],
            )

            if return_intermediates:
                intermediates.append(current)

        if return_intermediates:
            return current, intermediates

        return current


class UnrolledComplexUNetRecon(_UnrolledComplexReconBase):
    """Unrolled reconstruction using ComplexUNet as the regularizer baseline."""

    def __init__(
        self,
        num_cascades: int = 5,
        base_channels: int = 16,
        shared_denoiser: bool = True,
        initial_dc_weight: float = 0.1,
    ) -> None:
        super().__init__(
            denoiser_factory=lambda: ComplexUNet(base_channels=base_channels),
            num_cascades=num_cascades,
            shared_denoiser=shared_denoiser,
            initial_dc_weight=initial_dc_weight,
        )


class ReWaveNet(_UnrolledComplexReconBase):
    """Unrolled reconstruction with measured-residual-conditioned wavelet routing."""

    def __init__(
        self,
        num_cascades: int = 5,
        base_channels: int = 16,
        shared_denoiser: bool = True,
        initial_dc_weight: float = 0.1,
    ) -> None:
        super().__init__(
            denoiser_factory=lambda: ResidualConditionedWaveletComplexUNet(
                base_channels=base_channels
            ),
            num_cascades=num_cascades,
            shared_denoiser=shared_denoiser,
            initial_dc_weight=initial_dc_weight,
            residual_conditioned=True,
        )
