from __future__ import annotations

from _bootstrap import add_project_src_to_path

add_project_src_to_path()

import torch

from mri_recon.models.residual_conditioned_wavelet_unet import haar_dwt2, haar_iwt2
from mri_recon.models.unrolled_frequency_aware import (
    UnrolledResidualConditionedWaveletRecon,
)
from mri_recon.reconstruction.torch_ops import normalized_band_residuals_torch


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wavelet_input = torch.randn(2, 4, 32, 34, device=device)
    reconstructed = haar_iwt2(*haar_dwt2(wavelet_input))
    wavelet_error = float((wavelet_input - reconstructed).abs().max().item())
    if wavelet_error > 1e-5:
        raise AssertionError(f"Haar reconstruction error too large: {wavelet_error}")

    batch_size, height, width = 2, 64, 64
    image = torch.randn(batch_size, 2, height, width, device=device)
    measured_kspace = torch.randn(batch_size, 2, height, width, device=device)
    mask = torch.zeros(batch_size, width, dtype=torch.bool, device=device)
    mask[:, ::4] = True
    mask[:, width // 2 - 3 : width // 2 + 3] = True

    band_residuals = normalized_band_residuals_torch(
        predicted_channels=image,
        measured_kspace_channels=measured_kspace,
        mask=mask,
    )
    if band_residuals.shape != (batch_size, 3):
        raise AssertionError(f"Unexpected residual shape: {band_residuals.shape}")
    if not torch.isfinite(band_residuals).all():
        raise AssertionError("Band residuals contain non-finite values")

    model = UnrolledResidualConditionedWaveletRecon(
        num_cascades=2,
        base_channels=4,
    ).to(device)
    prediction = model(image=image, measured_kspace=measured_kspace, mask=mask)
    if prediction.shape != image.shape:
        raise AssertionError(f"Unexpected prediction shape: {prediction.shape}")

    prediction.abs().mean().backward()
    if not all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    ):
        raise AssertionError("Model gradients contain non-finite values")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Device: {device}")
    print(f"Max Haar reconstruction error: {wavelet_error:.3e}")
    print(f"Band residual shape: {tuple(band_residuals.shape)}")
    print(f"Prediction shape: {tuple(prediction.shape)}")
    print(f"Parameters: {parameter_count:,}")
    print("Residual-conditioned wavelet model smoke test passed.")


if __name__ == "__main__":
    main()
