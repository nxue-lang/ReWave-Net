from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from mri_recon.models.common import match_spatial_like


def haar_dwt2(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Apply an orthonormal 2D Haar transform to a feature map."""
    x00 = x[..., 0::2, 0::2]
    x01 = x[..., 0::2, 1::2]
    x10 = x[..., 1::2, 0::2]
    x11 = x[..., 1::2, 1::2]

    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 - x01 + x10 - x11) * 0.5
    hl = (x00 + x01 - x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return ll, lh, hl, hh


def haar_iwt2(
    ll: torch.Tensor,
    lh: torch.Tensor,
    hl: torch.Tensor,
    hh: torch.Tensor,
) -> torch.Tensor:
    """Invert an orthonormal 2D Haar transform."""
    output = torch.empty(
        ll.shape[0],
        ll.shape[1],
        ll.shape[-2] * 2,
        ll.shape[-1] * 2,
        dtype=ll.dtype,
        device=ll.device,
    )
    output[..., 0::2, 0::2] = (ll + lh + hl + hh) * 0.5
    output[..., 0::2, 1::2] = (ll - lh + hl - hh) * 0.5
    output[..., 1::2, 0::2] = (ll + lh - hl - hh) * 0.5
    output[..., 1::2, 1::2] = (ll - lh - hl + hh) * 0.5
    return output


class ResidualConditionedWaveletBlock(nn.Module):
    """Route Haar structure/detail features using measured band residuals."""

    def __init__(self, channels: int, condition_features: int = 4) -> None:
        super().__init__()
        self.channels = channels

        self.low_branch = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=7,
                padding=3,
                groups=channels,
            ),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
        )
        self.high_branch = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
            ),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        gate_hidden_features = max(channels // 2, 4)
        self.routing_gate = nn.Sequential(
            nn.Linear(channels * 2 + condition_features, gate_hidden_features),
            nn.GELU(),
            nn.Linear(gate_hidden_features, channels),
            nn.Sigmoid(),
        )
        self.output_projection = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
        )

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        original_height, original_width = x.shape[-2:]
        pad_bottom = original_height % 2
        pad_right = original_width % 2
        if pad_bottom or pad_right:
            x = F.pad(x, (0, pad_right, 0, pad_bottom), mode="replicate")

        ll, lh, hl, hh = haar_dwt2(x)
        low_features = self.low_branch(ll)
        high_bands = torch.cat([lh, hl, hh], dim=0)
        refined_high_bands = self.high_branch(high_bands)
        refined_lh, refined_hl, refined_hh = refined_high_bands.chunk(3, dim=0)

        low_summary = low_features.mean(dim=(-2, -1))
        high_summary = torch.cat(
            [refined_lh, refined_hl, refined_hh],
            dim=1,
        ).abs().mean(dim=(-2, -1))
        high_summary = high_summary.reshape(x.shape[0], 3, self.channels).mean(dim=1)
        gate_input = torch.cat([low_summary, high_summary, condition], dim=1)
        high_gate = self.routing_gate(gate_input)[:, :, None, None]

        routed_low = (1.0 - high_gate) * low_features
        routed_lh = high_gate * refined_lh
        routed_hl = high_gate * refined_hl
        routed_hh = high_gate * refined_hh
        routed = haar_iwt2(routed_low, routed_lh, routed_hl, routed_hh)
        routed = routed[..., :original_height, :original_width]
        residual = x[..., :original_height, :original_width]
        return residual + self.output_projection(routed)


class ConditionedConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.GELU(),
        )
        self.wavelet_block = ResidualConditionedWaveletBlock(out_channels)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.wavelet_block(self.conv(x), condition)


class ResidualConditionedWaveletComplexUNet(nn.Module):
    """Complex U-Net with measurement-residual-conditioned wavelet routing."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        self.enc1 = ConditionedConvBlock(in_channels, base_channels)
        self.enc2 = ConditionedConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConditionedConvBlock(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.bottleneck = ConditionedConvBlock(base_channels * 4, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2)
        self.dec3 = ConditionedConvBlock(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.dec2 = ConditionedConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.dec1 = ConditionedConvBlock(base_channels * 2, base_channels)
        self.output_layer = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(
        self,
        x: torch.Tensor,
        band_residuals: torch.Tensor,
        cascade_progress: torch.Tensor,
    ) -> torch.Tensor:
        condition = torch.cat([band_residuals, cascade_progress], dim=1)

        enc1 = self.enc1(x, condition)
        enc2 = self.enc2(self.pool(enc1), condition)
        enc3 = self.enc3(self.pool(enc2), condition)
        bottleneck = self.bottleneck(self.pool(enc3), condition)

        dec3 = self.up3(bottleneck)
        dec3 = self.dec3(
            torch.cat([dec3, match_spatial_like(enc3, dec3)], dim=1),
            condition,
        )
        dec2 = self.up2(dec3)
        dec2 = self.dec2(
            torch.cat([dec2, match_spatial_like(enc2, dec2)], dim=1),
            condition,
        )
        dec1 = self.up1(dec2)
        dec1 = self.dec1(
            torch.cat([dec1, match_spatial_like(enc1, dec1)], dim=1),
            condition,
        )

        correction = match_spatial_like(self.output_layer(dec1), x)
        return x + correction
