from __future__ import annotations

import torch
from torch import nn

from mri_recon.models.common import match_spatial_like


class KANStyleGate(nn.Module):
    """Lightweight KAN-style channel gate.

    This is not a full Kolmogorov-Arnold Network implementation.
    It is a lightweight nonlinear gate inspired by the KAN idea of using
    learnable nonlinear basis functions.

    Input:
        [B, 2C]

    Output:
        [B, C, 1, 1]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int | None = None,
    ) -> None:
        super().__init__()

        if hidden_features is None:
            hidden_features = max(out_features, 16)

        self.linear_base = nn.Linear(in_features, hidden_features)

        self.spline_like_branch = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
        )

        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, out_features),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear_base(x)
        nonlinear = self.spline_like_branch(x)
        gate = self.output_layer(base + nonlinear)

        return gate[:, :, None, None]


class KANFrequencyAwareBlock(nn.Module):
    """Frequency-aware block with KAN-style nonlinear low/high fusion gate."""

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.local_projection = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.GELU(),
        )

        self.low_frequency_branch = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=1, padding=2),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.InstanceNorm2d(channels),
            nn.GELU(),
        )

        self.high_frequency_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.InstanceNorm2d(channels),
            nn.GELU(),
        )

        self.channel_gate = KANStyleGate(
            in_features=channels * 2,
            out_features=channels,
            hidden_features=channels,
        )

        self.output_projection = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.InstanceNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_features = self.local_projection(x)

        low_features = self.low_frequency_branch(local_features)
        high_input = local_features - low_features
        high_features = self.high_frequency_branch(high_input)

        low_summary = low_features.mean(dim=(-2, -1))
        high_summary = high_features.mean(dim=(-2, -1))
        gate_input = torch.cat([low_summary, high_summary], dim=1)

        gate = self.channel_gate(gate_input)

        fused = gate * high_features + (1.0 - gate) * low_features
        fused = self.output_projection(fused)

        return x + fused


class ConvBlock(nn.Module):
    """Convolution block with KAN-gated frequency-aware refinement."""

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

        self.frequency_block = KANFrequencyAwareBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.frequency_block(x)
        return x


class KANFrequencyAwareComplexUNet(nn.Module):
    """KAN-gated frequency-aware full-resolution complex U-Net.

    Input:
        [B, 2, H, W]

    Output:
        [B, 2, H, W]
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_channels: int = 16,
    ) -> None:
        super().__init__()

        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(
            base_channels * 8,
            base_channels * 4,
            kernel_size=2,
            stride=2,
        )
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=2,
            stride=2,
        )
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            kernel_size=2,
            stride=2,
        )
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        self.output_layer = nn.Conv2d(
            base_channels,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))

        bottleneck = self.bottleneck(self.pool(enc3))

        dec3 = self.up3(bottleneck)
        enc3 = match_spatial_like(enc3, dec3)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.up2(dec3)
        enc2 = match_spatial_like(enc2, dec2)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.up1(dec2)
        enc1 = match_spatial_like(enc1, dec1)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        correction = self.output_layer(dec1)
        correction = match_spatial_like(correction, x)

        return x + correction
