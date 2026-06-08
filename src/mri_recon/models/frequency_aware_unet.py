from __future__ import annotations

import torch
from torch import nn

from mri_recon.models.common import match_spatial_like


class FrequencyAwareBlock(nn.Module):
    """Frequency-aware feature fusion block.

    It separates feature processing into:
    - low-frequency branch: smooth/global anatomical structures
    - high-frequency branch: edge/detail residuals
    - gate branch: adaptive fusion between low/high-frequency features
    """

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

        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
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

        gate = self.gate(torch.cat([low_features, high_features], dim=1))
        fused = gate * high_features + (1.0 - gate) * low_features
        fused = self.output_projection(fused)

        return x + fused


class ConvBlock(nn.Module):
    """Convolution block with optional frequency-aware refinement."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_frequency_block: bool = True,
    ) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.GELU(),
        )

        if use_frequency_block:
            self.frequency_block = FrequencyAwareBlock(out_channels)
        else:
            self.frequency_block = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.frequency_block(x)
        return x


class FrequencyAwareComplexUNet(nn.Module):
    """Frequency-aware full-resolution complex U-Net.

    Input:
        [batch, 2, height, width]

    Output:
        [batch, 2, height, width]

    The two channels represent real and imaginary parts of the complex image.
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
