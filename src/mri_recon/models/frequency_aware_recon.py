from __future__ import annotations

import torch
from torch import nn


class FrequencyAwareBlock(nn.Module):
    """Frequency-aware feature mixing block.

    The block separates feature processing into:
    - local branch: standard local convolution
    - low-frequency branch: smoothed/global anatomical structure
    - high-frequency branch: residual detail and edge information
    - gate branch: adaptive fusion between low/high-frequency features
    """

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.local_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.GELU(),
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
        local_features = self.local_branch(x)

        low_features = self.low_frequency_branch(local_features)

        # High-frequency details are modeled as residual detail after removing
        # the smoothed low-frequency component.
        high_input = local_features - low_features
        high_features = self.high_frequency_branch(high_input)

        fusion_gate = self.gate(torch.cat([low_features, high_features], dim=1))

        fused_features = (
            fusion_gate * high_features + (1.0 - fusion_gate) * low_features
        )
        fused_features = self.output_projection(fused_features)

        return x + fused_features


class FrequencyAwareComplexRecon(nn.Module):
    """Frequency-aware complex reconstruction network.

    This model operates on full-resolution complex-valued MRI images represented
    as two real channels.

    Input:
        [batch, 2, height, width]

    Output:
        [batch, 2, height, width]
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        feature_channels: int = 32,
        num_blocks: int = 4,
    ) -> None:
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(feature_channels),
            nn.GELU(),
        )

        self.blocks = nn.Sequential(
            *[FrequencyAwareBlock(feature_channels) for _ in range(num_blocks)]
        )

        self.output_projection = nn.Conv2d(
            feature_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.input_projection(x)
        features = self.blocks(features)
        correction = self.output_projection(features)

        # Residual complex-image reconstruction.
        return x + correction
