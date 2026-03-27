import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    """
    Ensures causality by removing extra padding.
    """

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()

        # For Conv1d with "same" intent in causal setup:
        padding = (kernel_size - 1) * dilation

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return torch.relu(y + res)


class TCNMultiTask(nn.Module):
    """
    Input: keypoints tensor shaped [B, T, J, C] or [B, T, F] where F=J*C.
    We flatten per-frame features then apply a TCN encoder over time.

    Outputs:
      - logits: [B, num_classes]
      - score: [B, 1] if regression enabled
    """

    def __init__(
        self,
        num_classes: int,
        num_frames: int,
        num_joints: int,
        num_channels: int,
        hidden_channels: int = 128,
        num_levels: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        enable_regression: bool = True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_frames = num_frames
        self.num_joints = num_joints
        self.num_channels = num_channels
        self.enable_regression = enable_regression

        in_channels = num_joints * num_channels

        blocks = []
        for level in range(num_levels):
            dilation = 2 ** level
            in_ch = in_channels if level == 0 else hidden_channels
            out_ch = hidden_channels
            blocks.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.tcn = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Linear(hidden_channels, num_classes)
        if enable_regression:
            self.regressor = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels // 2, 1),
            )
        else:
            self.regressor = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if x.dim() == 4:
            # [B, T, J, C] -> [B, F, T]
            x = x.reshape(x.size(0), x.size(1), -1).transpose(1, 2)
        elif x.dim() == 3:
            # [B, T, F] -> [B, F, T]
            x = x.transpose(1, 2)
        else:
            raise ValueError(f"Unexpected input shape: {tuple(x.shape)}")

        feat = self.tcn(x)  # [B, hidden, T]
        pooled = self.global_pool(feat).squeeze(-1)  # [B, hidden]
        logits = self.classifier(pooled)  # [B, num_classes]
        if self.enable_regression:
            score = self.regressor(pooled)  # [B, 1]
        else:
            score = None
        return logits, score

