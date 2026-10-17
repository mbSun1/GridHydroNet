"""Multi-resolution pyramid construction for Pyraformer."""

from __future__ import annotations

import torch
from torch import nn


class PyramidEncoder(nn.Module):
    def __init__(self, d_model: int, levels: int = 2, window_size: int = 4):
        super().__init__()
        self.downsamplers = nn.ModuleList(
            [
                nn.Conv1d(
                    d_model,
                    d_model,
                    kernel_size=window_size,
                    stride=window_size,
                    groups=d_model,
                )
                for _ in range(levels)
            ]
        )
        self.scale_embeddings = nn.Embedding(levels + 1, d_model)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        sequences = [values + self.scale_embeddings.weight[0].view(1, 1, -1)]
        current = values.transpose(1, 2)
        for index, downsampler in enumerate(self.downsamplers, start=1):
            if current.shape[-1] < downsampler.kernel_size[0]:
                break
            current = downsampler(current)
            level = current.transpose(1, 2)
            level = level + self.scale_embeddings.weight[index].view(1, 1, -1)
            sequences.append(level)
        return torch.cat(sequences, dim=1)
