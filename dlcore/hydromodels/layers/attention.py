"""Deterministic ProbSparse attention."""

from __future__ import annotations

import math

import torch
from torch import nn


class ProbSparseSelfAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, factor: int, dropout: float):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError("d_model must be divisible by heads")
        self.heads = int(heads)
        self.head_size = d_model // heads
        self.factor = int(factor)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, values: torch.Tensor) -> torch.Tensor:
        batch, length, channels = values.shape
        return values.reshape(batch, length, self.heads, channels // self.heads).transpose(1, 2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        queries = self.split_heads(self.query(values))
        keys = self.split_heads(self.key(values))
        payload = self.split_heads(self.value(values))
        length = values.shape[1]
        selected_count = min(length, max(1, self.factor * int(math.ceil(math.log(length + 1)))))
        sample_index = torch.linspace(
            0, length - 1, selected_count, device=values.device
        ).round().long()
        sampled_keys = keys[:, :, sample_index]
        sampled_scores = torch.matmul(queries, sampled_keys.transpose(-2, -1))
        sparsity = sampled_scores.max(dim=-1).values - sampled_scores.mean(dim=-1)
        query_index = torch.topk(sparsity, selected_count, dim=-1, sorted=False).indices
        gather_index = query_index.unsqueeze(-1).expand(-1, -1, -1, self.head_size)
        selected_queries = torch.gather(queries, 2, gather_index)
        scores = torch.matmul(selected_queries, keys.transpose(-2, -1)) / math.sqrt(self.head_size)
        attention = self.dropout(torch.softmax(scores, dim=-1))
        updates = torch.matmul(attention, payload)
        context = payload.mean(dim=2, keepdim=True).expand(-1, -1, length, -1).clone()
        context = context.scatter(2, gather_index, updates)
        merged = context.transpose(1, 2).reshape(values.shape[0], length, -1)
        return self.output(merged)
