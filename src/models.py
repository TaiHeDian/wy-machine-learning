from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig


class ConfigurableCNN(nn.Module):
    def __init__(self, num_classes: int, config: ModelConfig) -> None:
        super().__init__()
        if not (len(config.channels) == len(config.kernel_sizes) == len(config.pool_sizes)):
            raise RuntimeError("model.channels, model.kernel_sizes, and model.pool_sizes must have the same length.")

        blocks: list[nn.Module] = []
        in_channels = 1
        for out_channels, kernel_size, pool_size in zip(config.channels, config.kernel_sizes, config.pool_sizes):
            blocks.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2))
            if config.use_batch_norm:
                blocks.append(nn.BatchNorm1d(out_channels))
            blocks.append(nn.ReLU(inplace=True))
            if pool_size > 1:
                blocks.append(nn.MaxPool1d(pool_size))
            in_channels = out_channels

        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool1d(config.adaptive_pool_size))
        flattened_dim = config.channels[-1] * config.adaptive_pool_size
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, config.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.dropout),
        )
        self.classifier = nn.Linear(config.embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(x)
        embedding = self.embedding(features)
        logits = self.classifier(embedding)
        return logits, embedding
