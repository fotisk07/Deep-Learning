from torch import nn


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.name = "CNN"
        # -------- Feature extractor --------
        self.features = nn.Sequential(
            # Block 1: 32x32
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16
            # Block 2: 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8
            # Block 3: 8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 4x4
        )

        # -------- Classifier --------
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 100),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# src/models/vision/vit_clip.py
# Minimal ViT vision encoder suitable for CLIP-style dual-encoder training.

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuickGELU(nn.Module):
    # CLIP uses a "QuickGELU"-like activation in some implementations.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(
            embed_dim=width,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ln_2 = nn.LayerNorm(width)
        hidden = int(width * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden),
            QuickGELU(),
            nn.Linear(hidden, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, width)
        a = self.ln_1(x)
        attn_out, _ = self.attn(a, a, a, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x


class VisionTransformerCLIP(nn.Module):
    """
    CLIP-style ViT image encoder:
    - patch embedding via Conv2d
    - [CLS] token
    - learnable positional embedding
    - Transformer encoder blocks
    - final LayerNorm + linear projection to embed_dim

    Output: image embedding (B, embed_dim), L2-normalized by default.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        width: int = 768,
        layers: int = 12,
        heads: int = 12,
        embed_dim: int = 512,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        normalize: bool = True,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size ({image_size}) must be divisible by patch_size ({patch_size}).")

        self.image_size = image_size
        self.patch_size = patch_size
        self.width = width
        self.layers = layers
        self.heads = heads
        self.embed_dim = embed_dim
        self.normalize = normalize

        grid = image_size // patch_size
        num_patches = grid * grid
        self.num_tokens = 1 + num_patches  # cls + patches

        # Patchify + linear embed (Conv does both)
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)

        self.class_embedding = nn.Parameter(torch.randn(width) * (width**-0.5))
        self.positional_embedding = nn.Parameter(torch.randn(self.num_tokens, width) * (width**-0.5))

        self.ln_pre = nn.LayerNorm(width)
        self.transformer = nn.ModuleList(
            [
                ResidualAttentionBlock(width, heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(layers)
            ]
        )
        self.ln_post = nn.LayerNorm(width)

        # CLIP: final projection into the shared embedding space
        self.proj = nn.Parameter(torch.randn(width, embed_dim) * (width**-0.5))

        self._init_parameters()

    def _init_parameters(self) -> None:
        # Reasonable defaults; keep it simple and stable.
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.class_embedding, std=0.01)
        # conv1 weight is already Kaiming-ish by default; you may override if you want.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W). Ideally H=W=image_size. If different, you should resize before.
        """
        B, C, H, W = x.shape
        if H != self.image_size or W != self.image_size:
            # keep it explicit: CLIP typically uses a fixed resolution
            raise ValueError(f"Expected input size {(self.image_size, self.image_size)}, got {(H, W)}")

        # Patch embeddings: (B, width, grid, grid)
        x = self.conv1(x)
        # Flatten patches: (B, width, num_patches) -> (B, num_patches, width)
        x = x.flatten(2).transpose(1, 2)

        # Prepend CLS: (B, 1, width)
        cls = self.class_embedding.unsqueeze(0).unsqueeze(1).expand(B, 1, self.width)
        x = torch.cat([cls, x], dim=1)  # (B, 1+num_patches, width)

        # Add pos emb
        x = x + self.positional_embedding.unsqueeze(0)  # (B, T, width)
        x = self.ln_pre(x)

        # Transformer
        for block in self.transformer:
            x = block(x)

        # Take CLS token
        x = self.ln_post(x[:, 0, :])  # (B, width)

        # Project to CLIP embedding space
        x = x @ self.proj  # (B, embed_dim)

        if self.normalize:
            x = F.normalize(x, dim=-1)

        return x
