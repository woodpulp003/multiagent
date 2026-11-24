"""Conditional denoiser p_theta(x_{r-1} | x_r, r, H_t) (note, section 2).

SCAFFOLD: the architecture below is a minimal placeholder that satisfies the
I/O contract so the train -> export -> bot pipeline runs end to end. Replace it
with the real model (e.g. residual CNN or small transformer over 64 squares).

Contract (must match submission/belief.py DiffusionBelief):
  context float32[B, CONTEXT_PLANES, 8, 8]   features.encode_context
  tokens  int64[B, 64]                       classes 0..6, MASK, or VISIBLE
  t       float32[B]                         masking level in (0, 1]
  -> logits float32[B, 64, NUM_CLASSES]
"""
from __future__ import annotations

import torch
from torch import nn

from . import SUBMISSION  # noqa: F401  (puts submission/ on sys.path)
import features


class Denoiser(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 4):
        super().__init__()
        self.config = {"channels": channels, "blocks": blocks}
        self.token_embedding = nn.Embedding(features.NUM_TOKENS, channels)
        self.context_proj = nn.Conv2d(features.CONTEXT_PLANES, channels, 1)
        self.time_proj = nn.Sequential(nn.Linear(1, channels), nn.SiLU(), nn.Linear(channels, channels))
        # TODO: real architecture. Placeholder: plain 3x3 conv stack.
        self.body = nn.Sequential(*[
            nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU())
            for _ in range(blocks)])
        self.head = nn.Conv2d(channels, features.NUM_CLASSES, 1)

    def forward(self, context: torch.Tensor, tokens: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch = tokens.shape[0]
        x = self.token_embedding(tokens).transpose(1, 2).reshape(batch, -1, 8, 8)
        x = x + self.context_proj(context) + self.time_proj(t[:, None])[:, :, None, None]
        x = self.head(self.body(x))
        return x.flatten(2).transpose(1, 2)
