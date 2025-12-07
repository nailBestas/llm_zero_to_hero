from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadSelfAttention


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.0,
        is_causal: bool = True,
    ):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout=dropout, is_causal=is_causal)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim, dropout=dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention + residual + LayerNorm
        attn_out = self.attn(self.ln1(x), mask=mask)
        x = x + self.dropout(attn_out)

        # FFN + residual + LayerNorm
        ff_out = self.ff(self.ln2(x))
        x = x + self.dropout(ff_out)
        return x
