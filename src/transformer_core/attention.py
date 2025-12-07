import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    q, k, v: (batch, heads, seq, head_dim)
    mask:   (batch, 1, seq, seq) veya (batch, heads, seq, seq), True olan yerler -inf yapılır.
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # (b, h, s, s)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)

    out = torch.matmul(attn, v)  # (b, h, s, head_dim)
    return out, attn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, is_causal: bool = False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim num_heads'e bölünebilmeli"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.is_causal = is_causal

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)

    def _shape(self, x: torch.Tensor, bsz: int, seq_len: int) -> torch.Tensor:
        # (b, seq, embed) -> (b, heads, seq, head_dim)
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (batch, seq, embed_dim)
        mask: (batch, 1, seq, seq) veya (batch, heads, seq, seq) boolean
        """
        bsz, seq_len, _ = x.size()
        qkv = self.qkv_proj(x)  # (b, seq, 3*embed)
        q, k, v = qkv.chunk(3, dim=-1)

        q = self._shape(q, bsz, seq_len)
        k = self._shape(k, bsz, seq_len)
        v = self._shape(v, bsz, seq_len)

        if self.is_causal and mask is None:
            # Causal mask: üst üçgen True
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1
            )
            mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, s, s)

        out, attn = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.attn_dropout
        )  # (b, h, s, d)

        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
        out = self.out_proj(out)
        return out

class FastMultiHeadSelfAttention(nn.Module):
    """
    PyTorch'un optimize scaled_dot_product_attention fonksiyonunu kullanan hızlı self-attention.
    Sadece GPU ve belirli dtype'larda en iyi performansı verir.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, is_causal: bool = False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim num_heads'e bölünebilmeli"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.is_causal = is_causal

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_p = dropout

    def _shape(self, x: torch.Tensor, bsz: int, seq_len: int) -> torch.Tensor:
        # (b, seq, embed) -> (b, heads, seq, head_dim)
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (batch, seq, embed_dim)
        attn_mask: PyTorch'un scaled_dot_product_attention ile uyumlu mask
        """
        bsz, seq_len, _ = x.size()
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = self._shape(q, bsz, seq_len)
        k = self._shape(k, bsz, seq_len)
        v = self._shape(v, bsz, seq_len)

        # PyTorch 2.x'in F.scaled_dot_product_attention'ı, causal mask'i argüman olarak alabiliyor
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=self.is_causal and attn_mask is None,
        )  # (b, h, s, d)

        out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
        out = self.out_proj(out)
        return out




