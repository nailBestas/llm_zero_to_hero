from typing import Optional

import torch
import torch.nn as nn

from .layers import TransformerBlock


class MiniTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_heads: int = 4,
        ff_dim: int = 1024,
        num_layers: int = 4,
        max_seq_len: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    is_causal=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (batch, seq)
        return: logits (batch, seq, vocab_size)
        """
        bsz, seq_len = input_ids.size()
        assert seq_len <= self.max_seq_len, "seq_len max_seq_len'den büyük olamaz"

        device = input_ids.device
        positions = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq)

        x = self.token_embedding(input_ids) + self.pos_embedding(positions)  # (b, seq, embed)

        # Causal mask: (1, 1, seq, seq)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            x = layer(x, mask=causal_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
