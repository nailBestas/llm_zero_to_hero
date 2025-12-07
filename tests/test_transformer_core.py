import torch

from src.transformer_core.attention import scaled_dot_product_attention, MultiHeadSelfAttention
from src.transformer_core.layers import TransformerBlock
from src.transformer_core.model import MiniTransformerLM


def test_scaled_dot_product_attention_shapes():
    b, h, s, d = 2, 4, 8, 16
    q = torch.randn(b, h, s, d)
    k = torch.randn(b, h, s, d)
    v = torch.randn(b, h, s, d)

    out, attn = scaled_dot_product_attention(q, k, v)
    assert out.shape == (b, h, s, d)
    assert attn.shape == (b, h, s, s)


def test_multihead_self_attention_shapes():
    b, s, e, h = 2, 8, 32, 4
    x = torch.randn(b, s, e)
    attn = MultiHeadSelfAttention(embed_dim=e, num_heads=h)
    out = attn(x)
    assert out.shape == (b, s, e)


def test_transformer_block_shapes():
    b, s, e, h, ff = 2, 8, 32, 4, 64
    x = torch.randn(b, s, e)
    block = TransformerBlock(embed_dim=e, num_heads=h, ff_dim=ff)
    out = block(x)
    assert out.shape == (b, s, e)


def test_mini_transformer_lm_forward():
    b, s, vocab = 2, 8, 100
    model = MiniTransformerLM(vocab_size=vocab, embed_dim=32, num_heads=4, ff_dim=64, num_layers=2, max_seq_len=16)
    input_ids = torch.randint(0, vocab, (b, s))
    logits = model(input_ids)
    assert logits.shape == (b, s, vocab)
