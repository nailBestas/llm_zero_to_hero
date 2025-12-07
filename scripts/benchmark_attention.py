import time

import torch

from src.transformer_core.attention import MultiHeadSelfAttention, FastMultiHeadSelfAttention


def benchmark_module(module, bsz=4, seq_len=512, embed_dim=512, iters=20, device="cuda"):
    x = torch.randn(bsz, seq_len, embed_dim, device=device)

    # warmup
    for _ in range(5):
        _ = module(x)
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        _ = module(x)
    if device == "cuda":
        torch.cuda.synchronize()
    end = time.time()

    avg_ms = (end - start) * 1000.0 / iters
    return avg_ms


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    embed_dim = 512
    num_heads = 8
    bsz = 4
    seq_len = 512
    iters = 20

    slow_attn = MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.0, is_causal=True).to(device)
    fast_attn = FastMultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.0, is_causal=True).to(device)

    slow_ms = benchmark_module(slow_attn, bsz=bsz, seq_len=seq_len, embed_dim=embed_dim, iters=iters, device=device)
    fast_ms = benchmark_module(fast_attn, bsz=bsz, seq_len=seq_len, embed_dim=embed_dim, iters=iters, device=device)

    print(f"Slow attention: {slow_ms:.3f} ms")
    print(f"Fast attention: {fast_ms:.3f} ms")
    if fast_ms > 0:
        speedup = (slow_ms - fast_ms) / fast_ms * 100.0
        print(f"Speedup: {speedup:.2f} %")


if __name__ == "__main__":
    main()
