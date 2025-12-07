import torch

from src.foundations import (
    matmul,
    l2_norm,
    normalize,
    softmax,
    transpose_last2,
    Linear,
    LinearConfig,
    benchmark_matmul,
    benchmark_linear,
)


def test_matmul_shape():
    a = torch.randn(2, 3)
    b = torch.randn(3, 4)
    out = matmul(a, b)
    assert out.shape == (2, 4)


def test_l2_norm_and_normalize():
    x = torch.randn(5, 10)
    n = l2_norm(x, dim=-1)
    assert n.shape == (5, 1)
    x_norm = normalize(x, dim=-1)
    # Her vektörün normu yaklaşık 1 olmalı
    norms = torch.linalg.vector_norm(x_norm, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_softmax():
    x = torch.randn(3, 4)
    s = softmax(x, dim=-1)
    assert s.shape == (3, 4)
    sums = s.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    assert torch.all(s >= 0)


def test_transpose_last2():
    x = torch.randn(2, 3, 4)
    y = transpose_last2(x)
    assert y.shape == (2, 4, 3)


def test_linear_forward_shape():
    cfg = LinearConfig(in_features=16, out_features=32)
    layer = Linear(cfg)
    x = torch.randn(8, 16)
    y = layer(x)
    assert y.shape == (8, 32)


def test_benchmarks_run():
    ms, out = benchmark_matmul(m=64, k=64, n=64, iters=3, device="cpu")
    assert out.shape == (64, 64)
    assert ms > 0

    ms_lin = benchmark_linear(
        batch_size=128, in_features=32, out_features=64, iters=3, device="cpu"
    )
    assert ms_lin > 0
