import time
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========
# Temel tensör fonksiyonları
# ==========

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Basit matris çarpımı.
    a: (..., m, k)
    b: (..., k, n)
    return: (..., m, n)
    """
    return a @ b


def l2_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """
    L2 norm (vektör uzunluğu).
    """
    return torch.sqrt(torch.sum(x * x, dim=dim, keepdim=True) + eps)


def normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """
    Vektörleri L2 normuna göre normalize eder.
    """
    return x / (l2_norm(x, dim=dim, eps=eps) + eps)


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Nümerik stabil softmax.
    """
    shifted = x - x.max(dim=dim, keepdim=True).values
    exp = torch.exp(shifted)
    return exp / exp.sum(dim=dim, keepdim=True)


def transpose_last2(x: torch.Tensor) -> torch.Tensor:
    """
    Son iki boyutu transpoze eder.
    """
    return x.transpose(-2, -1)


# ==========
# Linear layer
# ==========

@dataclass
class LinearConfig:
    in_features: int
    out_features: int
    bias: bool = True


class Linear(nn.Module):
    """
    PyTorch nn.Linear ile aynı interface'e sahip basit linear katman.
    """

    def __init__(self, config: LinearConfig):
        super().__init__()
        self.in_features = config.in_features
        self.out_features = config.out_features

        self.weight = nn.Parameter(
            torch.empty(config.out_features, config.in_features)
        )
        if config.bias:
            self.bias = nn.Parameter(torch.empty(config.out_features))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features)
        y = x @ self.weight.t()
        if self.bias is not None:
            y = y + self.bias
        return y


# ==========
# Basit benchmark fonksiyonları
# ==========

def benchmark_matmul(
    m: int = 1024,
    k: int = 1024,
    n: int = 1024,
    iters: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[float, torch.Tensor]:
    """
    Matris çarpımı benchmark.
    """
    a = torch.randn(m, k, device=device)
    b = torch.randn(k, n, device=device)

    # warmup
    for _ in range(3):
        _ = a @ b

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    out = None
    for _ in range(iters):
        out = a @ b
    if device == "cuda":
        torch.cuda.synchronize()
    end = time.time()

    avg_ms = (end - start) * 1000.0 / iters
    return avg_ms, out


def benchmark_linear(
    batch_size: int = 4096,
    in_features: int = 1024,
    out_features: int = 4096,
    iters: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> float:
    """
    Linear katman benchmark.
    """
    cfg = LinearConfig(in_features=in_features, out_features=out_features)
    layer = Linear(cfg).to(device)
    x = torch.randn(batch_size, in_features, device=device)

    # warmup
    for _ in range(3):
        _ = layer(x)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        _ = layer(x)
    if device == "cuda":
        torch.cuda.synchronize()
    end = time.time()

    avg_ms = (end - start) * 1000.0 / iters
    return avg_ms


if __name__ == "__main__":
    # Küçük manuel test / demo
    print("Device:", "cuda" if torch.cuda.is_available() else "cpu")
    ms, _ = benchmark_matmul()
    print(f"Matmul avg: {ms:.3f} ms")
    ms_lin = benchmark_linear()
    print(f"Linear avg: {ms_lin:.3f} ms")
