"""SE(3) x S_N actions on nuclear coordinates."""

from __future__ import annotations

import torch


def center(R: torch.Tensor) -> torch.Tensor:
    return R - R.mean(dim=-2, keepdim=True)


def rotate(R: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    return R @ Q.T


def translate(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return R + t.reshape(*([1] * (R.ndim - 1)), 3)


def permute(R: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    return R.index_select(-2, order)


def random_so3(device=None, dtype=torch.float64) -> torch.Tensor:
    a = torch.randn(3, 3, device=device, dtype=dtype)
    q, _ = torch.linalg.qr(a)
    if torch.det(q) < 0:
        q = q.clone()
        q[:, 0] *= -1
    return q
