from __future__ import annotations

import torch
import torch.nn.functional as F


def randn(shape, dtype):
    return torch.randn(shape, device="npu", dtype=dtype)


def rand(shape, dtype):
    return torch.rand(shape, device="npu", dtype=dtype)


def beta(shape, dtype):
    return torch.randn(shape, device="npu", dtype=dtype).sigmoid()


def logsigmoid(shape, dtype=torch.float32):
    return F.logsigmoid(torch.randn(shape, device="npu", dtype=dtype))


def normalize(x):
    return F.normalize(x, dim=-1)


def make_cu_seqlens(lengths: list[int]) -> torch.Tensor:
    cu = [0]
    for item in lengths:
        cu.append(cu[-1] + item)
    return torch.tensor(cu, device="npu", dtype=torch.int32)
