from __future__ import annotations

import torch

from kernel_suite.acc_utils import assert_close_tree, clone_value
from kernel_suite.kernels.common_builders import logsigmoid, make_cu_seqlens
from fla.ops.utils.cumsum import chunk_local_cumsum

KERNEL_NAME = "chunk_local_cumsum"
SOURCE_PATH = "fla/ops/utils/cumsum.py:432"
SOURCE_TESTS = ["tests/ops/test_utils.py"]
CASES = [
    {"name": "fixed_scalar", "tags": ["fixed", "scalar"], "shape": (4, 2048, 8), "chunk_size": 128, "dtype": torch.float32},
    {"name": "fixed_vector", "tags": ["fixed", "vector"], "shape": (4, 2048, 8, 1024), "chunk_size": 128, "dtype": torch.float32},
    {"name": "varlen_vector", "tags": ["varlen", "vector"], "shape": (1, 2048, 2, 1024), "chunk_size": 128, "dtype": torch.float16, "cu_seqlens": [200, 312, 688, 848]},
]


def build_inputs(case):
    x = logsigmoid(case["shape"], case["dtype"])
    cu = make_cu_seqlens(case["cu_seqlens"]) if "cu_seqlens" in case else None
    return {"x": x, "chunk_size": case["chunk_size"], "cu_seqlens": cu}


def launch(inputs):
    return chunk_local_cumsum(inputs["x"], chunk_size=inputs["chunk_size"], cu_seqlens=inputs["cu_seqlens"])


def reference(inputs):
    x = inputs["x"]
    if x.dim() == 3:
        if inputs["cu_seqlens"] is None:
            return torch.cat([x[:, i:i + inputs["chunk_size"], :].float().cumsum(1) for i in range(0, x.shape[1], inputs["chunk_size"])], 1)
        ref = []
        cu = inputs["cu_seqlens"].tolist()
        for start, end in zip(cu[:-1], cu[1:], strict=False):
            ref.append(torch.cat([x[:, i:min(end, i + inputs["chunk_size"]), :].float().cumsum(1) for i in range(start, end, inputs["chunk_size"])], 1))
        return torch.cat(ref, 1)
    if inputs["cu_seqlens"] is None:
        return torch.cat([x[:, i:i + inputs["chunk_size"], :].float().cumsum(1) for i in range(0, x.shape[1], inputs["chunk_size"])], 1)
    ref = []
    cu = inputs["cu_seqlens"].tolist()
    for start, end in zip(cu[:-1], cu[1:], strict=False):
        ref.append(torch.cat([x[:, i:min(end, i + inputs["chunk_size"]), :].float().cumsum(1) for i in range(start, end, inputs["chunk_size"])], 1))
    return torch.cat(ref, 1)


def run_accuracy_case(case):
    inputs = build_inputs(case)
    actual = launch(clone_value(inputs))
    expected = reference(clone_value(inputs))
    assert_close_tree(actual, expected, atol=1e-3, rtol=1e-3)


def make_perf_case(case):
    inputs = build_inputs(case)
    return lambda: launch(inputs), {"kernel": KERNEL_NAME, "name": case["name"], "tags": case["tags"]}
