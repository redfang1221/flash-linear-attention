from __future__ import annotations

import torch

from kernel_suite.acc_utils import assert_close_tree, clone_value
from kernel_suite.kernels.common_builders import beta, logsigmoid, make_cu_seqlens, randn
from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd as source_fn

KERNEL_NAME = "recompute_w_u_fwd"
SOURCE_PATH = "fla/ops/gated_delta_rule/wy_fast.py"
SOURCE_TESTS = ["tests/ops/test_gated_delta.py", "fla/ops/gated_delta_rule/chunk.py", "fla/ops/gated_delta_product/chunk.py"]
CASES = [
    {"name": "fixed", "tags": ["fixed"], "shape": (4, 2048, 8, 64), "dtype": torch.float16},
    {"name": "varlen", "tags": ["varlen"], "shape": (1, 2000, 4, 100), "dtype": torch.float16, "cu_seqlens": [15, 85, 200, 300, 1200, 200]},
]


def build_inputs(case):
    b, t, h, d = case["shape"]
    return {
        "k": randn((b, t, h, d), case["dtype"]),
        "v": randn((b, t, h, d), case["dtype"]),
        "beta": beta((b, t, h), case["dtype"]),
        "A": randn((b, t, h, 64), torch.float32).tril(-1),
        "g": logsigmoid((b, t, h), torch.float32),
        "cu_seqlens": make_cu_seqlens(case["cu_seqlens"]) if "cu_seqlens" in case else None,
    }


def launch(inputs):
    return source_fn(k=inputs["k"], v=inputs["v"], beta=inputs["beta"], A=inputs["A"], g=inputs["g"], cu_seqlens=inputs["cu_seqlens"])


def reference(inputs):
    return source_fn(k=inputs["k"], v=inputs["v"], beta=inputs["beta"], A=inputs["A"], g=inputs["g"], cu_seqlens=inputs["cu_seqlens"])


def run_accuracy_case(case):
    inputs = build_inputs(case)
    assert_close_tree(launch(clone_value(inputs)), reference(clone_value(inputs)), atol=1e-4, rtol=1e-4)


def make_perf_case(case):
    inputs = build_inputs(case)
    return lambda: launch(inputs), {"kernel": KERNEL_NAME, "name": case["name"], "tags": case["tags"]}
