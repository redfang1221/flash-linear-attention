from __future__ import annotations

import torch

from kernel_suite.acc_utils import assert_close_tree, clone_value
from kernel_suite.kernels.common_builders import logsigmoid, randn
from fla.ops.common.chunk_o import chunk_fwd_o as source_fn

KERNEL_NAME = "chunk_fwd_o"
SOURCE_PATH = "fla/ops/common/chunk_o.py"
SOURCE_TESTS = ["tests/ops/test_simple_gla.py", "tests/ops/test_gated_delta.py", "tests/ops/test_delta.py"]
CASES = [
    {"name": "simple_gla_scalar_g", "tags": ["fixed", "scalar_g"], "q_shape": (4, 2048, 8, 64), "v_heads": 8, "dtype": torch.float16, "g_mode": "g"},
    {"name": "gated_delta_gqa", "tags": ["fixed", "gqa", "gva"], "q_shape": (2, 1024, 2, 128), "v_heads": 4, "dtype": torch.float16, "g_mode": "g"},
]


def build_inputs(case):
    b, t, hq, d = case["q_shape"]
    hv = case["v_heads"]
    return {
        "q": randn((b, t, hq, d), case["dtype"]),
        "k": randn((b, t, hq, d), case["dtype"]),
        "v": randn((b, t, hv, d), case["dtype"]),
        "h": randn((b, (t + 63) // 64, hv, d, d), case["dtype"]),
        "g": logsigmoid((b, t, hv), torch.float32) if case["g_mode"] == "g" else None,
        "scale": d ** -0.5,
    }


def launch(inputs):
    return source_fn(q=inputs["q"], k=inputs["k"], v=inputs["v"], h=inputs["h"], g=inputs["g"], scale=inputs["scale"])


def reference(inputs):
    return source_fn(q=inputs["q"], k=inputs["k"], v=inputs["v"], h=inputs["h"], g=inputs["g"], scale=inputs["scale"])


def run_accuracy_case(case):
    inputs = build_inputs(case)
    assert_close_tree(launch(clone_value(inputs)), reference(clone_value(inputs)), atol=1e-4, rtol=1e-4)


def make_perf_case(case):
    inputs = build_inputs(case)
    return lambda: launch(inputs), {"kernel": KERNEL_NAME, "name": case["name"], "tags": case["tags"]}
