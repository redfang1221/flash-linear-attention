from __future__ import annotations

import torch

from kernel_suite.acc_utils import assert_close_tree, clone_value
from kernel_suite.kernels.common_builders import logsigmoid, randn
from fla.ops.common.chunk_o import chunk_bwd_dv as source_fn

KERNEL_NAME = "chunk_bwd_dv"
SOURCE_PATH = "fla/ops/common/chunk_o.py"
SOURCE_TESTS = ["tests/ops/test_simple_gla.py"]
CASES = [
    {"name": "simple_gla_scalar_g", "tags": ["fixed", "scalar_g"], "shape": (4, 2048, 8, 64), "dtype": torch.float16},
]


def build_inputs(case):
    b, t, h, d = case["shape"]
    return {
        "q": randn((b, t, h, d), case["dtype"]),
        "k": randn((b, t, h, d), case["dtype"]),
        "do": randn((b, t, h, d), case["dtype"]),
        "dh": randn((b, (t + 63) // 64, h, d, d), case["dtype"]),
        "g": logsigmoid((b, t, h), torch.float32),
        "scale": d ** -0.5,
    }


def launch(inputs):
    return source_fn(q=inputs["q"], k=inputs["k"], do=inputs["do"], dh=inputs["dh"], g=inputs["g"], scale=inputs["scale"])


def reference(inputs):
    return source_fn(q=inputs["q"], k=inputs["k"], do=inputs["do"], dh=inputs["dh"], g=inputs["g"], scale=inputs["scale"])


def run_accuracy_case(case):
    inputs = build_inputs(case)
    assert_close_tree(launch(clone_value(inputs)), reference(clone_value(inputs)), atol=1e-4, rtol=1e-4)


def make_perf_case(case):
    inputs = build_inputs(case)
    return lambda: launch(inputs), {"kernel": KERNEL_NAME, "name": case["name"], "tags": case["tags"]}
