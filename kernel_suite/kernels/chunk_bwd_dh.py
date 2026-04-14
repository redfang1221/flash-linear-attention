from __future__ import annotations

import torch

from kernel_suite.acc_utils import assert_close_tree, clone_value
from kernel_suite.kernels.common_builders import logsigmoid, make_cu_seqlens, randn
from fla.ops.common.chunk_h import chunk_bwd_dh as source_fn

KERNEL_NAME = "chunk_bwd_dh"
SOURCE_PATH = "fla/ops/common/chunk_h.py"
SOURCE_TESTS = ["tests/ops/test_simple_gla.py", "tests/ops/test_gla.py", "fla/ops/gsa/chunk.py"]
CASES = [
    {"name": "scalar_g_fixed", "tags": ["fixed", "scalar_g"], "shape": (4, 2048, 8, 64), "g_mode": "g", "dtype": torch.float16},
    {"name": "vector_gk_varlen", "tags": ["varlen", "vector_gk"], "shape": (1, 2048, 4, 128), "g_mode": "gk", "dtype": torch.float16, "cu_seqlens": [1, 99, 200, 900, 848]},
]


def build_inputs(case):
    b, t, h, d = case["shape"]
    cu = make_cu_seqlens(case["cu_seqlens"]) if "cu_seqlens" in case else None
    n = len(case["cu_seqlens"]) if "cu_seqlens" in case else b
    return {
        "q": randn((b, t, h, d), case["dtype"]),
        "k": randn((b, t, h, d), case["dtype"]),
        "v": randn((b, t, h, d), case["dtype"]),
        "do": randn((b, t, h, d), case["dtype"]),
        "h0": randn((n, h, d, d), torch.float32),
        "dht": randn((n, h, d, d), torch.float32),
        "g": logsigmoid((b, t, h), torch.float32) if case["g_mode"] == "g" else None,
        "gk": logsigmoid((b, t, h, d), torch.float32) if case["g_mode"] == "gk" else None,
        "cu_seqlens": cu,
        "scale": d ** -0.5,
    }


def launch(inputs):
    return source_fn(q=inputs["q"], k=inputs["k"], v=inputs["v"], do=inputs["do"], h0=inputs["h0"], dht=inputs["dht"], scale=inputs["scale"], g=inputs["g"], gk=inputs["gk"], cu_seqlens=inputs["cu_seqlens"])


def reference(inputs):
    return source_fn(q=inputs["q"], k=inputs["k"], v=inputs["v"], do=inputs["do"], h0=inputs["h0"], dht=inputs["dht"], scale=inputs["scale"], g=inputs["g"], gk=inputs["gk"], cu_seqlens=inputs["cu_seqlens"])


def run_accuracy_case(case):
    inputs = build_inputs(case)
    assert_close_tree(launch(clone_value(inputs)), reference(clone_value(inputs)), atol=1e-4, rtol=1e-4)


def make_perf_case(case):
    inputs = build_inputs(case)
    return lambda: launch(inputs), {"kernel": KERNEL_NAME, "name": case["name"], "tags": case["tags"]}
