from __future__ import annotations

import torch

from kernel_suite.acc_utils import assert_close_tree, clone_value
from kernel_suite.kernels.common_builders import logsigmoid, make_cu_seqlens, randn
from fla.ops.common.fused_recurrent import fused_recurrent_fwd as source_fn

KERNEL_NAME = "fused_recurrent_fwd"
SOURCE_PATH = "fla/ops/common/fused_recurrent.py:340"
SOURCE_TESTS = ["tests/ops/test_simple_gla.py", "tests/ops/test_gla.py", "tests/ops/test_gated_delta.py", "tests/ops/test_delta.py"]
CASES = [
    {"name": "scalar_g_fixed", "tags": ["fixed", "scalar_g"], "shape": (4, 2048, 8, 64), "gate": "g", "dtype": torch.float16},
    {"name": "vector_gk_varlen", "tags": ["varlen", "vector_gk"], "shape": (1, 2048, 4, 100), "gate": "gk", "dtype": torch.float16, "cu_seqlens": [15, 85, 200, 300, 1200, 248]},
    {"name": "reverse_scalar", "tags": ["fixed", "reverse"], "shape": (2, 512, 4, 64), "gate": "g", "dtype": torch.float16, "reverse": True},
]


def build_inputs(case):
    b, t, h, d = case["shape"]
    cu = make_cu_seqlens(case["cu_seqlens"]) if "cu_seqlens" in case else None
    n = len(case["cu_seqlens"]) if "cu_seqlens" in case else b
    return {
        "q": randn((b, t, h, d), case["dtype"]),
        "k": randn((b, t, h, d), case["dtype"]),
        "v": randn((b, t, h, d), case["dtype"]),
        "g": logsigmoid((b, t, h), torch.float32) if case["gate"] == "g" else None,
        "gk": logsigmoid((b, t, h, d), torch.float32) if case["gate"] == "gk" else None,
        "initial_state": randn((n, h, d, d), torch.float32),
        "cu_seqlens": cu,
        "reverse": case.get("reverse", False),
    }


def launch(inputs):
    return source_fn(q=inputs["q"], k=inputs["k"], v=inputs["v"], g=inputs["g"], gk=inputs["gk"], initial_state=inputs["initial_state"], output_final_state=True, reverse=inputs["reverse"], cu_seqlens=inputs["cu_seqlens"])


def reference(inputs):
    return source_fn(q=inputs["q"], k=inputs["k"], v=inputs["v"], g=inputs["g"], gk=inputs["gk"], initial_state=inputs["initial_state"], output_final_state=True, reverse=inputs["reverse"], cu_seqlens=inputs["cu_seqlens"])


def run_accuracy_case(case):
    inputs = build_inputs(case)
    assert_close_tree(launch(clone_value(inputs)), reference(clone_value(inputs)), atol=1e-4, rtol=1e-4)


def make_perf_case(case):
    inputs = build_inputs(case)
    return lambda: launch(inputs), {"kernel": KERNEL_NAME, "name": case["name"], "tags": case["tags"]}
