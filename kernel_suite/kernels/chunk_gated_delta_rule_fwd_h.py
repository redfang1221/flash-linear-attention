from __future__ import annotations

import torch

from kernel_suite.acc_utils import assert_close_tree, clone_value
from kernel_suite.kernels.common_builders import logsigmoid, make_cu_seqlens, randn
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h as source_fn

KERNEL_NAME = "chunk_gated_delta_rule_fwd_h"
SOURCE_PATH = "fla/ops/common/chunk_delta_h.py"
SOURCE_TESTS = ["tests/ops/test_gated_delta.py", "tests/ops/test_delta.py", "fla/ops/kda/chunk_fwd.py"]
CASES = [
    {"name": "fixed_gqa", "tags": ["fixed", "gqa", "gva"], "shape": (2, 1024, 2, 128), "hv": 4, "dtype": torch.float16},
    {"name": "transpose_state", "tags": ["fixed", "transpose"], "shape": (4, 2048, 8, 64), "hv": 8, "dtype": torch.float16, "transpose": True},
    {"name": "varlen_gk", "tags": ["varlen", "gk"], "shape": (1, 2000, 4, 100), "hv": 4, "dtype": torch.float16, "cu_seqlens": [15, 85, 200, 300, 1200, 200]},
]


def build_inputs(case):
    b, t, h, d = case["shape"]
    hv = case["hv"]
    cu = make_cu_seqlens(case["cu_seqlens"]) if "cu_seqlens" in case else None
    n = len(case["cu_seqlens"]) if "cu_seqlens" in case else b
    state = randn((n, hv, d, d), torch.float32)
    if case.get("transpose"):
        state = state.transpose(-1, -2).contiguous()
    return {
        "k": randn((b, t, h, d), case["dtype"]),
        "w": randn((b, t, hv, d), case["dtype"]),
        "u": randn((b, t, hv, d), case["dtype"]),
        "g": logsigmoid((b, t, hv), torch.float32) if "cu_seqlens" not in case else None,
        "gk": logsigmoid((b, t, hv, d), torch.float32) if "cu_seqlens" in case else None,
        "initial_state": state,
        "cu_seqlens": cu,
        "transpose_state_layout": case.get("transpose", False),
    }


def launch(inputs):
    return source_fn(k=inputs["k"], w=inputs["w"], u=inputs["u"], g=inputs["g"], gk=inputs["gk"], initial_state=inputs["initial_state"], output_final_state=True, cu_seqlens=inputs["cu_seqlens"], transpose_state_layout=inputs["transpose_state_layout"])


def reference(inputs):
    return source_fn(k=inputs["k"], w=inputs["w"], u=inputs["u"], g=inputs["g"], gk=inputs["gk"], initial_state=inputs["initial_state"], output_final_state=True, cu_seqlens=inputs["cu_seqlens"], transpose_state_layout=inputs["transpose_state_layout"])


def run_accuracy_case(case):
    inputs = build_inputs(case)
    assert_close_tree(launch(clone_value(inputs)), reference(clone_value(inputs)), atol=1e-4, rtol=1e-4)


def make_perf_case(case):
    inputs = build_inputs(case)
    return lambda: launch(inputs), {"kernel": KERNEL_NAME, "name": case["name"], "tags": case["tags"]}
