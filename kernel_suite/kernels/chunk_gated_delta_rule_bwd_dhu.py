from __future__ import annotations

import torch

from kernel_suite.acc_utils import assert_close_tree, clone_value
from kernel_suite.kernels.common_builders import logsigmoid, make_cu_seqlens, randn
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu as source_fn

KERNEL_NAME = "chunk_gated_delta_rule_bwd_dhu"
SOURCE_PATH = "fla/ops/common/chunk_delta_h.py"
SOURCE_TESTS = ["tests/ops/test_gated_delta.py", "tests/ops/test_delta.py", "fla/ops/kda/chunk_bwd.py"]
CASES = [
    {"name": "fixed_gqa", "tags": ["fixed", "gqa", "gva"], "shape": (2, 1024, 2, 128), "hv": 4, "dtype": torch.float16},
    {"name": "varlen_gk", "tags": ["varlen", "gk"], "shape": (1, 2000, 4, 100), "hv": 4, "dtype": torch.float16, "cu_seqlens": [15, 85, 200, 300, 1200, 200]},
]


def build_inputs(case):
    b, t, h, d = case["shape"]
    hv = case["hv"]
    cu = make_cu_seqlens(case["cu_seqlens"]) if "cu_seqlens" in case else None
    n = len(case["cu_seqlens"]) if "cu_seqlens" in case else b
    return {
        "q": randn((b, t, h, d), case["dtype"]),
        "k": randn((b, t, h, d), case["dtype"]),
        "w": randn((b, t, hv, d), case["dtype"]),
        "do": randn((b, t, hv, d), case["dtype"]),
        "dv": randn((b, t, hv, d), case["dtype"]),
        "g": logsigmoid((b, t, hv), torch.float32) if "cu_seqlens" not in case else None,
        "gk": logsigmoid((b, t, hv, d), torch.float32) if "cu_seqlens" in case else None,
        "h0": randn((n, hv, d, d), torch.float32),
        "dht": randn((n, hv, d, d), torch.float32),
        "cu_seqlens": cu,
        "scale": d ** -0.5,
    }


def launch(inputs):
    return source_fn(q=inputs["q"], k=inputs["k"], w=inputs["w"], do=inputs["do"], dv=inputs["dv"], g=inputs["g"], gk=inputs["gk"], h0=inputs["h0"], dht=inputs["dht"], scale=inputs["scale"], cu_seqlens=inputs["cu_seqlens"])


def reference(inputs):
    return source_fn(q=inputs["q"], k=inputs["k"], w=inputs["w"], do=inputs["do"], dv=inputs["dv"], g=inputs["g"], gk=inputs["gk"], h0=inputs["h0"], dht=inputs["dht"], scale=inputs["scale"], cu_seqlens=inputs["cu_seqlens"])


def run_accuracy_case(case):
    inputs = build_inputs(case)
    assert_close_tree(launch(clone_value(inputs)), reference(clone_value(inputs)), atol=1e-4, rtol=1e-4)


def make_perf_case(case):
    inputs = build_inputs(case)
    return lambda: launch(inputs), {"kernel": KERNEL_NAME, "name": case["name"], "tags": case["tags"]}
