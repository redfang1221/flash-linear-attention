from __future__ import annotations

import torch
import torch.nn.functional as F

from kernel_suite.acc_utils import assert_close_tree, clone_value
from kernel_suite.kernels.common_builders import randn
from fla.modules.fused_norm_gate import layer_norm_gated_fwd

KERNEL_NAME = "layer_norm_gated_fwd"
SOURCE_PATH = "fla/modules/fused_norm_gate.py:443"
SOURCE_TESTS = ["tests/modules/test_layernorm_gated.py"]
CASES = [
    {"name": "ln_small_affine_bias", "tags": ["layernorm", "smallD"], "B": 2, "H": 2, "T": 512, "D": 128, "activation": "silu", "weight": True, "bias": True, "rms": False},
    {"name": "ln_large_sigmoid", "tags": ["layernorm", "largeD"], "B": 2, "H": 2, "T": 2048, "D": 1200, "activation": "sigmoid", "weight": True, "bias": False, "rms": False},
    {"name": "rms_large_silu", "tags": ["rmsnorm", "largeD"], "B": 2, "H": 2, "T": 2048, "D": 1200, "activation": "silu", "weight": True, "bias": False, "rms": True},
]


def build_inputs(case):
    t = case["B"] * case["H"] * case["T"]
    x = randn((t, case["D"]), torch.float16)
    g = randn((t, case["D"]), torch.float16)
    weight = randn((case["D"],), torch.float16) if case["weight"] else None
    bias = randn((case["D"],), torch.float16) if case["bias"] else None
    return {"x": x, "g": g, "weight": weight, "bias": bias, "activation": case["activation"], "is_rms_norm": case["rms"]}


def launch(inputs):
    return layer_norm_gated_fwd(
        x=inputs["x"], g=inputs["g"], weight=inputs["weight"], bias=inputs["bias"],
        activation=inputs["activation"], is_rms_norm=inputs["is_rms_norm"]
    )


def reference(inputs):
    x = inputs["x"].float()
    g = inputs["g"].float()
    act = F.silu if inputs["activation"] == "silu" else torch.sigmoid
    if inputs["is_rms_norm"]:
        y = torch.nn.functional.rms_norm(x, normalized_shape=(x.shape[-1],), weight=inputs["weight"].float() if inputs["weight"] is not None else None, eps=1e-5)
    else:
        y = torch.layer_norm(x, normalized_shape=(x.shape[-1],), weight=inputs["weight"].float() if inputs["weight"] is not None else None, bias=inputs["bias"].float() if inputs["bias"] is not None else None, eps=1e-5)
    return y.to(inputs["x"].dtype) * act(g).to(inputs["x"].dtype)


def run_accuracy_case(case):
    inputs = build_inputs(case)
    actual, _, _, _ = launch(clone_value(inputs))
    expected = reference(clone_value(inputs))
    assert_close_tree(actual, expected, atol=3e-2, rtol=3e-2)


def make_perf_case(case):
    inputs = build_inputs(case)
    return lambda: launch(inputs), {"kernel": KERNEL_NAME, "name": case["name"], "tags": case["tags"]}
