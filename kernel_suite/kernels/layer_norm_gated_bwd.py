from __future__ import annotations

import torch

from kernel_suite.acc_utils import assert_close_tree, clone_value
from kernel_suite.kernels.common_builders import randn
from fla.modules.fused_norm_gate import layer_norm_gated_bwd, layer_norm_gated_fwd

KERNEL_NAME = "layer_norm_gated_bwd"
SOURCE_PATH = "fla/modules/fused_norm_gate.py:527"
SOURCE_TESTS = ["tests/modules/test_layernorm_gated.py"]
CASES = [
    {"name": "ln_small_affine_bias", "tags": ["layernorm", "smallD"], "T": 2048, "D": 128, "activation": "silu", "has_bias": True, "has_residual": True},
]


def build_inputs(case):
    x = randn((case["T"], case["D"]), torch.float16)
    g = randn((case["T"], case["D"]), torch.float16)
    weight = randn((case["D"],), torch.float16)
    bias = randn((case["D"],), torch.float16) if case["has_bias"] else None
    residual = randn((case["T"], case["D"]), torch.float16) if case["has_residual"] else None
    dy = randn((case["T"], case["D"]), torch.float16)
    _, mean, rstd, _ = layer_norm_gated_fwd(x=x, g=g, weight=weight, bias=bias, activation=case["activation"], residual=residual)
    return {"dy": dy, "x": x, "g": g, "weight": weight, "bias": bias, "mean": mean, "rstd": rstd, "activation": case["activation"], "has_residual": case["has_residual"]}


def launch(inputs):
    return layer_norm_gated_bwd(
        dy=inputs["dy"], x=inputs["x"], g=inputs["g"], weight=inputs["weight"], bias=inputs["bias"],
        activation=inputs["activation"], mean=inputs["mean"], rstd=inputs["rstd"], has_residual=inputs["has_residual"]
    )


def reference(inputs):
    return layer_norm_gated_bwd(
        dy=inputs["dy"], x=inputs["x"], g=inputs["g"], weight=inputs["weight"], bias=inputs["bias"],
        activation=inputs["activation"], mean=inputs["mean"], rstd=inputs["rstd"], has_residual=inputs["has_residual"]
    )


def run_accuracy_case(case):
    inputs = build_inputs(case)
    actual = launch(clone_value(inputs))
    expected = reference(clone_value(inputs))
    assert_close_tree(actual, expected, atol=1e-4, rtol=1e-4)


def make_perf_case(case):
    inputs = build_inputs(case)
    return lambda: launch(inputs), {"kernel": KERNEL_NAME, "name": case["name"], "tags": case["tags"]}
