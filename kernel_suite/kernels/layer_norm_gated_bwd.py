from __future__ import annotations

import torch
import triton
import math

from kernel_suite.acc_utils import assert_close_tree, clone_value
from kernel_suite.kernels.common_builders import randn
from fla.modules.fused_norm_gate import layer_norm_gated_bwd, layer_norm_gated_fwd, layer_norm_gated_bwd_kernel, layer_norm_gated_bwd_kernel1
from fla.utils import autotune_cache_kwargs, check_shared_mem
from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.utils import autotune_cache_kwargs, get_multiprocessor_count, input_guard

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


def return_args(inputs):
    activation = "swish"
    eps = 1e-5
    mean = None
    rstd = None
    dresidual = None
    has_residual = False
    is_rms_norm = False
    x_dtype: torch.dtype = None
    recompute_output: bool = False
    dy=inputs["dy"]
    x=inputs["x"]
    g=inputs["g"]
    weight=inputs["weight"]
    bias=inputs["bias"]
    activation=inputs["activation"]
    mean=inputs["mean"]
    rstd=inputs["rstd"]
    has_residual=inputs["has_residual"]
    T, D = x.shape
    assert dy.shape == (T, D)
    if dresidual is not None:
        assert dresidual.shape == (T, D)
    if weight is not None:
        assert weight.shape == (D,)
    if bias is not None:
        assert bias.shape == (D,)
    # allocate output
    dx = torch.empty_like(x) if x_dtype is None else torch.empty(T, D, dtype=x_dtype, device=x.device)
    dg = torch.empty_like(g) if x_dtype is None else torch.empty(T, D, dtype=x_dtype, device=x.device)
    dresidual_in = torch.empty_like(x) if has_residual and dx.dtype != x.dtype else None
    y = torch.empty(T, D, dtype=dy.dtype, device=dy.device) if recompute_output else None

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # cap program count to T so no program is completely idle.
    # without this, high-SM GPUs (e.g. B200, 160 SMs) with small T would
    # launch idle programs whose make_block_ptr offsets exceed the tensor shape.
    NS = min(get_multiprocessor_count(x.device.index), T)
    BS = math.ceil(T / NS)

    dw = torch.empty((NS, D), dtype=torch.float, device=weight.device) if weight is not None else None
    db = torch.empty((NS, D), dtype=torch.float, device=bias.device) if bias is not None else None
    grid = (NS,)

    if D <= 512:
        NB = triton.cdiv(T, 2048 * 32)
        return {"grid": grid, "input_data": {
            "x": x, "g": g, "w": weight, "b": bias, "y": y, "dy": dy, "dx": dx, "dg": dg, "dw": dw, "db": db, "dresidual": dresidual, "dresidual_in": dresidual_in, "mean": mean, "rstd": rstd, "T": T, "D": D, "BS": BS, "BD": BD, "NB": NB, "ACTIVATION": activation, "IS_RMS_NORM": is_rms_norm, "STORE_DRESIDUAL": dresidual_in is not None,
        }}
    else:
        return {"grid": grid, "input_data": {
            "x": x, "g": g, "w": weight, "b": bias, "y": y, "dy": dy, "dx": dx, "dg": dg, "dw": dw, "db": db, "dresidual": dresidual, "dresidual_in": dresidual_in, "mean": mean, "rstd": rstd, "T": T, "D": D, "BS": BS, "BD": BD, "ACTIVATION": activation, "IS_RMS_NORM": is_rms_norm, "STORE_DRESIDUAL": dresidual_in is not None,
        }}

def fn_triton(grid, input_data):
    if input_data["D"] <= 512:
        layer_norm_gated_bwd_kernel[grid](**input_data)
    else:
        layer_norm_gated_bwd_kernel1[grid](**input_data)
    return


def make_perf_case(case):
    inputs = build_inputs(case)
    data = return_args(inputs)
    return fn_triton, data
