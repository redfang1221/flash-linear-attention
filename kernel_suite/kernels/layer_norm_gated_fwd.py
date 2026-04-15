from __future__ import annotations

import torch
import triton
import math
import torch.nn.functional as F

from kernel_suite.acc_utils import assert_close_tree, clone_value
from kernel_suite.kernels.common_builders import randn
from fla.modules.fused_norm_gate import layer_norm_gated_fwd, layer_norm_gated_fwd_kernel, layer_norm_gated_fwd_kernel1
from fla.utils import autotune_cache_kwargs, check_shared_mem
from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.utils import autotune_cache_kwargs, get_multiprocessor_count, input_guard

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


def return_args(inputs):
    actual = launch(clone_value(inputs))
    activation: str = "swish"
    eps: float = 1e-5
    residual: torch.Tensor = None
    out_dtype: torch.dtype = None
    residual_dtype: torch.dtype = None
    is_rms_norm: bool = False
    x=inputs["x"]
    g=inputs["g"]
    weight=inputs["weight"]
    bias=inputs["bias"]
    activation=inputs["activation"]
    is_rms_norm=inputs["is_rms_norm"]
    if residual is not None:
        residual_dtype = residual.dtype
    T, D = x.shape
    if residual is not None:
        assert residual.shape == (T, D)
    if weight is not None:
        assert weight.shape == (D,)
    if bias is not None:
        assert bias.shape == (D,)
    # allocate output
    y = torch.empty_like(x, dtype=x.dtype if out_dtype is None else out_dtype)
    if residual is not None or (residual_dtype is not None and residual_dtype != x.dtype):
        residual_out = torch.empty(T, D, device=x.device, dtype=residual_dtype)
    else:
        residual_out = None
    mean = torch.empty((T,), dtype=torch.float, device=x.device) if not is_rms_norm else None
    rstd = torch.empty((T,), dtype=torch.float, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    if D <= 512:
        NB = triton.cdiv(T, 2048 * 32)
        print(f"Best Config: {layer_norm_gated_fwd_kernel.fn.best_config}")
        config_obj = layer_norm_gated_fwd_kernel.fn.best_config
        params = config_obj.kwargs
        BS = params.get("BT")
        def grid(meta):
            return (triton.cdiv(T, meta["BT"]),)
        return {"grid": grid, "input_data": {
            "x": x, "g": g, "y": y, "w": weight, "b": bias, "residual": residual, "residual_out": residual_out, "mean": mean, "rstd": rstd, "eps": eps, "T": T, "D": D, "BD": BD, "NB": NB, "ACTIVATION": activation, "IS_RMS_NORM": is_rms_norm,
        }}
    else:
        return {"grid": (T,), "input_data": {
            "x": x, "g": g, "y": y, "w": weight, "b": bias, "residual": residual, "residual_out": residual_out, "mean": mean, "rstd": rstd, "eps": eps, "D": D, "BD": BD, "ACTIVATION": activation, "IS_RMS_NORM": is_rms_norm
        }}


def fn_triton(grid, input_data):
    if input_data["D"] <= 512:
        layer_norm_gated_fwd_kernel[grid](**input_data)
    else:
        layer_norm_gated_fwd_kernel1[grid](**input_data)
    return


def make_perf_case(case):
    inputs = build_inputs(case)
    data = return_args(inputs)
    return fn_triton, data

