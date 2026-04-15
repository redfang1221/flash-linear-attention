from __future__ import annotations

import torch
import triton

from kernel_suite.acc_utils import assert_close_tree, clone_value
from kernel_suite.kernels.common_builders import make_cu_seqlens, randn
from fla.modules.conv.triton.kernels import causal_conv1d_bwd_kernel
from fla.modules.conv.triton.ops import causal_conv1d_bwd, causal_conv1d_fwd
from fla.ops.utils import prepare_chunk_indices

KERNEL_NAME = "causal_conv1d_bwd_kernel"
SOURCE_PATH = "fla/modules/conv/triton/kernels.py"
SOURCE_TESTS = ["tests/modules/test_conv.py", "tests/context_parallel/test_cp_conv.py"]
CASES = [
    {"name": "fixed_fp32", "tags": ["fixed", "bias", "residual"], "B": 2, "T": 64, "D": 128, "W": 3, "dtype": torch.float32, "activation": "swish", "has_bias": True, "has_residual": True},
    {"name": "varlen_prefill", "tags": ["varlen", "prefill"], "B": 1, "T": 256, "D": 128, "W": 4, "dtype": torch.float16, "activation": "swish", "has_bias": True, "has_residual": True, "cu_seqlens": [16, 48, 64, 128], "initial_state_N": 4},
]


def build_inputs(case):
    x = randn((case["B"], case["T"], case["D"]), case["dtype"])
    dy = randn((case["B"], case["T"], case["D"]), case["dtype"])
    dx = torch.empty_like(x)
    weight = randn((case["D"], case["W"]), case["dtype"])
    bias = randn((case["D"],), case["dtype"]) if case["has_bias"] else None
    residual = randn((case["B"], case["T"], case["D"]), case["dtype"]) if case["has_residual"] else None
    initial_state = randn((case.get("initial_state_N", case["B"]), case["D"], case["W"]), case["dtype"])
    dht = randn(initial_state.shape, case["dtype"])
    cu_seqlens = make_cu_seqlens(case["cu_seqlens"]) if "cu_seqlens" in case else None
    chunk_indices = prepare_chunk_indices(cu_seqlens, 64, cu_seqlens_cpu=cu_seqlens.cpu()) if cu_seqlens is not None else None
    print("chunk_indices: ", chunk_indices)
    y, _ = causal_conv1d_fwd(
        x=x, weight=weight, bias=bias, residual=None, initial_state=initial_state, output_final_state=False,
        activation=None, cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens.cpu() if cu_seqlens is not None else None, chunk_indices=chunk_indices
    )
    dw = weight.new_empty(case["B"] * (len(chunk_indices) if chunk_indices is not None else triton.cdiv(case["T"], 64)), *weight.shape, dtype=torch.float32)
    db = bias.new_empty(case["B"] * (len(chunk_indices) if chunk_indices is not None else triton.cdiv(case["T"], 64)), *bias.shape, dtype=torch.float32) if bias is not None else None
    return {
        "x": x, "y": y, "dy": dy, "dx": dx, "weight": weight, "bias": bias, "residual": residual,
        "initial_state": initial_state, "dht": dht, "dw": dw, "db": db, "cu_seqlens": cu_seqlens,
        "chunk_indices": chunk_indices, "B": case["B"], "T": case["T"], "D": case["D"], "W": case["W"], "activation": case["activation"],
    }


def launch(inputs):
    x = inputs["x"]
    B, T, D, W = inputs["B"], inputs["T"], inputs["D"], inputs["W"]
    NT = len(inputs["chunk_indices"]) if inputs["chunk_indices"] is not None else triton.cdiv(T, 64)
    NB = triton.cdiv(B * T, 1024)
    BW = triton.next_power_of_2(W)
    stride_x_n, stride_x_t, stride_x_d = x.stride()
    stride_dx_n, stride_dx_t, stride_dx_d = inputs["dx"].stride()

    def grid(meta):
        return (triton.cdiv(D, meta["BD"]), NT, B)

    causal_conv1d_bwd_kernel[grid](
        x=x,
        y=inputs["y"] if inputs["activation"] is not None else None,
        weight=inputs["weight"],
        initial_state=inputs["initial_state"],
        dht=inputs["dht"],
        dy=inputs["dy"],
        dx=inputs["dx"],
        dw=inputs["dw"],
        db=inputs["db"],
        cu_seqlens=inputs["cu_seqlens"],
        chunk_indices=inputs["chunk_indices"],
        B=B,
        T=T,
        D=D,
        W=W,
        BT=64,
        BW=BW,
        NB=NB,
        stride_x_n=stride_x_n,
        stride_x_t=stride_x_t,
        stride_x_d=stride_x_d,
        stride_dx_n=stride_dx_n,
        stride_dx_t=stride_dx_t,
        stride_dx_d=stride_dx_d,
        ACTIVATION=inputs["activation"],
    )
    return inputs["dx"], inputs["dw"].sum(0), (inputs["db"].sum(0) if inputs["db"] is not None else None)


def reference(inputs):
    dx, dw, db, _, _ = causal_conv1d_bwd(
        x=inputs["x"], dy=inputs["dy"], dht=inputs["dht"], weight=inputs["weight"], bias=inputs["bias"],
        residual=inputs["residual"], initial_state=inputs["initial_state"], activation=inputs["activation"],
        cu_seqlens=inputs["cu_seqlens"], cu_seqlens_cpu=inputs["cu_seqlens"].cpu() if inputs["cu_seqlens"] is not None else None,
        chunk_indices=inputs["chunk_indices"]
    )
    return dx, dw, db


def run_accuracy_case(case):
    inputs = build_inputs(case)
    actual = launch(clone_value(inputs))
    expected = reference(clone_value(inputs))
    assert_close_tree(actual, expected, atol=3e-2, rtol=3e-2)


def return_args(inputs):
    x = inputs["x"]
    B, T, D, W = inputs["B"], inputs["T"], inputs["D"], inputs["W"]
    NT = len(inputs["chunk_indices"]) if inputs["chunk_indices"] is not None else triton.cdiv(T, 64)
    NB = triton.cdiv(B * T, 1024)
    BW = triton.next_power_of_2(W)
    stride_x_n, stride_x_t, stride_x_d = x.stride()
    stride_dx_n, stride_dx_t, stride_dx_d = inputs["dx"].stride()

    def grid(meta):
        return (triton.cdiv(D, meta["BD"]), NT, B)

    return {"grid": grid,
    "input_data": {
        "x": x, "y": inputs["y"] if inputs["activation"] is not None else None, "weight": inputs["weight"], "initial_state": inputs["initial_state"], "dht": inputs["dht"], "dy": inputs["dy"], "dx": inputs["dx"], "dw": inputs["dw"], "db": inputs["db"], "cu_seqlens": inputs["cu_seqlens"], "chunk_indices": inputs["chunk_indices"], "B": B, "T": T, "D": D, "W": W, "BT": 64, "BW": BW, "NB": NB, "stride_x_n": stride_x_n, "stride_x_t": stride_x_t, "stride_x_d": stride_x_d, "stride_dx_n": stride_dx_n, "stride_dx_t": stride_dx_t, "stride_dx_d": stride_dx_d, "ACTIVATION": inputs["activation"]
    }}


def fn_triton(grid, input_data):
    causal_conv1d_bwd_kernel[grid](**input_data)
    return


def make_perf_case(case):
    inputs = build_inputs(case)
    data = return_args(inputs)
    return fn_triton, data
