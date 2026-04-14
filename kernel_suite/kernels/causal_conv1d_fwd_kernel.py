from __future__ import annotations

import torch
import triton

from kernel_suite.acc_utils import assert_close_tree, clone_value
from kernel_suite.kernels.common_builders import make_cu_seqlens, randn
from fla.modules.conv.triton.kernels import causal_conv1d_fwd_kernel
from fla.modules.conv.triton.ops import causal_conv1d_fwd
from fla.ops.utils import prepare_chunk_indices

KERNEL_NAME = "causal_conv1d_fwd_kernel"
SOURCE_PATH = "fla/modules/conv/triton/kernels.py"
SOURCE_TESTS = ["tests/modules/test_conv.py", "tests/context_parallel/test_cp_conv.py"]
CASES = [
    {"name": "fixed_fp32", "tags": ["fixed", "bias", "residual"], "B": 2, "T": 64, "D": 128, "W": 3, "dtype": torch.float32, "activation": "swish", "has_bias": True, "has_residual": True},
    {"name": "fixed_fp16", "tags": ["fixed"], "B": 2, "T": 128, "D": 128, "W": 4, "dtype": torch.float16, "activation": None, "has_bias": False, "has_residual": False},
    {"name": "varlen_prefill", "tags": ["varlen", "prefill"], "B": 1, "T": 256, "D": 128, "W": 4, "dtype": torch.float16, "activation": "swish", "has_bias": True, "has_residual": True, "cu_seqlens": [16, 48, 64, 128], "initial_state_N": 4},
]


def build_inputs(case):
    x = randn((case["B"], case["T"], case["D"]), case["dtype"])
    y = torch.empty_like(x)
    weight = randn((case["D"], case["W"]), case["dtype"])
    bias = randn((case["D"],), case["dtype"]) if case["has_bias"] else None
    residual = randn((case["B"], case["T"], case["D"]), case["dtype"]) if case["has_residual"] else None
    initial_state = randn((case.get("initial_state_N", case["B"]), case["D"], case["W"]), case["dtype"]) if "initial_state_N" in case else None
    cu_seqlens = make_cu_seqlens(case["cu_seqlens"]) if "cu_seqlens" in case else None
    chunk_indices = prepare_chunk_indices(cu_seqlens, 64, cu_seqlens_cpu=cu_seqlens.cpu()) if cu_seqlens is not None else None
    return {
        "x": x, "y": y, "weight": weight, "bias": bias, "residual": residual,
        "initial_state": initial_state, "cu_seqlens": cu_seqlens, "chunk_indices": chunk_indices,
        "activation": case["activation"], "B": case["B"], "T": case["T"], "D": case["D"], "W": case["W"],
    }


def launch(inputs):
    x = inputs["x"]
    B, T, D = inputs["B"], inputs["T"], inputs["D"]
    W = inputs["W"]
    NT = len(inputs["chunk_indices"]) if inputs["chunk_indices"] is not None else triton.cdiv(T, 64)
    NB = triton.cdiv(B * T, 1024)
    BW = triton.next_power_of_2(W)
    stride_x_n, stride_x_t, stride_x_d = x.stride()

    def grid(meta):
        return (triton.cdiv(D, meta["BD"]), NT, B)

    causal_conv1d_fwd_kernel[grid](
        x=x,
        y=inputs["y"],
        weight=inputs["weight"],
        bias=inputs["bias"],
        residual=inputs["residual"],
        cu_seqlens=inputs["cu_seqlens"],
        initial_state=inputs["initial_state"],
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
        ACTIVATION=inputs["activation"],
    )
    return inputs["y"]


def reference(inputs):
    y, _ = causal_conv1d_fwd(
        x=inputs["x"],
        weight=inputs["weight"],
        bias=inputs["bias"],
        residual=inputs["residual"],
        initial_state=inputs["initial_state"],
        output_final_state=False,
        activation=inputs["activation"],
        cu_seqlens=inputs["cu_seqlens"],
        cu_seqlens_cpu=inputs["cu_seqlens"].cpu() if inputs["cu_seqlens"] is not None else None,
        chunk_indices=inputs["chunk_indices"],
    )
    return y


def run_accuracy_case(case):
    inputs = build_inputs(case)
    actual = launch(clone_value(inputs))
    expected = reference(clone_value(inputs))
    assert_close_tree(actual, expected, atol=2e-3, rtol=2e-3)


def make_perf_case(case):
    inputs = build_inputs(case)
    return lambda: launch(inputs), {"kernel": KERNEL_NAME, "name": case["name"], "tags": case["tags"]}
