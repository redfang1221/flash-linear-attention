from __future__ import annotations

import torch
import triton

from kernel_suite.acc_utils import assert_close_tree, clone_value
from kernel_suite.kernels.common_builders import logsigmoid, randn
from fla.ops.common.chunk_o import chunk_fwd_o, chunk_fwd_kernel_o
from fla.utils import autotune_cache_kwargs, check_shared_mem

KERNEL_NAME = "chunk_fwd_o"
SOURCE_PATH = "fla/ops/common/chunk_o.py"
SOURCE_TESTS = ["tests/ops/test_simple_gla.py", "tests/ops/test_gated_delta.py", "tests/ops/test_delta.py"]
CASES = [
    {"name": "simple_gla_scalar_g", "tags": ["fixed", "scalar_g"], "q_shape": (4, 2048, 8, 64), "v_heads": 8, "dtype": torch.float16, "g_mode": "g"},
    {"name": "gated_delta_gqa", "tags": ["fixed", "gqa", "gva"], "q_shape": (2, 1024, 2, 128), "v_heads": 4, "dtype": torch.float16, "g_mode": "g"},
]


def build_inputs(case):
    b, t, hq, d = case["q_shape"]
    hv = case["v_heads"]
    return {
        "q": randn((b, t, hq, d), case["dtype"]),
        "k": randn((b, t, hq, d), case["dtype"]),
        "v": randn((b, t, hv, d), case["dtype"]),
        "h": randn((b, (t + 63) // 64, hv, d, d), case["dtype"]),
        "g": logsigmoid((b, t, hv), torch.float32) if case["g_mode"] == "g" else None,
        "scale": d ** -0.5,
    }


def launch(inputs):
    return chunk_fwd_o(q=inputs["q"], k=inputs["k"], v=inputs["v"], h=inputs["h"], g=inputs["g"], scale=inputs["scale"])


def reference(inputs):
    return chunk_fwd_o(q=inputs["q"], k=inputs["k"], v=inputs["v"], h=inputs["h"], g=inputs["g"], scale=inputs["scale"])


def run_accuracy_case(case):
    inputs = build_inputs(case)
    assert_close_tree(launch(clone_value(inputs)), reference(clone_value(inputs)), atol=1e-4, rtol=1e-4)


def return_args(inputs):
    q=inputs["q"]
    k=inputs["k"]
    v=inputs["v"]
    h=inputs["h"]
    g=inputs["g"]
    g_gamma=None
    scale=inputs["scale"]
    cu_seqlens=None
    chunk_size=64
    chunk_indices=None
    use_exp2=False
    transpose_state_layout=False
    B, T, H, K, V, HV = *q.shape, v.shape[-1], v.shape[2]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o = torch.empty_like(v)
    def grid(meta): return (triton.cdiv(V, meta['BV']), NT, B * HV)
    return {"grid": grid,
    "input_data": {
        "q": q, "k": k, "v": v, "h": h, "g": g, "g_gamma": g_gamma, "o": o, "cu_seqlens": cu_seqlens, "chunk_indices": chunk_indices, "scale": scale, "T": T, "H": H, "HV": HV, "K": K, "V": V, "BT": BT, "USE_EXP2": use_exp2, "TRANSPOSE_STATE": transpose_state_layout
    }}


def fn_triton(grid, input_data):
    chunk_fwd_kernel_o[grid](**input_data)
    return


def make_perf_case(case):
    inputs = build_inputs(case)
    data = return_args(inputs)
    return fn_triton, data