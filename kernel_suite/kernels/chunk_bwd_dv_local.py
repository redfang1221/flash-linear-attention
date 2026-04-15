from __future__ import annotations

import torch
import triton

from kernel_suite.acc_utils import assert_close_tree, clone_value
from kernel_suite.kernels.common_builders import logsigmoid, randn
from fla.utils import autotune_cache_kwargs, check_shared_mem
from fla.ops.common.chunk_o import chunk_bwd_dv_local, chunk_bwd_kernel_dv_local

KERNEL_NAME = "chunk_bwd_dv_local"
SOURCE_PATH = "fla/ops/common/chunk_o.py"
SOURCE_TESTS = ["tests/ops/test_gated_delta.py", "tests/ops/test_delta.py"]
CASES = [
    {"name": "delta_rule", "tags": ["fixed"], "q_shape": (4, 2048, 8, 64), "hv": 8, "dtype": torch.float16},
    {"name": "gated_delta_gqa", "tags": ["fixed", "gqa", "gva"], "q_shape": (2, 1024, 2, 128), "hv": 4, "dtype": torch.float16},
]


def build_inputs(case):
    b, t, hq, d = case["q_shape"]
    hv = case["hv"]
    return {
        "q": randn((b, t, hq, d), case["dtype"]),
        "k": randn((b, t, hq, d), case["dtype"]),
        "do": randn((b, t, hv, d), case["dtype"]),
        "g": logsigmoid((b, t, hv), torch.float32),
        "A": randn((b, t, hv, 64), torch.float32).tril(-1),
        "scale": d ** -0.5,
    }


def launch(inputs):
    return chunk_bwd_dv_local(q=inputs["q"], k=inputs["k"], do=inputs["do"], g=inputs["g"], A=inputs["A"], scale=inputs["scale"])


def reference(inputs):
    return chunk_bwd_dv_local(q=inputs["q"], k=inputs["k"], do=inputs["do"], g=inputs["g"], A=inputs["A"], scale=inputs["scale"])


def run_accuracy_case(case):
    inputs = build_inputs(case)
    assert_close_tree(launch(clone_value(inputs)), reference(clone_value(inputs)), atol=1e-4, rtol=1e-4)


def return_args(inputs):
    q=inputs["q"]
    k=inputs["k"]
    do=inputs["do"]
    g=inputs["g"]
    g_gamma=None
    A=inputs["A"]
    scale=inputs["scale"]
    cu_seqlens=None
    chunk_size=64
    chunk_indices=None
    use_exp2=False
    B, T, H, K, V, HV = *k.shape, do.shape[-1], do.shape[2]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    # H100 can have larger block size
    if check_shared_mem('hopper', k.device.index):
        CONST_TILING = 128
    elif check_shared_mem('ada', k.device.index):
        CONST_TILING = 64
    else:
        CONST_TILING = 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dv = torch.empty_like(do)
    grid = (NT, B * HV)
    return {"grid": grid,
    "input_data": {
        "q": q,"k": k,"g": g,"g_gamma": g_gamma,"A": A,"do": do,"dv": dv,"cu_seqlens": cu_seqlens,"chunk_indices": chunk_indices,"scale": scale,"T": T,"H": H,"HV": HV,"K": K,"V": V,"BT": BT,"BK": BK,"BV": BV,"USE_EXP2": use_exp2
    }}


def fn_triton(grid, input_data):
    chunk_bwd_kernel_dv_local[grid](**input_data)
    return


def make_perf_case(case):
    inputs = build_inputs(case)
    data = return_args(inputs)
    return fn_triton, data

