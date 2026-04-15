from __future__ import annotations

import torch
import triton
import math

from kernel_suite.acc_utils import assert_close_tree, clone_value
from kernel_suite.kernels.common_builders import beta, logsigmoid, make_cu_seqlens, randn
from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd, prepare_wy_repr_bwd_kernel
from fla.utils import autotune_cache_kwargs, check_shared_mem
from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.utils import autotune_cache_kwargs, get_multiprocessor_count, input_guard


KERNEL_NAME = "prepare_wy_repr_bwd"
SOURCE_PATH = "fla/ops/gated_delta_rule/wy_fast.py"
SOURCE_TESTS = ["tests/ops/test_gated_delta.py", "fla/ops/gated_delta_rule/chunk.py"]
CASES = [
    {"name": "fixed", "tags": ["fixed"], "shape": (4, 2048, 8, 64), "dtype": torch.float16},
    {"name": "varlen", "tags": ["varlen"], "shape": (1, 2000, 4, 100), "dtype": torch.float16, "cu_seqlens": [15, 85, 200, 300, 1200, 200]},
]


def build_inputs(case):
    b, t, h, d = case["shape"]
    return {
        "k": randn((b, t, h, d), case["dtype"]),
        "v": randn((b, t, h, d), case["dtype"]),
        "beta": beta((b, t, h), case["dtype"]),
        "A": randn((b, t, h, 64), torch.float32).tril(-1),
        "dw": randn((b, t, h, d), case["dtype"]),
        "du": randn((b, t, h, d), case["dtype"]),
        "g": logsigmoid((b, t, h), torch.float32),
        "cu_seqlens": make_cu_seqlens(case["cu_seqlens"]) if "cu_seqlens" in case else None,
    }


def launch(inputs):
    return prepare_wy_repr_bwd(k=inputs["k"], v=inputs["v"], beta=inputs["beta"], A=inputs["A"], dw=inputs["dw"], du=inputs["du"], g=inputs["g"], cu_seqlens=inputs["cu_seqlens"])


def reference(inputs):
    return prepare_wy_repr_bwd(k=inputs["k"], v=inputs["v"], beta=inputs["beta"], A=inputs["A"], dw=inputs["dw"], du=inputs["du"], g=inputs["g"], cu_seqlens=inputs["cu_seqlens"])


def run_accuracy_case(case):
    inputs = build_inputs(case)
    assert_close_tree(launch(clone_value(inputs)), reference(clone_value(inputs)), atol=1e-4, rtol=1e-4)


def return_args(inputs):
    g: torch.Tensor = None,
    cu_seqlens: torch.LongTensor = None,
    chunk_indices: torch.LongTensor = None,
    use_exp2: bool = False,
    k=inputs["k"]
    v=inputs["v"]
    beta=inputs["beta"]
    A=inputs["A"]
    dw=inputs["dw"]
    du=inputs["du"]
    g=inputs["g"]
    cu_seqlens=inputs["cu_seqlens"]
    B, T, H, K, V, HV = *k.shape, v.shape[-1], v.shape[2]
    BT = 64
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    CONST_TILING = 64 if check_shared_mem() else 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    dk = k.new_empty(B, T, HV, K)
    dv = torch.empty_like(v)
    dg = torch.empty_like(g) if g is not None else None
    db = torch.empty_like(beta)
    return {"grid": (NT, B * HV), "input_data": {
        "k": k, "v": v, "beta": beta, "g": g, "A": A, "dw": dw, "du": du, "dk": dk, "dv": dv, "db": db, "dg": dg, "cu_seqlens": cu_seqlens, "chunk_indices": chunk_indices, "T": T, "H": H, "HV": HV, "K": K, "V": V, "BT": BT, "BK": BK, "BV": BV, "USE_EXP2": use_exp2
    }}


def fn_triton(grid, input_data):
    prepare_wy_repr_bwd_kernel[grid](**input_data)


def make_perf_case(case):
    inputs = build_inputs(case)
    data = return_args(inputs)
    return fn_triton, data