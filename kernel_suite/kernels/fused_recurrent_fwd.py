from __future__ import annotations

import torch
import triton

from kernel_suite.acc_utils import assert_close_tree, clone_value
from kernel_suite.kernels.common_builders import logsigmoid, make_cu_seqlens, randn
from fla.ops.common.fused_recurrent import fused_recurrent_fwd, fused_recurrent_fwd_kernel
from fla.utils import autotune_cache_kwargs, check_shared_mem
from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets

KERNEL_NAME = "fused_recurrent_fwd"
SOURCE_PATH = "fla/ops/common/fused_recurrent.py:340"
SOURCE_TESTS = ["tests/ops/test_simple_gla.py", "tests/ops/test_gla.py", "tests/ops/test_gated_delta.py", "tests/ops/test_delta.py"]
CASES = [
    {"name": "scalar_g_fixed", "tags": ["fixed", "scalar_g"], "shape": (4, 2048, 8, 64), "gate": "g", "dtype": torch.float16},
    {"name": "vector_gk_varlen", "tags": ["varlen", "vector_gk"], "shape": (1, 2048, 4, 100), "gate": "gk", "dtype": torch.float16, "cu_seqlens": [15, 85, 200, 300, 1200, 248]},
    {"name": "reverse_scalar", "tags": ["fixed", "reverse"], "shape": (2, 512, 4, 64), "gate": "g", "dtype": torch.float16, "reverse": True},
]


def build_inputs(case):
    b, t, h, d = case["shape"]
    cu = make_cu_seqlens(case["cu_seqlens"]) if "cu_seqlens" in case else None
    n = len(case["cu_seqlens"]) if "cu_seqlens" in case else b
    return {
        "q": randn((b, t, h, d), case["dtype"]),
        "k": randn((b, t, h, d), case["dtype"]),
        "v": randn((b, t, h, d), case["dtype"]),
        "g": logsigmoid((b, t, h), torch.float32) if case["gate"] == "g" else None,
        "gk": logsigmoid((b, t, h, d), torch.float32) if case["gate"] == "gk" else None,
        "initial_state": randn((n, h, d, d), torch.float32),
        "cu_seqlens": cu,
        "reverse": case.get("reverse", False),
    }


def launch(inputs):
    return fused_recurrent_fwd(q=inputs["q"], k=inputs["k"], v=inputs["v"], g=inputs["g"], gk=inputs["gk"], initial_state=inputs["initial_state"], output_final_state=True, reverse=inputs["reverse"], cu_seqlens=inputs["cu_seqlens"])


def reference(inputs):
    return fused_recurrent_fwd(q=inputs["q"], k=inputs["k"], v=inputs["v"], g=inputs["g"], gk=inputs["gk"], initial_state=inputs["initial_state"], output_final_state=True, reverse=inputs["reverse"], cu_seqlens=inputs["cu_seqlens"])


def run_accuracy_case(case):
    inputs = build_inputs(case)
    assert_close_tree(launch(clone_value(inputs)), reference(clone_value(inputs)), atol=1e-4, rtol=1e-4)


def return_args(inputs):
    g: torch.Tensor = None
    g_gamma: torch.Tensor = None
    gk: torch.Tensor = None
    gv: torch.Tensor = None
    scale: float = None
    initial_state: torch.Tensor = None
    output_final_state: bool = False
    reverse: bool = False
    cu_seqlens: torch.LongTensor = None
    q=inputs["q"]
    k=inputs["k"]
    v=inputs["v"]
    g=inputs["g"]
    gk=inputs["gk"]
    initial_state=inputs["initial_state"]
    output_final_state=True
    reverse=inputs["reverse"]
    cu_seqlens=inputs["cu_seqlens"]
    B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = min(triton.next_power_of_2(K), 64), min(triton.next_power_of_2(V), 64)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

    h0 = initial_state
    ht = q.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    o = q.new_empty(NK, *v.shape, dtype=torch.float32)

    grid = (NV, NK, N * H)
    return {"grid": grid, "input_data": {
        "q": q, "k": k, "v": v, "g": g, "g_gamma": g_gamma, "gk": gk, "gv": gv, "o": o, "h0": h0, "ht": ht, "cu_seqlens": cu_seqlens, "scale": scale, "T": T, "B": B, "H": H, "K": K, "V": V, "BK": BK, "BV": BV, "USE_G": g is not None, "USE_G_GAMMA": g_gamma is not None, "USE_GK": gk is not None, "USE_GV": gv is not None, "REVERSE": reverse
    }}


def fn_triton(grid, input_data):
    fused_recurrent_fwd_kernel[grid](**input_data)
    return


def make_perf_case(case):
    inputs = build_inputs(case)
    data = return_args(inputs)
    return fn_triton, data
