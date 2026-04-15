from __future__ import annotations

import torch
import triton

from kernel_suite.acc_utils import assert_close_tree, clone_value
from kernel_suite.kernels.common_builders import logsigmoid, make_cu_seqlens, randn
from fla.ops.common.fused_recurrent import fused_recurrent_bwd, fused_recurrent_bwd_kernel
from fla.utils import autotune_cache_kwargs, check_shared_mem
from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets

KERNEL_NAME = "fused_recurrent_bwd"
SOURCE_PATH = "fla/ops/common/fused_recurrent.py:394"
SOURCE_TESTS = ["tests/ops/test_simple_gla.py", "tests/ops/test_gla.py"]
CASES = [
    {"name": "scalar_g_fixed", "tags": ["fixed", "scalar_g"], "shape": (4, 2048, 8, 64), "gate": "g", "dtype": torch.float16},
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
        "o": randn((b, t, h, d), torch.float32),
        "do": randn((b, t, h, d), case["dtype"]),
        "dht": randn((n, h, d, d), torch.float32),
        "initial_state": randn((n, h, d, d), torch.float32),
        "cu_seqlens": cu,
        "reverse": case.get("reverse", False),
    }


def launch(inputs):
    return fused_recurrent_bwd(q=inputs["q"], k=inputs["k"], v=inputs["v"], g=inputs["g"], o=inputs["o"], do=inputs["do"], dht=inputs["dht"], initial_state=inputs["initial_state"], reverse=inputs["reverse"], cu_seqlens=inputs["cu_seqlens"])


def reference(inputs):
    return fused_recurrent_bwd(q=inputs["q"], k=inputs["k"], v=inputs["v"], g=inputs["g"], o=inputs["o"], do=inputs["do"], dht=inputs["dht"], initial_state=inputs["initial_state"], reverse=inputs["reverse"], cu_seqlens=inputs["cu_seqlens"])


def run_accuracy_case(case):
    inputs = build_inputs(case)
    assert_close_tree(launch(clone_value(inputs)), reference(clone_value(inputs)), atol=1e-4, rtol=1e-4)


def return_args(inputs):
    g = None
    g_gamma = None
    gk = None
    gv = None
    scale = None
    q=inputs["q"]
    k=inputs["k"]
    v=inputs["v"]
    g=inputs["g"]
    o=inputs["o"]
    do=inputs["do"]
    dht=inputs["dht"]
    initial_state=inputs["initial_state"]
    reverse=inputs["reverse"]
    cu_seqlens=inputs["cu_seqlens"]
    B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    BK, BV = min(triton.next_power_of_2(K), 64), min(triton.next_power_of_2(V), 64)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

    h0 = initial_state
    dq = q.new_empty(NV, *q.shape, dtype=torch.float32)
    dk = q.new_empty(NV, *k.shape, dtype=torch.float32)
    dv = q.new_empty(NK, *v.shape, dtype=torch.float32)
    dh0 = torch.empty_like(h0) if h0 is not None else None

    dg, dgk, dgv = None, None, None
    if g is not None:
        dg = g.new_empty(NK*NV, *g.shape, dtype=torch.float32)
    if gk is not None:
        dgk = gk.new_empty(NV, *gk.shape, dtype=torch.float32)
    if gv is not None:
        dgv = gv.new_empty(NK, *gv.shape, dtype=torch.float32)

    grid = (NV, NK, N * H)
    return {"grid": grid, "input_data": {
        "q": q, "k": k, "v": v, "g": g, "g_gamma": g_gamma, "gk": gk, "gv": gv, "o": o, "h0": h0, "do": do, "dq": dq, "dk": dk, "dv": dv, "dg": dg, "dgk": dgk, "dgv": dgv, "dht": dht, "dh0": dh0, "cu_seqlens": cu_seqlens, "scale": scale, "B": B, "T": T, "H": H, "K": K, "V": V, "BK": BK, "BV": BV, "USE_G": g is not None, "USE_G_GAMMA": g_gamma is not None, "USE_GK": gk is not None, "USE_GV": gv is not None, "REVERSE": reverse
    }}


def fn_triton(grid, input_data):
    fused_recurrent_bwd_kernel[grid](**input_data)
    return


def make_perf_case(case):
    inputs = build_inputs(case)
    data = return_args(inputs)
    return fn_triton, data
