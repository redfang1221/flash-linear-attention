from __future__ import annotations

import torch
import triton

from kernel_suite.acc_utils import assert_close_tree, clone_value
from kernel_suite.kernels.common_builders import logsigmoid, make_cu_seqlens, randn
from fla.utils import check_shared_mem
from fla.ops.common.chunk_h import chunk_bwd_dh, chunk_bwd_kernel_dh
from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets

KERNEL_NAME = "chunk_bwd_dh"
SOURCE_PATH = "fla/ops/common/chunk_h.py"
SOURCE_TESTS = ["tests/ops/test_simple_gla.py", "tests/ops/test_gla.py", "fla/ops/gsa/chunk.py"]
CASES = [
    {"name": "scalar_g_fixed", "tags": ["fixed", "scalar_g"], "shape": (4, 2048, 8, 64), "g_mode": "g", "dtype": torch.float16},
    {"name": "vector_gk_varlen", "tags": ["varlen", "vector_gk"], "shape": (1, 2048, 4, 128), "g_mode": "gk", "dtype": torch.float16, "cu_seqlens": [1, 99, 200, 900, 848]},
]


def build_inputs(case):
    b, t, h, d = case["shape"]
    cu = make_cu_seqlens(case["cu_seqlens"]) if "cu_seqlens" in case else None
    n = len(case["cu_seqlens"]) if "cu_seqlens" in case else b
    return {
        "q": randn((b, t, h, d), case["dtype"]),
        "k": randn((b, t, h, d), case["dtype"]),
        "v": randn((b, t, h, d), case["dtype"]),
        "do": randn((b, t, h, d), case["dtype"]),
        "h0": randn((n, h, d, d), torch.float32),
        "dht": randn((n, h, d, d), torch.float32),
        "g": logsigmoid((b, t, h), torch.float32) if case["g_mode"] == "g" else None,
        "gk": logsigmoid((b, t, h, d), torch.float32) if case["g_mode"] == "gk" else None,
        "cu_seqlens": cu,
        "scale": d ** -0.5,
    }


def launch(inputs):
    return chunk_bwd_dh(q=inputs["q"], k=inputs["k"], v=inputs["v"], do=inputs["do"], h0=inputs["h0"], dht=inputs["dht"], scale=inputs["scale"], g=inputs["g"], gk=inputs["gk"], cu_seqlens=inputs["cu_seqlens"])


def reference(inputs):
    return chunk_bwd_dh(q=inputs["q"], k=inputs["k"], v=inputs["v"], do=inputs["do"], h0=inputs["h0"], dht=inputs["dht"], scale=inputs["scale"], g=inputs["g"], gk=inputs["gk"], cu_seqlens=inputs["cu_seqlens"])


def run_accuracy_case(case):
    inputs = build_inputs(case)
    assert_close_tree(launch(clone_value(inputs)), reference(clone_value(inputs)), atol=1e-4, rtol=1e-4)


def return_args(inputs):
    q=inputs["q"]
    k=inputs["k"]
    v=inputs["v"]
    do=inputs["do"]
    h0=inputs["h0"]
    dht=inputs["dht"]
    scale=inputs["scale"]
    g=inputs["g"]
    gk=inputs["gk"]
    g_gamma = None
    gv=None
    cu_seqlens=inputs["cu_seqlens"]
    chunk_size = 64
    split_size = None
    states_in_fp32 = False
    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]
    BT = chunk_size
    BS = BT if split_size is None else split_size
    # N: the actual number of sequences in the batch with either equal or variable lengths
    # NG: number of groups in GQA
    if cu_seqlens is None:
        N, NS, split_offsets = B, triton.cdiv(T, BS), None
    else:
        split_offsets = prepare_chunk_offsets(cu_seqlens, BS)
        N, NS = len(cu_seqlens) - 1, split_offsets[-1].item()
    NG = HQ // H
    dh = k.new_empty(B, NS, HQ, K, V, dtype=k.dtype if not states_in_fp32 else torch.float)
    dh0 = torch.empty_like(h0, dtype=torch.float) if h0 is not None else None
    def grid(meta): return (triton.cdiv(K, meta['BK']), triton.cdiv(V, meta['BV']), N * H)
    return {"grid": grid,
    "input_data": {
        "q": q, "g": g, "g_gamma": g_gamma, "gk": gk, "gv": gv, "do": do, "dh": dh, "dht": dht, "dh0": dh0, "cu_seqlens": cu_seqlens, "split_offsets": split_offsets, "scale": scale, "T": T, "HQ": HQ, "H": H, "K": K, "V": V, "BT": BT, "BS": BS, "NG": NG, "USE_G": g is not None, "USE_G_GAMMA": g_gamma is not None, "USE_GK": gk is not None, "USE_GV": gv is not None
    }}


def fn_triton(grid, input_data):
    chunk_bwd_kernel_dh[grid](**input_data)
    return


def make_perf_case(case):
    inputs = build_inputs(case)
    data = return_args(inputs)
    return fn_triton, data
