from __future__ import annotations

import torch
import triton

from kernel_suite.acc_utils import assert_close_tree, clone_value
from kernel_suite.kernels.common_builders import logsigmoid, make_cu_seqlens, randn
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64
from fla.utils import autotune_cache_kwargs, check_shared_mem
from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets

KERNEL_NAME = "chunk_gated_delta_rule_bwd_dhu"
SOURCE_PATH = "fla/ops/common/chunk_delta_h.py"
SOURCE_TESTS = ["tests/ops/test_gated_delta.py", "tests/ops/test_delta.py", "fla/ops/kda/chunk_bwd.py"]
CASES = [
    {"name": "fixed_gqa", "tags": ["fixed", "gqa", "gva"], "shape": (2, 1024, 2, 128), "hv": 4, "dtype": torch.float16},
    {"name": "varlen_gk", "tags": ["varlen", "gk"], "shape": (1, 2000, 4, 100), "hv": 4, "dtype": torch.float16, "cu_seqlens": [15, 85, 200, 300, 1200, 200]},
]


def build_inputs(case):
    b, t, h, d = case["shape"]
    hv = case["hv"]
    cu = make_cu_seqlens(case["cu_seqlens"]) if "cu_seqlens" in case else None
    n = len(case["cu_seqlens"]) if "cu_seqlens" in case else b
    return {
        "q": randn((b, t, h, d), case["dtype"]),
        "k": randn((b, t, h, d), case["dtype"]),
        "w": randn((b, t, hv, d), case["dtype"]),
        "do": randn((b, t, hv, d), case["dtype"]),
        "dv": randn((b, t, hv, d), case["dtype"]),
        "g": logsigmoid((b, t, hv), torch.float32) if "cu_seqlens" not in case else None,
        "gk": logsigmoid((b, t, hv, d), torch.float32) if "cu_seqlens" in case else None,
        "h0": randn((n, hv, d, d), torch.float32),
        "dht": randn((n, hv, d, d), torch.float32),
        "cu_seqlens": cu,
        "scale": d ** -0.5,
    }


def launch(inputs):
    return chunk_gated_delta_rule_bwd_dhu(q=inputs["q"], k=inputs["k"], w=inputs["w"], do=inputs["do"], dv=inputs["dv"], g=inputs["g"], gk=inputs["gk"], h0=inputs["h0"], dht=inputs["dht"], scale=inputs["scale"], cu_seqlens=inputs["cu_seqlens"])


def reference(inputs):
    return chunk_gated_delta_rule_bwd_dhu(q=inputs["q"], k=inputs["k"], w=inputs["w"], do=inputs["do"], dv=inputs["dv"], g=inputs["g"], gk=inputs["gk"], h0=inputs["h0"], dht=inputs["dht"], scale=inputs["scale"], cu_seqlens=inputs["cu_seqlens"])


def run_accuracy_case(case):
    inputs = build_inputs(case)
    assert_close_tree(launch(clone_value(inputs)), reference(clone_value(inputs)), atol=1e-4, rtol=1e-4)


def return_args(inputs):
    q=inputs["q"]
    k=inputs["k"]
    w=inputs["w"]
    do=inputs["do"]
    dv=inputs["dv"]
    g=inputs["g"]
    gk=inputs["gk"]
    h0=inputs["h0"]
    dht=inputs["dht"]
    scale=inputs["scale"]
    cu_seqlens=inputs["cu_seqlens"]
    chunk_size=64
    chunk_indices=None
    use_exp2=False
    transpose_state_layout=False
    B, T, H, K, V, HV = *q.shape, do.shape[-1], do.shape[2]
    # N: the actual number of sequences in the batch with either equal or variable lengths
    BT = 64
    assert K <= 256, "current kernel does not support head dimension being larger than 256."
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)

    if transpose_state_layout:
        dh = q.new_empty(B, NT, HV, V, K)
    else:
        dh = q.new_empty(B, NT, HV, K, V)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    dv2 = torch.empty_like(dv)
    def grid(meta): return (triton.cdiv(V, meta['BV']), N*HV)
    return {"grid": grid, "input_data": {
        "q": q, "k": k, "w": w, "g": g, "gk": gk, "dht": dht, "dh0": dh0, "do": do, "dh": dh, "dv": dv, "dv2": dv2, "cu_seqlens": cu_seqlens, "chunk_offsets": chunk_offsets, "scale": scale, "T": T, "H": H, "HV": HV, "K": K, "V": V, "BT": BT, "USE_EXP2": use_exp2, "TRANSPOSE_STATE": transpose_state_layout
    }}


def fn_triton(grid, input_data):
    chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64[grid](**input_data)
    return


def make_perf_case(case):
    inputs = build_inputs(case)
    data = return_args(inputs)
    return fn_triton, data