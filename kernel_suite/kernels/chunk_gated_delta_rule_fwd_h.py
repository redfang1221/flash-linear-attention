from __future__ import annotations

import torch
import triton

from kernel_suite.acc_utils import assert_close_tree, clone_value
from kernel_suite.kernels.common_builders import logsigmoid, make_cu_seqlens, randn
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h, chunk_gated_delta_rule_fwd_kernel_h_blockdim64
from fla.utils import autotune_cache_kwargs, check_shared_mem
from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets

KERNEL_NAME = "chunk_gated_delta_rule_fwd_h"
SOURCE_PATH = "fla/ops/common/chunk_delta_h.py"
SOURCE_TESTS = ["tests/ops/test_gated_delta.py", "tests/ops/test_delta.py", "fla/ops/kda/chunk_fwd.py"]
CASES = [
    {"name": "fixed_gqa", "tags": ["fixed", "gqa", "gva"], "shape": (2, 1024, 2, 128), "hv": 4, "dtype": torch.float16},
    {"name": "transpose_state", "tags": ["fixed", "transpose"], "shape": (4, 2048, 8, 64), "hv": 8, "dtype": torch.float16, "transpose": True},
    {"name": "varlen_gk", "tags": ["varlen", "gk"], "shape": (1, 2000, 4, 100), "hv": 4, "dtype": torch.float16, "cu_seqlens": [15, 85, 200, 300, 1200, 200]},
]


def build_inputs(case):
    b, t, h, d = case["shape"]
    hv = case["hv"]
    cu = make_cu_seqlens(case["cu_seqlens"]) if "cu_seqlens" in case else None
    n = len(case["cu_seqlens"]) if "cu_seqlens" in case else b
    state = randn((n, hv, d, d), torch.float32)
    if case.get("transpose"):
        state = state.transpose(-1, -2).contiguous()
    return {
        "k": randn((b, t, h, d), case["dtype"]),
        "w": randn((b, t, hv, d), case["dtype"]),
        "u": randn((b, t, hv, d), case["dtype"]),
        "g": logsigmoid((b, t, hv), torch.float32) if "cu_seqlens" not in case else None,
        "gk": logsigmoid((b, t, hv, d), torch.float32) if "cu_seqlens" in case else None,
        "initial_state": state,
        "cu_seqlens": cu,
        "transpose_state_layout": case.get("transpose", False),
    }


def launch(inputs):
    return chunk_gated_delta_rule_fwd_h(k=inputs["k"], w=inputs["w"], u=inputs["u"], g=inputs["g"], gk=inputs["gk"], initial_state=inputs["initial_state"], output_final_state=True, cu_seqlens=inputs["cu_seqlens"], transpose_state_layout=inputs["transpose_state_layout"])


def reference(inputs):
    return chunk_gated_delta_rule_fwd_h(k=inputs["k"], w=inputs["w"], u=inputs["u"], g=inputs["g"], gk=inputs["gk"], initial_state=inputs["initial_state"], output_final_state=True, cu_seqlens=inputs["cu_seqlens"], transpose_state_layout=inputs["transpose_state_layout"])


def run_accuracy_case(case):
    inputs = build_inputs(case)
    assert_close_tree(launch(clone_value(inputs)), reference(clone_value(inputs)), atol=1e-4, rtol=1e-4)


def return_args(inputs):
    k=inputs["k"]
    w=inputs["w"]
    u=inputs["u"]
    g=inputs["g"]
    gk=inputs["gk"]
    initial_state=inputs["initial_state"]
    output_final_state=True
    chunk_size=64
    save_new_value=True
    cu_seqlens=inputs["cu_seqlens"]
    cu_seqlens_cpu=None
    chunk_indices=None
    use_exp2=False
    transpose_state_layout=inputs["transpose_state_layout"]
    B, T, H, K, V, HV = *k.shape, u.shape[-1], u.shape[2]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."
    if transpose_state_layout:
        h = k.new_empty(B, NT, HV, V, K)
        final_state = k.new_zeros(N, HV, V, K, dtype=torch.float32) if output_final_state else None
    else:
        h = k.new_empty(B, NT, HV, K, V)
        final_state = k.new_zeros(N, HV, K, V, dtype=torch.float32) if output_final_state else None
    v_new = torch.empty_like(u) if save_new_value else None
    def grid(meta): return (triton.cdiv(V, meta['BV']), N*HV)
    return {"grid": grid, "input_data": {
        "k": k, "v": u, "w": w, "v_new": v_new, "g": g, "gk": gk, "h": h, "h0": initial_state, "ht": final_state, "cu_seqlens": cu_seqlens, "chunk_offsets": chunk_offsets, "T": T, "H": H, "HV": HV, "K": K, "V": V, "BT": BT, "USE_EXP2": use_exp2, "TRANSPOSE_STATE": transpose_state_layout
    }}


def fn_triton(grid, input_data):
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](**input_data)
    return


def make_perf_case(case):
    inputs = build_inputs(case)
    data = return_args(inputs)
    return fn_triton, data
