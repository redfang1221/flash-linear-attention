from __future__ import annotations

import torch
import triton

from kernel_suite.acc_utils import assert_close_tree, clone_value
from kernel_suite.kernels.common_builders import beta, logsigmoid, make_cu_seqlens, normalize, randn
from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd, chunk_scaled_dot_kkt_fwd_kernel
from fla.utils import autotune_cache_kwargs, check_shared_mem
from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets

KERNEL_NAME = "chunk_scaled_dot_kkt_fwd"
SOURCE_PATH = "fla/ops/common/chunk_scaled_dot_kkt.py"
SOURCE_TESTS = ["tests/ops/test_solve_tril.py", "fla/ops/gated_delta_rule/chunk.py", "fla/ops/delta_rule/wy_fast.py"]
CASES = [
    {"name": "fixed_bt16", "tags": ["fixed"], "shape": (2, 500, 4, 64), "hv": 4, "chunk_size": 16, "dtype": torch.bfloat16},
    {"name": "fixed_bt64_with_g", "tags": ["fixed", "gate"], "shape": (2, 1000, 2, 128), "hv": 8, "chunk_size": 64, "dtype": torch.float16, "with_g": True},
    {"name": "varlen_bt32", "tags": ["varlen"], "shape": (1, 2048, 4, 128), "hv": 4, "chunk_size": 32, "dtype": torch.bfloat16, "cu_seqlens": [200, 312, 688, 848]},
]


def build_inputs(case):
    b, t, h, d = case["shape"]
    k = normalize(randn((b, t, h, d), case["dtype"]))
    g = logsigmoid((b, t, case["hv"])) if case.get("with_g") else None
    bt = beta((b, t, case["hv"]), case["dtype"])
    cu = make_cu_seqlens(case["cu_seqlens"]) if "cu_seqlens" in case else None
    return {"k": k, "g": g, "beta": bt, "cu_seqlens": cu, "chunk_size": case["chunk_size"]}


def launch(inputs):
    return chunk_scaled_dot_kkt_fwd(
        k=inputs["k"],
        g=inputs["g"],
        beta=inputs["beta"],
        cu_seqlens=inputs["cu_seqlens"],
        chunk_size=inputs["chunk_size"],
    )


def reference(inputs):
    k = inputs["k"].float()
    beta = inputs["beta"].float()
    g = inputs["g"].float() if inputs["g"] is not None else None
    b, t, h, d = inputs["k"].shape
    hv = beta.shape[2]
    out = torch.zeros((b, t, hv, inputs["chunk_size"]), device=k.device, dtype=torch.float32)
    cu = [0, t] if inputs["cu_seqlens"] is None else inputs["cu_seqlens"].tolist()
    group = hv // h
    for bi in range(b):
        for start, end in zip(cu[:-1], cu[1:], strict=False):
            for bos in range(start, end, inputs["chunk_size"]):
                eos = min(end, bos + inputs["chunk_size"])
                local = eos - bos
                kk = k[bi, bos:eos]
                for hv_idx in range(hv):
                    head = hv_idx // group
                    vec = kk[:, head]
                    scores = vec @ vec.transpose(0, 1)
                    scores = torch.tril(scores, diagonal=0)
                    scores = scores * beta[bi, bos:eos, hv_idx][None, :]
                    if g is not None:
                        delta = g[bi, bos:eos, hv_idx][None, :] - g[bi, bos:eos, hv_idx][:, None]
                        scores = scores * torch.exp(delta).tril()
                    out[bi, bos:eos, hv_idx, :local] = scores[:, :local]
    return out


def run_accuracy_case(case):
    inputs = build_inputs(case)
    actual = launch(clone_value(inputs))
    expected = reference(clone_value(inputs))
    assert_close_tree(actual[:, :, :, : expected.shape[-1]], expected, atol=3e-2, rtol=3e-2)


def return_args(inputs):
    k=inputs["k"]
    g=inputs["g"]
    beta=inputs["beta"]
    cu_seqlens=inputs["cu_seqlens"]
    chunk_size=inputs["chunk_size"]
    B, T, H, K, HV = *k.shape, beta.shape[2]
    chunk_indices=None
    output_dtype=torch.float32
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    A = torch.empty(B, T, HV, BT, device=k.device, dtype=output_dtype)
    return {"grid": (NT, B * HV), "input_data": {
        "k": k, "g": g, "beta": beta, "A": A, "cu_seqlens": cu_seqlens, "chunk_indices": chunk_indices, "T": T, "H": H, "HV": HV, "K": K, "BT": BT
    }}


def fn_triton(grid, input_data):
    chunk_scaled_dot_kkt_fwd_kernel[grid](**input_data)
    return


def make_perf_case(case):
    inputs = build_inputs(case)
    data = return_args(inputs)
    return fn_triton, data

