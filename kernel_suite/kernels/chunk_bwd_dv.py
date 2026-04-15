from __future__ import annotations

import torch
import triton

from kernel_suite.acc_utils import assert_close_tree, clone_value
from kernel_suite.kernels.common_builders import logsigmoid, randn
from fla.utils import autotune_cache_kwargs, check_shared_mem
from fla.ops.common.chunk_o import chunk_bwd_dv, chunk_bwd_kernel_dv

KERNEL_NAME = "chunk_bwd_dv"
SOURCE_PATH = "fla/ops/common/chunk_o.py"
SOURCE_TESTS = ["tests/ops/test_simple_gla.py"]
CASES = [
    {"name": "simple_gla_scalar_g", "tags": ["fixed", "scalar_g"], "shape": (4, 2048, 8, 64), "dtype": torch.float16},
]


# @triton.heuristics({
#     'USE_G': lambda args: args['g'] is not None,
#     'USE_G_GAMMA': lambda args: args['g_gamma'] is not None,
#     'USE_A': lambda args: args['A'] is not None,
#     'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
# })
# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=num_warps, num_stages=num_stages)
#         for num_warps in NUM_WARPS
#         for num_stages in [2, 3, 4]
#     ],
#     key=['H', 'HV', 'K', 'V', 'BT', 'BK', 'BV', 'USE_G'],
#     **autotune_cache_kwargs,
# )
# @triton.jit(do_not_specialize=['T'])
# def chunk_bwd_kernel_dv_local(
#     q,
#     k,
#     g,
#     g_gamma,
#     A,
#     do,
#     dv,
#     cu_seqlens,
#     chunk_indices,
#     scale,
#     T,
#     H: tl.constexpr,
#     HV: tl.constexpr,
#     K: tl.constexpr,
#     V: tl.constexpr,
#     BT: tl.constexpr,
#     BK: tl.constexpr,
#     BV: tl.constexpr,
#     USE_G: tl.constexpr,
#     USE_G_GAMMA: tl.constexpr,
#     USE_EXP2: tl.constexpr,
#     USE_A: tl.constexpr,
#     IS_VARLEN: tl.constexpr,
# ):
#     i_t, i_bh = tl.program_id(0), tl.program_id(1)
#     i_b, i_h = i_bh // HV, i_bh % HV
#     if IS_VARLEN:
#         i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
#         bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
#         T = eos - bos
#     else:
#         bos, eos = i_b * T, i_b * T + T

#     # offset calculation
#     q += (bos * H + i_h // (HV // H)) * K
#     k += (bos * H + i_h // (HV // H)) * K
#     do += (bos * HV + i_h) * V
#     dv += (bos * HV + i_h) * V

#     if USE_A:
#         p_A = tl.make_block_ptr(A + (bos * HV + i_h) * BT, (BT, T), (1, HV*BT), (0, i_t * BT), (BT, BT), (0, 1))
#         b_A = tl.load(p_A, boundary_check=(0, 1))
#     else:
#         if USE_G:
#             g += bos * HV + i_h
#             p_g = tl.make_block_ptr(g, (T,), (HV,), (i_t * BT,), (BT,), (0,))
#             b_g = tl.load(p_g, boundary_check=(0,))
#         if USE_G_GAMMA:
#             b_gamma = tl.load(g_gamma + i_h)
#             b_g = b_gamma * (tl.arange(0, BT) + 1)

#         b_A = tl.zeros([BT, BT], dtype=tl.float32)
#         for i_k in range(tl.cdiv(K, BK)):
#             p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
#             p_q = tl.make_block_ptr(q, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))

#             b_k = tl.load(p_k, boundary_check=(0, 1))
#             b_q = tl.load(p_q, boundary_check=(0, 1))
#             b_A += tl.dot(b_k, b_q) * scale
#         if USE_G or USE_G_GAMMA:
#             if USE_EXP2:
#                 b_A *= exp2(b_g[None, :] - b_g[:, None])
#             else:
#                 b_A *= exp(b_g[None, :] - b_g[:, None])

#     o_t = i_t * BT + tl.arange(0, BT)
#     m_t = o_t < T
#     m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t)
#     b_A = tl.where(m_A, b_A, 0).to(do.dtype.element_ty)

#     for i_v in range(tl.cdiv(V, BV)):
#         p_do = tl.make_block_ptr(do, (T, V), (HV*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
#         p_dv = tl.make_block_ptr(dv, (T, V), (HV*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
#         b_do = tl.load(p_do, boundary_check=(0, 1))
#         b_dv = tl.dot(b_A.to(b_do.dtype), b_do)
#         tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


# def chunk_bwd_dv(
#     q: torch.Tensor,
#     k: torch.Tensor,
#     do: torch.Tensor,
#     dh: torch.Tensor,
#     g: torch.Tensor | None = None,
#     g_gamma: torch.Tensor | None = None,
#     scale: float | None = None,
#     cu_seqlens: torch.LongTensor | None = None,
#     chunk_size: int = 64,
#     chunk_indices: torch.LongTensor | None = None,
#     use_exp2: bool = False,
# ) -> torch.Tensor:
#     B, T, H, K, V, HV = *k.shape, do.shape[-1], do.shape[2]
#     BT = chunk_size
#     if chunk_indices is None and cu_seqlens is not None:
#         chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
#     # H100 can have larger block size
#     if check_shared_mem('hopper', k.device.index):
#         CONST_TILING = 128
#     elif check_shared_mem('ada', k.device.index):
#         CONST_TILING = 64
#     else:
#         CONST_TILING = 32
#     BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
#     BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
#     NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
#     NV = triton.cdiv(V, BV)
#     if scale is None:
#         scale = k.shape[-1] ** -0.5

#     dv = torch.empty_like(do)
#     grid = (NV, NT, B * HV)
#     chunk_bwd_kernel_dv[grid](
#         q=q,
#         k=k,
#         g=g,
#         g_gamma=g_gamma,
#         do=do,
#         dv=dv,
#         dh=dh,
#         cu_seqlens=cu_seqlens,
#         chunk_indices=chunk_indices,
#         scale=scale,
#         T=T,
#         H=H,
#         HV=HV,
#         K=K,
#         V=V,
#         BT=BT,
#         BK=BK,
#         BV=BV,
#         USE_EXP2=use_exp2,
#     )
#     return dv


def build_inputs(case):
    b, t, h, d = case["shape"]
    return {
        "q": randn((b, t, h, d), case["dtype"]),
        "k": randn((b, t, h, d), case["dtype"]),
        "do": randn((b, t, h, d), case["dtype"]),
        "dh": randn((b, (t + 63) // 64, h, d, d), case["dtype"]),
        "g": logsigmoid((b, t, h), torch.float32),
        "scale": d ** -0.5,
    }


def launch(inputs):
    return chunk_bwd_dv(q=inputs["q"], k=inputs["k"], do=inputs["do"], dh=inputs["dh"], g=inputs["g"], scale=inputs["scale"])


def reference(inputs):
    return chunk_bwd_dv(q=inputs["q"], k=inputs["k"], do=inputs["do"], dh=inputs["dh"], g=inputs["g"], scale=inputs["scale"])


def run_accuracy_case(case):
    inputs = build_inputs(case)
    assert_close_tree(launch(clone_value(inputs)), reference(clone_value(inputs)), atol=1e-4, rtol=1e-4)


def return_args(inputs):
    q=inputs["q"]
    k=inputs["k"]
    g=inputs["g"]
    g_gamma=None
    do=inputs["do"]
    dh=inputs["dh"]
    cu_seqlens=None
    chunk_indices=None
    scale=inputs["scale"]
    chunk_size=64
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
    NV = triton.cdiv(V, BV)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    dv = torch.empty_like(do)
    grid = (NV, NT, B * HV)
    return {"grid": grid,
    "input_data": {
        "q": q, "k": k, "g": g, "g_gamma": g_gamma, "do": do, "dv": dv, "dh": dh, "cu_seqlens": cu_seqlens, "chunk_indices": chunk_indices, "scale": scale, "T": T, "H": H, "HV": HV, "K": K, "V": V, "BT": BT, "BK": BK, "BV": BV, "USE_EXP2": use_exp2
    }}


def fn_triton(grid, input_data):
    chunk_bwd_kernel_dv[grid](**input_data)
    return


def make_perf_case(case):
    inputs = build_inputs(case)
    data = return_args(inputs)
    return fn_triton, data
