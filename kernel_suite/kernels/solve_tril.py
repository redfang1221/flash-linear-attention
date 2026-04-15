from __future__ import annotations

import torch
import triton
import math

from kernel_suite.acc_utils import assert_close_tree, clone_value
from kernel_suite.kernels.common_builders import make_cu_seqlens, normalize, randn
from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.utils.solve_tril import solve_tril, solve_tril_16x16_kernel, merge_16x16_to_32x32_inverse_kernel, merge_16x16_to_64x64_inverse_kernel
from fla.utils import autotune_cache_kwargs, check_shared_mem
from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.utils import autotune_cache_kwargs, get_multiprocessor_count, input_guard
from fla.utils import IS_TMA_SUPPORTED, autotune_cache_kwargs, input_guard


KERNEL_NAME = "solve_tril"
SOURCE_PATH = "fla/ops/utils/solve_tril.py"
SOURCE_TESTS = ["tests/ops/test_solve_tril.py"]
CASES = [
    {"name": "fixed_bt16", "tags": ["fixed"], "B": 1, "T": 63, "H": 1, "D": 64, "chunk_size": 16, "dtype": torch.float32},
    {"name": "fixed_bt64", "tags": ["fixed"], "B": 4, "T": 2048, "H": 8, "D": 64, "chunk_size": 64, "dtype": torch.float32},
    {"name": "varlen_bt32", "tags": ["varlen"], "H": 4, "D": 128, "chunk_size": 32, "cu_seqlens": [200, 312, 688, 848], "dtype": torch.bfloat16},
]


def build_inputs(case):
    if "cu_seqlens" not in case:
        k = normalize(randn((case["B"], case["H"], case["T"], case["D"]), case["dtype"]))
        pad = (case["chunk_size"] - case["T"] % case["chunk_size"]) % case["chunk_size"]
        kp = torch.nn.functional.pad(k, (0, 0, 0, pad, 0, 0, 0, 0)).reshape(case["B"], case["H"], -1, case["chunk_size"], case["D"])
        A = (kp @ kp.transpose(-1, -2)).tril(-1).reshape(case["B"], case["H"], -1, case["chunk_size"])[:, :, : case["T"], :].transpose(1, 2)
        return {"A": A, "cu_seqlens": None}
    cu = make_cu_seqlens(case["cu_seqlens"])
    t = int(cu[-1].item())
    k = normalize(randn((1, t, case["H"], case["D"]), case["dtype"]))
    beta = torch.randn((1, t, case["H"]), device="npu", dtype=case["dtype"]).sigmoid()
    A = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, cu_seqlens=cu, chunk_size=case["chunk_size"])
    return {"A": A, "cu_seqlens": cu}


def launch(inputs):
    return solve_tril(inputs["A"], cu_seqlens=inputs["cu_seqlens"])


def reference(inputs):
    A = inputs["A"]
    ref = torch.zeros_like(A)
    if inputs["cu_seqlens"] is None:
        eye = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)[None, None, ...]
        return torch.inverse(A.transpose(1, 2) + eye).transpose(1, 2)
    cu = inputs["cu_seqlens"].tolist()
    bt = A.shape[-1]
    for i in range(len(cu) - 1):
        for j in range(cu[i], cu[i + 1], bt):
            size = min(bt, cu[i + 1] - j)
            ref[:, j:j + size, :, :size] = torch.inverse(
                A[:, j:j + size, :, :size].transpose(1, 2)
                + torch.eye(size, device=A.device, dtype=A.dtype)[None, None, ...]
            ).transpose(1, 2)
    return ref


def run_accuracy_case(case):
    inputs = build_inputs(case)
    actual = launch(clone_value(inputs))
    expected = reference(clone_value(inputs))
    assert_close_tree(actual, expected, atol=1e-4, rtol=1e-4)


def return_args(inputs):
    cu_seqlens: torch.Tensor = None
    chunk_indices: torch.LongTensor = None
    output_dtype: torch.dtype = torch.float
    A = inputs["A"]
    cu_seqlens=inputs["cu_seqlens"]
    assert A.shape[-1] in [16, 32, 64]
    output_dtype = A.dtype if output_dtype is None else output_dtype

    B, T, H, BT = A.shape
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)

    Ai = torch.zeros_like(A, dtype=output_dtype)
    if BT == 16:
        merge_fn = solve_tril_16x16_kernel
    elif BT == 32:
        merge_fn = merge_16x16_to_32x32_inverse_kernel
    elif BT == 64:
        merge_fn = merge_16x16_to_64x64_inverse_kernel
    return {"grid": (NT, B * H), "input_data": {
        "A": A, "Ai": Ai, "cu_seqlens": cu_seqlens, "chunk_indices": chunk_indices, "T": T, "H": H, "BT": BT, "USE_TMA": IS_TMA_SUPPORTED
    }}

def fn_triton(grid, input_data):
    if input_data["BT"] == 16:
        solve_tril_16x16_kernel[grid](**input_data)
    elif input_data["BT"] == 32:
        merge_16x16_to_32x32_inverse_kernel[grid](**input_data)
    elif input_data["BT"] == 64:
        merge_16x16_to_64x64_inverse_kernel[grid](**input_data)
    return


def make_perf_case(case):
    inputs = build_inputs(case)
    data = return_args(inputs)
    return fn_triton, data