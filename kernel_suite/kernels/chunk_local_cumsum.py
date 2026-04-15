from __future__ import annotations

import torch
import triton

from kernel_suite.acc_utils import assert_close_tree, clone_value
from kernel_suite.kernels.common_builders import logsigmoid, make_cu_seqlens
from fla.ops.utils.cumsum import chunk_local_cumsum, chunk_local_cumsum_scalar, chunk_local_cumsum_vector, chunk_local_cumsum_vector_kernel, chunk_local_cumsum_scalar_kernel
from fla.utils import autotune_cache_kwargs, check_shared_mem
from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets

KERNEL_NAME = "chunk_local_cumsum"
SOURCE_PATH = "fla/ops/utils/cumsum.py:432"
SOURCE_TESTS = ["tests/ops/test_utils.py"]
CASES = [
    {"name": "fixed_scalar", "tags": ["fixed", "scalar"], "shape": (4, 2048, 8), "chunk_size": 128, "dtype": torch.float32},
    {"name": "fixed_vector", "tags": ["fixed", "vector"], "shape": (4, 2048, 8, 1024), "chunk_size": 128, "dtype": torch.float32},
    {"name": "varlen_vector", "tags": ["varlen", "vector"], "shape": (1, 2048, 2, 1024), "chunk_size": 128, "dtype": torch.float16, "cu_seqlens": [200, 312, 688, 848]},
]


def build_inputs(case):
    x = logsigmoid(case["shape"], case["dtype"])
    cu = make_cu_seqlens(case["cu_seqlens"]) if "cu_seqlens" in case else None
    return {"x": x, "chunk_size": case["chunk_size"], "cu_seqlens": cu}


def launch(inputs):
    return chunk_local_cumsum(inputs["x"], chunk_size=inputs["chunk_size"], cu_seqlens=inputs["cu_seqlens"])


def reference(inputs):
    x = inputs["x"]
    if x.dim() == 3:
        if inputs["cu_seqlens"] is None:
            return torch.cat([x[:, i:i + inputs["chunk_size"], :].float().cumsum(1) for i in range(0, x.shape[1], inputs["chunk_size"])], 1)
        ref = []
        cu = inputs["cu_seqlens"].tolist()
        for start, end in zip(cu[:-1], cu[1:], strict=False):
            ref.append(torch.cat([x[:, i:min(end, i + inputs["chunk_size"]), :].float().cumsum(1) for i in range(start, end, inputs["chunk_size"])], 1))
        return torch.cat(ref, 1)
    if inputs["cu_seqlens"] is None:
        return torch.cat([x[:, i:i + inputs["chunk_size"], :].float().cumsum(1) for i in range(0, x.shape[1], inputs["chunk_size"])], 1)
    ref = []
    cu = inputs["cu_seqlens"].tolist()
    for start, end in zip(cu[:-1], cu[1:], strict=False):
        ref.append(torch.cat([x[:, i:min(end, i + inputs["chunk_size"]), :].float().cumsum(1) for i in range(start, end, inputs["chunk_size"])], 1))
    return torch.cat(ref, 1)


def run_accuracy_case(case):
    inputs = build_inputs(case)
    actual = launch(clone_value(inputs))
    expected = reference(clone_value(inputs))
    assert_close_tree(actual, expected, atol=1e-3, rtol=1e-3)


def return_args(inputs):
    actual = launch(clone_value(inputs))
    reverse = False
    scale = None
    cu_seqlens = None
    head_first = False
    output_dtype = torch.float
    chunk_indices = None
    g = inputs["x"]
    chunk_size = inputs["chunk_size"]
    cu_seqlens = inputs["cu_seqlens"]
    if cu_seqlens is not None:
        assert g.shape[0] == 1, "Only batch size 1 is supported when cu_seqlens are provided"
    if len(g.shape) == 3:
        if head_first:
            B, H, T = g.shape
        else:
            B, T, H = g.shape
        assert chunk_size == 2**(chunk_size.bit_length()-1), "chunk_size must be a power of 2"
        BT = chunk_size
        if chunk_indices is None and cu_seqlens is not None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
        g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
        grid = (NT, B * H)
        return {"grid": grid, "input_data": {
            "s": g_org, "o": g, "scale": scale, "cu_seqlens": cu_seqlens, "chunk_indices": chunk_indices, "T": T, "B": B, "H": H, "BT": BT, "HEAD_FIRST": head_first, "REVERSE": reverse
        }}
    elif len(g.shape) == 4:
        print(f"Best Config: {chunk_local_cumsum_vector_kernel.fn.best_config}")
        config_obj = chunk_local_cumsum_vector_kernel.fn.best_config
        params = config_obj.kwargs
        BS = params.get("BS")
        if head_first:
            B, H, T, S = g.shape
        else:
            B, T, H, S = g.shape
        BT = chunk_size
        if chunk_indices is None and cu_seqlens is not None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
        assert chunk_size == 2**(chunk_size.bit_length()-1), "chunk_size must be a power of 2"

        g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
        def grid(meta): return (triton.cdiv(S, BS), NT, B * H)
        return {"grid": grid, "input_data": {
            "s": g_org, "o": g, "scale": scale, "cu_seqlens": cu_seqlens, "chunk_indices": chunk_indices, "T": T, "B": B, "H": H, "S": S, "BT": BT, "HEAD_FIRST": head_first, "REVERSE": reverse
        }}
    else:
        raise ValueError(
            f"Unsupported input shape {g.shape}, "
            f"which should be (B, T, H, D) if `head_first=False` "
            f"or (B, H, T, D) otherwise",
        )


def fn_triton(grid, input_data):
    if 'S' not in input_data.keys():
        chunk_local_cumsum_scalar_kernel[grid](**input_data)
    else:
        chunk_local_cumsum_vector_kernel[grid](**input_data)


def make_perf_case(case):
    inputs = build_inputs(case)
    data = return_args(inputs)
    return fn_triton, data