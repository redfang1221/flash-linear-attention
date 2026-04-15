from __future__ import annotations

import os
from typing import Callable

import torch
import triton


def perf_test(func: Callable[[], object], arg: dict) -> float:
    warmup_ms = int(arg.get("warmup_ms", os.environ.get("FLA_KERNEL_BENCH_WARMUP_MS", 25)))
    rep_ms = int(arg.get("rep_ms", os.environ.get("FLA_KERNEL_BENCH_REP_MS", 100)))
    torch.cuda.synchronize()
    ms = triton.testing.do_bench(func, warmup=warmup_ms, rep=rep_ms, quantiles=[0.5, 0.2, 0.8])[0]
    us = float(ms) * 1000
    print(
        f"[PERF] {arg['kernel']} | {arg['name']} | "
        f"Avg Latency: {us:.2f} us"
    )
    return float(ms)


def run_performance_cases(module) -> None:
    print(f"Running performance cases for {module.KERNEL_NAME}")
    for case in module.CASES:
        func, meta = module.make_perf_case(case)
        perf_test(func, meta)
