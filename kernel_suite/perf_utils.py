from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import torch
import torch_npu
import triton


def configure_triton_dump(kernel_name: str, case: dict) -> str:
    case_name = case["name"]
    dump_dir = f"./{kernel_name}_{case_name}_cache"
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    os.environ["TRITON_DEBUG"] = "1"
    os.environ["TRITON_KERNEL_DUMP"] = "1"
    os.environ["TRITON_DUMP_DIR"] = dump_dir
    Path(dump_dir).mkdir(parents=True, exist_ok=True)
    return dump_dir


def perf_test(fn_triton, args, save_path="./result_dir"):
    experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False
        )
    with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.NPU],
            with_stack=False,
            record_shapes=False,
            profile_memory=False,
            schedule=torch_npu.profiler.schedule(wait=1,
                                                warmup=1,
                                                active=30,
                                                repeat=1,
                                                skip_first=1),
            experimental_config=experimental_config,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(save_path)
    ) as prof:
        for i in range(30):
            fn_triton(**args)
            torch.npu.synchronize()
            prof.step()


def run_performance_cases(module) -> None:
    print(f"Running performance cases for {module.KERNEL_NAME}")
    save_path = str(module.KERNEL_NAME) + "_perf"
    for case in module.CASES:
        dump_dir = configure_triton_dump(module.KERNEL_NAME, case)
        print(f"case={case['name']} TRITON_DUMP_DIR={dump_dir}")
        func, data = module.make_perf_case(case)
        perf_test(func, data, save_path)
