from __future__ import annotations

from kernel_suite.kernels.layer_norm_gated_fwd import build_inputs, launch, run_accuracy_case, make_perf_case

KERNEL_NAME = "layer_norm_gated_fwd_kernel"
SOURCE_PATH = "fla/modules/fused_norm_gate.py:35"
SOURCE_TESTS = ["tests/modules/test_layernorm_gated.py"]
CASES = [
    {"name": "kernel_path_smallD", "tags": ["smallD"], "B": 2, "H": 2, "T": 512, "D": 128, "activation": "silu", "weight": True, "bias": True, "rms": False},
    {"name": "kernel1_path_largeD", "tags": ["largeD"], "B": 2, "H": 2, "T": 2048, "D": 1200, "activation": "sigmoid", "weight": True, "bias": False, "rms": False},
]
