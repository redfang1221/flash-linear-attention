import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kernel_suite.perf_utils import run_performance_cases
from kernel_suite.kernels import solve_tril as module

if __name__ == "__main__":
    run_performance_cases(module)
