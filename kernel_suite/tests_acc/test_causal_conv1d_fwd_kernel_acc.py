import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kernel_suite.acc_utils import run_accuracy_cases
from kernel_suite.kernels import causal_conv1d_fwd_kernel as module

if __name__ == "__main__":
    run_accuracy_cases(module)
