# Kernel Suite

This directory isolates the requested FLA kernels into per-kernel modules and
adds per-kernel accuracy/performance entrypoints.

## FLA structure summary

- `fla/modules/conv/triton/`: raw Triton convolution kernels plus launch code.
- `fla/ops/common/`: shared chunk/recurrent kernels reused by multiple ops.
- `fla/ops/gated_delta_rule/wy_fast.py`: WY representation forward/backward helpers.
- `fla/ops/utils/`: utility kernels such as local cumsum and triangular solve.
- `tests/modules/` and `tests/ops/`: primary source of shape/dtype/generalization cases.
- `tests/context_parallel/`: extra branches and bug-regression coverage for conv/GDN.

## Directory layout

- `kernels/<kernel>.py`: per-kernel module with source path, collected cases, launch logic.
- `tests_acc/test_<kernel>_acc.py`: run all collected accuracy cases for one kernel.
- `tests_perf/test_<kernel>_perf.py`: run all collected performance cases for one kernel.
- `acc_utils.py`: shared accuracy runner/helpers.
- `perf_utils.py`: shared performance runner/helpers.

## Notes

- For kernels with an easy mathematical reference, accuracy compares against a
  torch/reference implementation.
- For shared internal kernels whose standalone reference is expensive to
  reconstruct, accuracy compares the isolated launch path against the original
  source function using the same collected cases.
