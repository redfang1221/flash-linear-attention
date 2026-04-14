#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if ! command -v pytest >/dev/null 2>&1; then
  echo "pytest is required but was not found in PATH." >&2
  exit 1
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# Run all repository-provided accuracy/generalization cases that cover the
# requested kernels directly or indirectly through end-to-end operator tests.
#
# Coverage notes:
# - Some names below are Triton kernels or shared internal helpers, not public
#   Python entry points. For those, the stable way to validate precision is to
#   run the higher-level pytest suites that exercise them.
# - This script intentionally prefers repository tests over ad-hoc direct calls,
#   so coverage tracks upstream changes in shapes/dtypes/varlen/GQA/decoding.
#
# Requested kernels covered by these suites:
# - chunk_local_cumsum
# - chunk_scaled_dot_kkt_fwd
# - solve_tril
# - recompute_w_u_fwd
# - chunk_gated_delta_rule_fwd_h
# - chunk_fwd_o
# - chunk_bwd_dv_local
# - chunk_gated_delta_rule_bwd_dhu
# - chunk_bwd_dqkwg
# - prepare_wy_repr_bwd
# - layer_norm_gated_fwd / layer_norm_gated_bwd / layer_norm_gated_fwd_kernel
# - fused_recurrent_fwd / fused_recurrent_bwd
# - chunk_fwd_h / chunk_bwd_dh / chunk_bwd_dv
# - causal_conv1d_fwd_kernel / causal_conv1d_bwd_kernel
# - l2norm_fwd
# - chunk_kda_fwd_intra / chunk_kda_bwd_kernel_wy_dqkg_fused / kda_gate_fwd_kernel
# - chunk_gla_fwd_o_gk

declare -a TARGETS=(
  "tests/ops/test_utils.py"
  "tests/ops/test_solve_tril.py"
  "tests/modules/test_layernorm_gated.py"
  "tests/modules/test_l2norm.py"
  "tests/modules/test_conv.py"
  "tests/ops/test_delta.py"
  "tests/ops/test_gated_delta.py"
  "tests/ops/test_comba.py"
  "tests/ops/test_oja.py"
  "tests/ops/test_gla.py"
  "tests/ops/test_simple_gla.py"
  "tests/ops/test_gsa.py"
  "tests/ops/test_kda.py"
  "tests/ops/test_delta_product.py"
  "tests/ops/test_gated_delta_product.py"
  "tests/ops/test_mesa.py"
  "tests/ops/test_rwkv6.py"
  "tests/ops/test_iplr_delta.py"
  "tests/ops/test_dplr_delta.py"
  "tests/ops/test_deltaformer.py"
  "tests/ops/test_intracard_cache.py"
)

echo "Repository root: ${ROOT_DIR}"
echo "Running kernel accuracy/generalization suite with ${#TARGETS[@]} pytest targets..."
printf '  - %s\n' "${TARGETS[@]}"

pytest -s -v "${TARGETS[@]}" "$@"
