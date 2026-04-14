#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

for file in kernel_suite/tests_perf/test_*_perf.py; do
  echo "==> ${file}"
  python "${file}" "$@"
done
