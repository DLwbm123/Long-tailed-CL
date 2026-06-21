#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$PYTHON_BIN" "$ROOT/tools/run_cifar100lt_singlehead.py" "$@"
