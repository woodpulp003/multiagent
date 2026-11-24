#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
# FOW_PYTHON lets local tools point at the venv; the arena uses its own python3.
exec "${FOW_PYTHON:-python3}" -u bot.py "$@"
