#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d womd_protos ]; then
  ./generate_protos.sh
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
export KMP_INIT_AT_FORK=FALSE
export NUMEXPR_NUM_THREADS=1
export MKL_THREADING_LAYER=SEQUENTIAL

python3 -m pytest tests/ -q
