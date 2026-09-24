#!/usr/bin/env bash
set -euo pipefail
source /home/dc2339/apps/miniconda3/etc/profile.d/conda.sh
conda activate topokit
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd /home/dc2339/topokit_project
exec python -u topokit/workflows/protein_ligand_persistent_homology/run_study.py "$@"
