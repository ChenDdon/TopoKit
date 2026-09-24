#!/usr/bin/env bash
# Execute on the user's authorized AWS host inside the uploaded checkout.
# This script does not transfer files, provision instances, or change libraries.
set -euo pipefail

benchmark_python=${1:?"usage: bash run_aws.sh PYTHON_EXECUTABLE [RESULT_DIRECTORY]"}
benchmark_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
benchmark_results=${2:-"${benchmark_root}/examples/pressure_test/results/aws_2026-09-05"}
mkdir -p "${benchmark_results}"
benchmark_results=$(cd "${benchmark_results}" && pwd)
cd "${benchmark_root}"
export PYTHONPATH="${benchmark_root}/src"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 MPLBACKEND=Agg
export MPLCONFIGDIR="${benchmark_results}/matplotlib-cache"

"${benchmark_python}" -m pip freeze > "${benchmark_results}/requirements.lock"
"${benchmark_python}" -m pytest -q 2>&1 | tee "${benchmark_results}/pytest.log"

mkdir -p "${benchmark_results}/baseline"
tar -xzf examples/pressure_test/readability/source_before.tar.gz -C "${benchmark_results}/baseline"
"${benchmark_python}" examples/pressure_test/check_readability.py \
  --baseline-src "${benchmark_results}/baseline/src" --current-src src \
  --output "${benchmark_results}/readability.json" \
  2>&1 | tee "${benchmark_results}/readability.log"

benchmark_exit=0
for benchmark_suite in smoke compact_smoke compact_pressure pressure_q0 pressure_q1 pressure_q2 supplemental; do
  benchmark_case_exit=0
  "${benchmark_python}" examples/pressure_test/run_pressure.py \
    --matrix "examples/pressure_test/matrices/${benchmark_suite}.json" \
    --wall-seconds 600 --rss-gib 16 --address-gib 22 \
    --output "${benchmark_results}/${benchmark_suite}" --resume \
    2>&1 | tee -a "${benchmark_results}/${benchmark_suite}.log" || benchmark_case_exit=$?
  "${benchmark_python}" examples/pressure_test/summarize.py \
    "${benchmark_results}/${benchmark_suite}" \
    --output "${benchmark_results}/${benchmark_suite}/RESULTS.md"
  if [[ "${benchmark_suite}" == *smoke ]]; then
    "${benchmark_python}" -c 'import json,sys; m=json.load(open(sys.argv[1])); assert m["status"] == "complete" and m.get("status_counts") == {"success": m["planned_cases"]}, {k:m.get(k) for k in ("status","status_counts","completed_cases","planned_cases")}' \
      "${benchmark_results}/${benchmark_suite}/manifest.json"
  fi
  if (( benchmark_case_exit != 0 )); then benchmark_exit=1; fi
done
exit "${benchmark_exit}"
