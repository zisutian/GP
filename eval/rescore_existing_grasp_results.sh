#!/usr/bin/env bash
set -euo pipefail

GP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${GP_ROOT}"

OUT_DIR="${OUT_DIR:-${GP_ROOT}/rescore_result/vcot_grasp_summary_analysis}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${OUT_DIR}/analysis}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUT_DIR}/summary.csv}"
CHECKPOINT_CSV="${CHECKPOINT_CSV:-${OUT_DIR}/checkpoint_manifest.csv}"

RESULT_ROOTS=()
for root in result/vcot_grasp_direct result/vcot_grasp_crop result/vcot_grasp_vcot; do
  if [[ -d "${root}" ]]; then
    RESULT_ROOTS+=("${root}")
  fi
done

if [[ "${#RESULT_ROOTS[@]}" -eq 0 ]]; then
  echo "No result directories found." >&2
  exit 1
fi

mapfile -t RESULT_FILES < <(
  find "${RESULT_ROOTS[@]}" \
    -type f \
    -name '*.json' \
    | sort
)

if [ "${#RESULT_FILES[@]}" -eq 0 ]; then
  echo "No result JSON files found." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
echo "Scoring/analyzing ${#RESULT_FILES[@]} result files"
echo "Using score_vcot_grasp_results.py default thresholds."
echo "No rescored JSON copies will be written."
echo "Summary CSV: ${SUMMARY_CSV}"
echo "Enhanced analysis dir: ${ANALYSIS_DIR}"
echo "Checkpoint manifest: ${CHECKPOINT_CSV}"

ARGS=(
  eval/score_vcot_grasp_results.py
  --summary-csv "${SUMMARY_CSV}"
  --analysis-out-dir "${ANALYSIS_DIR}"
)
ARGS+=("${RESULT_FILES[@]}")

python "${ARGS[@]}"

python eval/collect_checkpoint_manifest.py \
  --out "${CHECKPOINT_CSV}" \
  "${RESULT_FILES[@]}"
