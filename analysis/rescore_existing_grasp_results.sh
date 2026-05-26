#!/usr/bin/env bash
set -euo pipefail

GP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${GP_ROOT}/grasp_paths.sh"
cd "${GP_ROOT}"

ALL_METHODS_OUT_DIR="${ALL_METHODS_OUT_DIR:-${GRASP_ANALYSIS_ALL_ROOT}}"
DIRECT_OUT_DIR="${DIRECT_OUT_DIR:-${GRASP_ANALYSIS_DIRECT_ROOT}}"
DIRECT_ANALYSIS_DIR="${DIRECT_ANALYSIS_DIR:-${DIRECT_OUT_DIR}}"
DIRECT_SUMMARY_CSV="${DIRECT_SUMMARY_CSV:-${DIRECT_OUT_DIR}/summary.csv}"
DIRECT_CHECKPOINT_CSV="${DIRECT_CHECKPOINT_CSV:-${DIRECT_OUT_DIR}/checkpoint_manifest.csv}"
ORACLE_CROP_OUT_DIR="${ORACLE_CROP_OUT_DIR:-${GRASP_ANALYSIS_ORACLE_CROP_ROOT}}"
ORACLE_CROP_ANALYSIS_DIR="${ORACLE_CROP_ANALYSIS_DIR:-${ORACLE_CROP_OUT_DIR}}"
ORACLE_CROP_SUMMARY_CSV="${ORACLE_CROP_SUMMARY_CSV:-${ORACLE_CROP_OUT_DIR}/summary.csv}"
ORACLE_CROP_CHECKPOINT_CSV="${ORACLE_CROP_CHECKPOINT_CSV:-${ORACLE_CROP_OUT_DIR}/checkpoint_manifest.csv}"
PREDICTED_VCOT_OUT_DIR="${PREDICTED_VCOT_OUT_DIR:-${GRASP_ANALYSIS_PREDICTED_VCOT_ROOT}}"
OUT_DIR="${OUT_DIR:-${ALL_METHODS_OUT_DIR}}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${OUT_DIR}}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUT_DIR}/summary.csv}"
CHECKPOINT_CSV="${CHECKPOINT_CSV:-${OUT_DIR}/checkpoint_manifest.csv}"
RUN_TASK_DIRECT="${RUN_TASK_DIRECT:-True}"
RUN_TASK_ORACLE_CROP="${RUN_TASK_ORACLE_CROP:-True}"
RUN_TASK_PREDICTED_VCOT="${RUN_TASK_PREDICTED_VCOT:-True}"
RUN_CROP_FRAME_PRIOR="${RUN_CROP_FRAME_PRIOR:-True}"
CROP_FRAME_PRIOR_SAMPLE_LIMIT="${CROP_FRAME_PRIOR_SAMPLE_LIMIT:-5000}"
DIRECT_RESULT_ROOT="${DIRECT_RESULT_ROOT:-${GRASP_DIRECT_RESULT_ROOT}}"
ORACLE_CROP_RESULT_ROOT="${ORACLE_CROP_RESULT_ROOT:-${GRASP_CROP_RESULT_ROOT}}"
VCOT_RESULT_ROOT="${VCOT_RESULT_ROOT:-${GRASP_VCOT_RESULT_ROOT}}"
VCOT_OUT_DIR="${VCOT_OUT_DIR:-${PREDICTED_VCOT_OUT_DIR}}"
VCOT_ANALYSIS_DIR="${VCOT_ANALYSIS_DIR:-${VCOT_OUT_DIR}}"
VCOT_SUMMARY_CSV="${VCOT_SUMMARY_CSV:-${VCOT_OUT_DIR}/summary.csv}"
VCOT_CHECKPOINT_CSV="${VCOT_CHECKPOINT_CSV:-${VCOT_OUT_DIR}/checkpoint_manifest.csv}"

result_files_from_roots() {
  local roots=("$@")
  local existing_roots=()
  local root

  for root in "${roots[@]}"; do
    if [[ -d "${root}" ]]; then
      existing_roots+=("${root}")
    fi
  done

  if [[ "${#existing_roots[@]}" -eq 0 ]]; then
    return
  fi

  find "${existing_roots[@]}" \
    -type f \
    -name '*.json' \
    | sort
}

result_files_from_list() {
  local list_path="$1"
  shift
  python analysis/filter_result_list.py --list "${list_path}" "$@"
}

if [[ -n "${DIRECT_RESULT_LIST:-}" ]]; then
  mapfile -t DIRECT_RESULT_FILES < <(result_files_from_list "${DIRECT_RESULT_LIST}")
elif [[ -n "${RESULT_LIST:-}" ]]; then
  mapfile -t DIRECT_RESULT_FILES < <(result_files_from_list "${RESULT_LIST}" --mode direct_grasp)
else
  mapfile -t DIRECT_RESULT_FILES < <(result_files_from_roots "${DIRECT_RESULT_ROOT}")
fi

if [[ -n "${ORACLE_CROP_RESULT_LIST:-}" ]]; then
  mapfile -t ORACLE_CROP_RESULT_FILES < <(result_files_from_list "${ORACLE_CROP_RESULT_LIST}")
elif [[ -n "${RESULT_LIST:-}" ]]; then
  mapfile -t ORACLE_CROP_RESULT_FILES < <(result_files_from_list "${RESULT_LIST}" --mode oracle_crop)
else
  mapfile -t ORACLE_CROP_RESULT_FILES < <(result_files_from_roots "${ORACLE_CROP_RESULT_ROOT}")
fi

if [[ -n "${VCOT_RESULT_LIST:-}" ]]; then
  mapfile -t VCOT_RESULT_FILES < <(result_files_from_list "${VCOT_RESULT_LIST}")
elif [[ -n "${RESULT_LIST:-}" ]]; then
  mapfile -t VCOT_RESULT_FILES < <(result_files_from_list "${RESULT_LIST}" --mode predicted_vcot)
else
  mapfile -t VCOT_RESULT_FILES < <(result_files_from_roots "${VCOT_RESULT_ROOT}")
fi

if [[ -n "${RESULT_LIST:-}" ]]; then
  mapfile -t RESULT_FILES < <(result_files_from_list "${RESULT_LIST}")
else
  RESULT_FILES=("${DIRECT_RESULT_FILES[@]}" "${ORACLE_CROP_RESULT_FILES[@]}" "${VCOT_RESULT_FILES[@]}")
fi

if [[ "${#RESULT_FILES[@]}" -eq 0 ]]; then
  echo "No result JSON files found." >&2
  exit 1
fi

score_and_analyze() {
  local label="$1"
  local summary_csv="$2"
  local analysis_dir="$3"
  shift 3
  local result_files=("$@")

  if [ "${#result_files[@]}" -eq 0 ]; then
    echo "Skip ${label}: no result JSON files found."
    return
  fi

  echo "Scoring/analyzing ${label}: ${#result_files[@]} result files"
  echo "Summary CSV: ${summary_csv}"
  echo "Enhanced analysis dir: ${analysis_dir}"

  local args=(
    analysis/score_vcot_grasp_results.py
    --summary-csv "${summary_csv}"
    --analysis-out-dir "${analysis_dir}"
  )
  args+=("${result_files[@]}")

  python "${args[@]}"
}

collect_checkpoints() {
  local label="$1"
  local checkpoint_csv="$2"
  shift 2
  local result_files=("$@")

  if [ "${#result_files[@]}" -eq 0 ]; then
    return
  fi

  echo "Checkpoint manifest (${label}): ${checkpoint_csv}"
  python analysis/collect_checkpoint_manifest.py \
    --out "${checkpoint_csv}" \
    "${result_files[@]}"
}

diagnose_crop_frame_prior() {
  local label="$1"
  local summary_csv="$2"
  local analysis_dir="$3"
  shift 3
  local result_files=("$@")

  if [[ "${RUN_CROP_FRAME_PRIOR}" != "True" || "${#result_files[@]}" -eq 0 ]]; then
    return
  fi

  echo "Crop-frame constant-prior diagnosis (${label}): ${analysis_dir}/methods/oracle_crop/diagnostics/crop_frame_prior_summary.csv"
  python analysis/diagnose_crop_frame_prior.py \
    --summary-csv "${summary_csv}" \
    --out-dir "${analysis_dir}" \
    --sample-limit "${CROP_FRAME_PRIOR_SAMPLE_LIMIT}"
}

echo "Using score_vcot_grasp_results.py default thresholds."
echo "No rescored JSON copies will be written."
echo "Direct result root: ${DIRECT_RESULT_ROOT} (${#DIRECT_RESULT_FILES[@]} files)"
echo "Oracle crop result root: ${ORACLE_CROP_RESULT_ROOT} (${#ORACLE_CROP_RESULT_FILES[@]} files)"
echo "Predicted VCoT result root: ${VCOT_RESULT_ROOT} (${#VCOT_RESULT_FILES[@]} files)"
echo "Full checkpoint manifest: ${CHECKPOINT_CSV}"

mkdir -p "${OUT_DIR}"
score_and_analyze \
  "all current grasp results (direct + oracle crop + predicted VCoT)" \
  "${SUMMARY_CSV}" \
  "${ANALYSIS_DIR}" \
  "${RESULT_FILES[@]}"
diagnose_crop_frame_prior \
  "all current grasp results" \
  "${SUMMARY_CSV}" \
  "${ANALYSIS_DIR}" \
  "${ORACLE_CROP_RESULT_FILES[@]}"
collect_checkpoints "all current grasp results" "${CHECKPOINT_CSV}" "${RESULT_FILES[@]}"

if [[ "${RUN_TASK_DIRECT}" == "True" ]]; then
  mkdir -p "${DIRECT_OUT_DIR}"
  echo "Direct-only checkpoint manifest: ${DIRECT_CHECKPOINT_CSV}"
  score_and_analyze \
    "direct_grasp task results" \
    "${DIRECT_SUMMARY_CSV}" \
    "${DIRECT_ANALYSIS_DIR}" \
    "${DIRECT_RESULT_FILES[@]}"
  collect_checkpoints "direct_grasp task results" "${DIRECT_CHECKPOINT_CSV}" "${DIRECT_RESULT_FILES[@]}"
fi

if [[ "${RUN_TASK_ORACLE_CROP}" == "True" ]]; then
  mkdir -p "${ORACLE_CROP_OUT_DIR}"
  echo "Oracle-crop-only checkpoint manifest: ${ORACLE_CROP_CHECKPOINT_CSV}"
  score_and_analyze \
    "oracle_crop task results" \
    "${ORACLE_CROP_SUMMARY_CSV}" \
    "${ORACLE_CROP_ANALYSIS_DIR}" \
    "${ORACLE_CROP_RESULT_FILES[@]}"
  diagnose_crop_frame_prior \
    "oracle_crop task results" \
    "${ORACLE_CROP_SUMMARY_CSV}" \
    "${ORACLE_CROP_ANALYSIS_DIR}" \
    "${ORACLE_CROP_RESULT_FILES[@]}"
  collect_checkpoints "oracle_crop task results" "${ORACLE_CROP_CHECKPOINT_CSV}" "${ORACLE_CROP_RESULT_FILES[@]}"
fi

if [[ "${RUN_TASK_PREDICTED_VCOT}" == "True" ]]; then
  mkdir -p "${VCOT_OUT_DIR}"
  echo "Predicted-VCoT-only checkpoint manifest: ${VCOT_CHECKPOINT_CSV}"
  score_and_analyze \
    "predicted_vcot task results" \
    "${VCOT_SUMMARY_CSV}" \
    "${VCOT_ANALYSIS_DIR}" \
    "${VCOT_RESULT_FILES[@]}"
  collect_checkpoints "predicted_vcot task results" "${VCOT_CHECKPOINT_CSV}" "${VCOT_RESULT_FILES[@]}"
fi
