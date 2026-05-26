#!/usr/bin/env bash

if [[ -z "${GP_ROOT:-}" ]]; then
  GP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
source "${GP_ROOT}/grasp_paths.sh"
PYTHON_BIN="${PYTHON_BIN:-${GRASP_PYTHON_BIN}}"

stage() {
  echo
  echo "===== $* ====="
}

has_checkpoint_artifact() {
  local work_dir="$1"
  find "${work_dir}" -maxdepth 1 -type d -name 'checkpoint-*' -print -quit 2>/dev/null | grep -q .
}

latest_checkpoint() {
  local work_dir="$1"
  find "${work_dir}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1
}

training_action() {
  local work_dir="$1"
  local overwrite_output_dir="$2"
  if [[ "${overwrite_output_dir}" == "True" ]]; then
    echo "train"
  elif has_checkpoint_artifact "${work_dir}"; then
    echo "skip"
  elif [[ -d "${work_dir}" ]]; then
    echo "overwrite"
  else
    echo "train"
  fi
}

prepare_eval_dir() {
  mkdir -p "$1"
}

should_run_eval() {
  [[ "${RUN_EVAL:-${GRASP_RUN_EVAL}}" == "1" ]]
}

skip_eval_without_checkpoint() {
  local work_dir="$1"
  if has_checkpoint_artifact "${work_dir}"; then
    return 1
  fi
  echo "Skip eval: no checkpoint found in ${work_dir}"
  return 0
}

has_eval_result() {
  local out_dir="$1"
  local dataset="$2"
  find "${out_dir}" -maxdepth 1 -type f -name "${dataset}_*.json" -print -quit 2>/dev/null | grep -q .
}

missing_eval_datasets() {
  local out_dir="$1"
  local datasets_csv="$2"
  local overwrite_eval_results="$3"
  local old_ifs datasets missing=() dataset

  if [[ "${overwrite_eval_results}" == "True" ]]; then
    echo "${datasets_csv}"
    return
  fi

  old_ifs="${IFS}"
  IFS=","
  read -r -a datasets <<< "${datasets_csv}"
  IFS="${old_ifs}"

  for dataset in "${datasets[@]}"; do
    [[ -z "${dataset}" ]] && continue
    if ! has_eval_result "${out_dir}" "${dataset}"; then
      missing+=("${dataset}")
    fi
  done

  old_ifs="${IFS}"
  IFS=","
  echo "${missing[*]}"
  IFS="${old_ifs}"
}

default_dataset_splits() {
  local datasets_csv="$1"
  local old_ifs datasets splits=() dataset
  old_ifs="${IFS}"
  IFS=","
  read -r -a datasets <<< "${datasets_csv}"
  IFS="${old_ifs}"
  for dataset in "${datasets[@]}"; do
    case "${dataset}" in
      test_seen|test_unseen) splits+=("${dataset}") ;;
    esac
  done
  old_ifs="${IFS}"
  IFS=" "
  echo "${splits[*]}"
  IFS="${old_ifs}"
}

require_data_index_splits() {
  local data_index_root="$1"
  local datasets_csv="$2"
  local split
  for split in $(default_dataset_splits "${datasets_csv}"); do
    if [[ ! -s "${data_index_root}/${split}.jsonl" ]]; then
      echo "Missing data index: ${data_index_root}/${split}.jsonl" >&2
      echo "Run the data stage explicitly before eval." >&2
      return 1
    fi
  done
}
