#!/usr/bin/env bash

has_training_artifact() {
  local work_dir="$1"
  [[ -f "${work_dir}/model.safetensors" ]] || \
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
  elif has_training_artifact "${work_dir}"; then
    echo "skip"
  elif [[ -d "${work_dir}" ]]; then
    echo "overwrite"
  else
    echo "train"
  fi
}

prepare_eval_dir() {
  local out_dir="$1"
  mkdir -p "${out_dir}"
}

should_run_eval() {
  [[ "${RUN_EVAL:-1}" == "1" ]]
}

skip_eval_without_checkpoint() {
  local work_dir="$1"
  if has_training_artifact "${work_dir}"; then
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
  local overwrite_eval_results="${3:-${OVERWRITE_EVAL_RESULTS:-False}}"
  local old_ifs
  local datasets
  local missing=()
  local dataset

  if [[ "${overwrite_eval_results}" == "True" ]]; then
    echo "${datasets_csv}"
    return
  fi

  old_ifs="${IFS}"
  IFS=","
  read -r -a datasets <<< "${datasets_csv}"
  IFS="${old_ifs}"

  for dataset in "${datasets[@]}"; do
    if [[ -z "${dataset}" ]]; then
      continue
    fi
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
  local old_ifs
  local datasets
  local splits=()
  local dataset

  old_ifs="${IFS}"
  IFS=","
  read -r -a datasets <<< "${datasets_csv}"
  IFS="${old_ifs}"

  for dataset in "${datasets[@]}"; do
    case "${dataset}" in
      test_seen|test_unseen)
        splits+=("${dataset}")
        ;;
    esac
  done

  old_ifs="${IFS}"
  IFS=" "
  echo "${splits[*]}"
  IFS="${old_ifs}"
}
