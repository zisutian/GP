#!/usr/bin/env bash

if [[ -z "${GP_ROOT:-}" ]]; then
  GP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

GRASP_CONFIG_PYTHON="${GRASP_CONFIG_PYTHON:-${PYTHON_BIN:-python}}"
eval "$("${GRASP_CONFIG_PYTHON}" "${GP_ROOT}/grasp_settings.py" env)"
