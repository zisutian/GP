set -e

GP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${GP_ROOT}/InternVL/internvl_chat"

GP_ROOT="${GP_ROOT}" bash shell/internvl2.5/2nd_finetune/internvl2_5_1b_grasp_crop_lmdb_lora.sh
