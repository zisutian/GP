# VCoTGrasp GP

本目录是 Grasp-Anything / VCoT-style grasp 实验的工程入口，当前目标是用
`InternVL2.5-1B` 的原生图像输入做三条 grasp pipeline 的训练、评估和汇总分析。

核心约定：

```text
原始 LMDB 不移动、不改写：
../VCoT-Grasp-self/data/grasp_anything/lmdb/

GP 目录只保存：
manifest / meta / config / train output / eval result / rescore analysis
```

## Pipeline

| pipeline | 训练任务 | 评估任务 | 定位 |
| --- | --- | --- | --- |
| `direct_grasp` | 原图 + `grasp the {obj_name}` -> grasp | 原图直接预测 grasp | direct baseline |
| `oracle_crop` | 原图 + GT mask object crop + `grasp the {obj_name}` -> grasp | 使用 GT mask crop | oracle crop upper bound |
| `predicted_vcot` | joint: bbox detect + crop grasp | stage1 预测 bbox，stage2 用 predicted crop 预测 grasp | 非 oracle two-stage 闭环 |

注意：`oracle_crop` 是 GT mask crop upper bound，不是公平 predicted pipeline；
`predicted_vcot` 才是不使用 oracle crop 的 two-stage 闭环。

## Data Flow

整体链路：

```text
origin_split/*.csv + LMDB
  -> manifest (*.jsonl)
  -> internvl_meta_train.json
  -> train 写 vcot_config.json
  -> checkpoint 复制 vcot_config.json
  -> eval 从 checkpoint/父目录读取 vcot_config.json
  -> result JSON
  -> rescore/analysis CSV
```

几个名字的含义：

```text
manifest:
  *.jsonl 样本清单。每行保存 image_key / grasp_key / mask_key / obj_name / crop 参数等。
  它不是图像数据本体，也不保存 crop 图像。

internvl_meta_train.json:
  InternVL 训练入口读取的数据集配置。
  它指向 manifest，并声明 vcot_dataset、repeat_time、crop 参数等。

vcot_config.json:
  每个实验/checkpoint 的可回溯配置。
  eval/rescore 依赖它确认 pipeline、meta_path、crop 参数、bbox_ratio 等。
```

`data/vcot_grasp` 里有三类目录，语义不同：

```text
shared canonical manifests:
  可以跨实验共用，因为 manifest 不含 crop 设计参数。

default/example manifests:
  给默认 train/eval 入口和快速 sanity check 使用。
  它们不是所有 hparam 实验都应该共用的目录。

hparam manifests:
  每个实验私有，尤其用于保存 crop 参数相关的 manifest。
```

共享 canonical manifests：

```text
data/vcot_grasp/direct/
  train.jsonl
  test_seen.jsonl
  test_unseen.jsonl
  internvl_meta_train.json

data/vcot_grasp/bbox/
  train.jsonl
  test_seen.jsonl
  test_unseen.jsonl
  internvl_meta_train.json
```

`direct` 只依赖原图和 grasp label，`bbox` 只依赖原图和 mask；它们不包含
`bbox_edge_expand / min_bbox_half_size / target_coordinate_frame` 这类 crop 设计参数，所以可以被 hparam 实验共用。

默认/example manifests：

```text
data/vcot_grasp/crop/
  train.jsonl
  test_seen.jsonl
  test_unseen.jsonl
  internvl_meta_train.json

data/vcot_grasp/vcot/
  test_seen.jsonl
  test_unseen.jsonl
  internvl_meta_train.json
```

`data/vcot_grasp/crop/` 是默认 oracle crop 配置，不应被不同 crop 设计的 hparam 实验静默共用。
`data/vcot_grasp/vcot/` 是默认 VCoT eval/meta 入口。真正的 VCoT hparam 会使用下面的私有目录。

hparam manifests：

```text
data/vcot_grasp/crop_hparams/{experiment}/
  train.jsonl
  internvl_meta_train.json

data/vcot_grasp/vcot_hparams/{experiment}/
  crop/train.jsonl
  test_seen.jsonl
  test_unseen.jsonl
  internvl_meta_train.json
```

crop/VCoT hparam 目录是实验私有的，用来避免不同 crop 设置共用同一个语义不清的 `train.jsonl`。

`scripts/ensure_grasp_data.py` 会在 train/eval 前检查 manifest/meta：

```bash
python scripts/ensure_grasp_data.py direct --splits train test_seen test_unseen
python scripts/ensure_grasp_data.py crop --splits train test_seen test_unseen
python scripts/ensure_grasp_data.py vcot --eval-splits test_seen test_unseen
```

如果文件缺失会自动从 `origin_split/*.csv` 重建；如果已有 crop manifest 的参数和当前实验参数不匹配，会报错而不是静默覆盖。
hparam sweep 会传入自己的 `--meta-path/--output-root/--crop-root`，因此不会把私有实验数据写进默认/example 目录。

## Important Files

公共工具：

| 文件 | 作用 |
| --- | --- |
| `scripts/ensure_grasp_data.py` | 检查/自动构建 manifest 和 `internvl_meta_train.json` |
| `scripts/grasp_config.py` | 写入实验目录和 checkpoint 下的 `vcot_config.json` |
| `scripts/grasp_run_common.sh` | sweep/eval 共用的 checkpoint/result 跳过逻辑 |

数据准备与懒加载：

| 文件 | 作用 |
| --- | --- |
| `data_tools/prepare_grasp_anything_direct.py` | 生成 direct manifest/meta |
| `data_tools/prepare_grasp_anything_crop.py` | 生成 oracle crop manifest/meta |
| `data_tools/prepare_grasp_anything_bbox.py` | 生成 bbox detection manifest/meta |
| `data_tools/prepare_grasp_anything_vcot.py` | 生成 VCoT joint train meta 与 eval manifest |
| `data_tools/vcot_direct_lmdb.py` | direct 样本构造 |
| `data_tools/vcot_crop_lmdb.py` | GT mask crop、坐标转换、crop grasp 样本构造 |
| `data_tools/vcot_bbox_lmdb.py` | mask -> bbox detection 样本构造 |

训练 sweep：

| 文件 | 作用 |
| --- | --- |
| `run_grasp_direct_hparam_sweep.sh` | direct hparam sweep |
| `run_grasp_crop_design_sweep.sh` | oracle crop design sweep |
| `run_grasp_vcot_design_sweep.sh` | predicted VCoT design sweep |

底层训练脚本由 sweep 直接调用，根目录不再保留单次 train wrapper：

```text
InternVL/internvl_chat/shell/internvl2.5/2nd_finetune/
  internvl2_5_1b_grasp_direct_lmdb_lora.sh
  internvl2_5_1b_grasp_crop_lmdb_lora.sh
  internvl2_5_1b_grasp_vcot_lmdb_lora.sh
```

评估与分析：

| 文件 | 作用 |
| --- | --- |
| `eval/eval_grasp_direct_lmdb_lora.sh` | direct eval shell |
| `eval/eval_grasp_crop_lmdb_lora.sh` | oracle crop eval shell |
| `eval/eval_grasp_vcot_lmdb_lora.sh` | predicted VCoT eval shell |
| `eval/evaluate_direct_grasp.py` | direct 推理评估 |
| `eval/evaluate_crop_grasp.py` | oracle crop 推理评估 |
| `eval/evaluate_vcot_grasp.py` | predicted bbox -> predicted crop -> grasp 推理评估 |
| `eval/rescore_existing_grasp_results.sh` | 对已有 result JSON 统一重打分和分析 |
| `eval/score_vcot_grasp_results.py` | result JSON 指标重算 |
| `eval/collect_checkpoint_manifest.py` | 从 result/config 收集 checkpoint manifest |
| `eval/analyze_grasp_results.py` | 输出 summary/error/sample/threshold/cross 分析 |

## Run

推荐使用已有 conda 环境运行：

```bash
conda run --no-capture-output -n 260513-internvl bash run_grasp_direct_hparam_sweep.sh
conda run --no-capture-output -n 260513-internvl bash run_grasp_crop_design_sweep.sh
conda run --no-capture-output -n 260513-internvl bash run_grasp_vcot_design_sweep.sh
```

sweep 的行为：

```text
1. ensure manifest/meta
2. 检查 work_dir 是否已有 checkpoint/model
3. 缺 checkpoint 则训练
4. 写/复制 vcot_config.json
5. 检查 result 目录是否已有对应 split JSON
6. 缺 result 则 eval
```

常用环境变量：

```bash
RUN_EVAL=0                     # 只训练，不 eval
OVERWRITE_OUTPUT_DIR=True      # 强制重训
OVERWRITE_EVAL_RESULTS=True    # 强制重评估
EVAL_DATASETS=test_seen        # 只评估一个 split
CUDA_VISIBLE_DEVICES=0,1
GPUS=2
```

单独 eval：

```bash
conda run --no-capture-output -n 260513-internvl bash eval/eval_grasp_direct_lmdb_lora.sh
conda run --no-capture-output -n 260513-internvl bash eval/eval_grasp_crop_lmdb_lora.sh
conda run --no-capture-output -n 260513-internvl bash eval/eval_grasp_vcot_lmdb_lora.sh
```

单独 eval 默认会从对应 `WORK_DIR` 找最新 `checkpoint-*`。也可以显式传：

```bash
CHECKPOINT=/path/to/checkpoint-1 \
OUT_DIR=result/custom_eval \
conda run --no-capture-output -n 260513-internvl bash eval/eval_grasp_vcot_lmdb_lora.sh
```

统一 rescore/analysis：

```bash
conda run --no-capture-output -n 260513-internvl \
  bash eval/rescore_existing_grasp_results.sh
```

输出目录：

```text
rescore_result/vcot_grasp_summary_analysis/
  summary.csv
  checkpoint_manifest.csv
  analysis/main_summary.csv
  analysis/error_stats.csv
  analysis/sample_metrics.csv
  analysis/threshold_sweep_geometry.csv
  analysis/threshold_sweep_iou.csv
  analysis/direct_vs_crop_cross.csv
  analysis/crop_quality_samples.csv
  analysis/crop_quality_summary.csv
  analysis/manifest.json
```

## Outputs

训练输出：

```text
InternVL/internvl_chat/work_dirs/internvl_chat_v2_5/
  grasp_direct_hparams/{experiment}/
  grasp_crop_hparams/{experiment}/
  grasp_vcot_hparams/{experiment}/
```

评估结果：

```text
result/vcot_grasp_direct/hparams/{experiment}/
result/vcot_grasp_crop/hparams/{experiment}/
result/vcot_grasp_vcot/hparams/{experiment}/
```

每个 result JSON 的 summary 会记录 checkpoint 和 loaded config。rescore 只依赖这些 summary 和 `vcot_config.json`，不再从目录名猜参数。

## Current Reference Results

当前 direct best：

```text
direct lora16 lr8e-5 ep1
seen official:   72.47
seen top1:       49.40
unseen official: 53.87
unseen top1:     33.83
```

当前较好的 oracle crop：

```text
crop_frame_lora16_lr8e-5_ep1_patch6_edge5_half40
seen official:   76.70
seen top1:       59.33
unseen official: 67.65
unseen top1:     49.97
```

predicted VCoT 示例：

```text
vcot_lora16_lr8e-5_ep1_patch6_bbox0.5
seen official:   71.97
unseen official: 52.19

vcot_full_lora16_lr8e-5_ep1_patch6_edge15_half50_bbox0.5
seen official:   72.30
unseen official: 54.47
```

## Development Checks

修改 shell/Python 后建议跑：

```bash
python -m py_compile scripts/ensure_grasp_data.py scripts/grasp_config.py \
  data_tools/prepare_grasp_anything_direct.py \
  data_tools/prepare_grasp_anything_crop.py \
  data_tools/prepare_grasp_anything_bbox.py \
  data_tools/prepare_grasp_anything_vcot.py

bash -n scripts/grasp_run_common.sh \
  run_grasp_direct_hparam_sweep.sh \
  run_grasp_crop_design_sweep.sh \
  run_grasp_vcot_design_sweep.sh \
  eval/eval_grasp_direct_lmdb_lora.sh \
  eval/eval_grasp_crop_lmdb_lora.sh \
  eval/eval_grasp_vcot_lmdb_lora.sh

git diff --check
```

更详细的工程约定见 `info/grasp_pipeline_overview.md`。
