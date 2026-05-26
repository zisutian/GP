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
  eval/analysis 依赖它确认 pipeline、meta_path、crop 参数、bbox_ratio 等。
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
  test_seen.jsonl
  test_unseen.jsonl
  internvl_meta_train.json

data/vcot_grasp/vcot_hparams/{experiment}/
  crop/train.jsonl
  test_seen.jsonl
  test_unseen.jsonl
  internvl_meta_train.json
```

crop/VCoT hparam 目录是实验私有的，用来避免不同 crop 设置共用同一个语义不清的
`train.jsonl` 或 eval split manifest。sweep eval 会通过 `DATASET_ROOT` 指向这些私有目录。

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

模型推理评估：

| 文件 | 作用 |
| --- | --- |
| `eval/eval_grasp_direct_lmdb_lora.sh` | direct eval shell |
| `eval/eval_grasp_crop_lmdb_lora.sh` | oracle crop eval shell |
| `eval/eval_grasp_vcot_lmdb_lora.sh` | predicted VCoT eval shell |
| `eval/evaluate_direct_grasp.py` | direct 推理评估 |
| `eval/evaluate_crop_grasp.py` | oracle crop 推理评估 |
| `eval/evaluate_vcot_grasp.py` | predicted bbox -> predicted crop -> grasp 推理评估 |

结果分析：

| 文件 | 作用 |
| --- | --- |
| `analysis/rescore_existing_grasp_results.sh` | 对已有 result JSON 统一重打分和分析 |
| `analysis/score_vcot_grasp_results.py` | result JSON 指标重算 |
| `analysis/collect_checkpoint_manifest.py` | 从 result/config 收集 checkpoint manifest |
| `analysis/analyze_grasp_results.py` | 输出 result 级 summary/error/sweep/cross/diagnostics 分析 |
| `analysis/diagnose_crop_frame_prior.py` | 诊断 oracle crop 的 crop-frame 坐标先验 |

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
2. 检查 work_dir 是否已有 checkpoint
3. 缺 checkpoint 则训练
4. 写/复制 vcot_config.json
5. 检查 result 目录是否已有对应 split JSON
6. 缺 result 则 eval
7. 若本轮重新训练，则强制 eval，避免 stale result
```

训练参数契约：

```text
checkpoint-* 是唯一可复用训练产物。
只有已有 checkpoint-* 时 sweep 才跳过训练；单独的 model.safetensors 不再算完成训练。

底层模型和可训练结构:
  model_name_or_path = InternVL/pretrained/OpenGVLab/InternVL2_5-1B
  conv_style = internvl2_5
  freeze_llm = True
  freeze_mlp = True
  freeze_backbone = True
  use_llm_lora = {rank}
  unfreeze_lm_head = True
  dynamic_image_size = True
  use_thumbnail = True
  bf16 = True
  deepspeed = zero_stage1_config.json

sweep -> train script -> vcot_config.json 使用同一组参数：
  experiment_name
  meta_path / crop_root / bbox_root
  use_llm_lora / learning_rate / num_train_epochs
  max_dynamic_patch / force_image_size
  bbox_ratio / bbox_edge_expand / min_bbox_half_size
  target_coordinate_frame / target_grasp_index

train 完成后会再次写 vcot_config.json，并复制到所有 checkpoint-* 下。
eval 和 analysis 均以 result summary 里的 checkpoint + loaded_vcot_config 为准。
```

当前 direct sweep 搜索范围：

```text
baseline_lora16_lr4e-5_ep1_patch6
lora8_lr4e-5_ep1_patch6
lora32_lr4e-5_ep1_patch6
lora16_lr2e-5_ep1_patch6
lora16_lr8e-5_ep1_patch6
lora16_lr4e-5_ep2_patch6
lora16_lr4e-5_ep1_patch4
lora16_lr4e-5_ep1_patch1
```

当前 oracle crop design sweep 搜索范围：

```text
crop_object_lora{r}_lr{lr}_ep{ep}_patch6_edge15_half50{g}
crop_full_lora{r}_lr{lr}_ep{ep}_patch6_edge5_half40{g}
crop_full_lora{r}_lr{lr}_ep{ep}_patch6_edge10_half40{g}
crop_frame_lora{r}_lr{lr}_ep{ep}_patch6_edge5_half40{g}
crop_frame_lora{r}_lr{lr}_ep{ep}_patch6_edge10_half40{g}
crop_frame_lora{r}_lr{lr}_ep{ep}_patch8_edge10_half40{g}

默认:
  r = 16
  lr = 8e-5
  ep = 1
  target_grasp_index = 0
  force_image_size = 448
```

当前 `run_grasp_vcot_design_sweep.sh` 固定为小范围 predicted 诊断搜索，不再大规模扫 LoRA/LR/epoch：

```text
LoRA r = 16
lr = 8e-5
epoch = 1
target_grasp_index = 0
force_image_size = 448
```

固定 checkpoint/result 目录名：

```text
vcot_full_lora16_lr8e-5_ep1_patch6_edge15_half50_bbox0.25
vcot_full_lora16_lr8e-5_ep1_patch6_edge15_half50_bbox0.5
vcot_full_lora16_lr8e-5_ep1_patch6_edge15_half50_bbox1.0
vcot_frame_lora16_lr8e-5_ep1_patch6_edge5_half40_bbox0.5
vcot_frame_lora16_lr8e-5_ep1_patch6_edge10_half40_bbox0.5
vcot_frame_lora16_lr8e-5_ep1_patch8_edge10_half40_bbox0.5
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

eval 参数契约：

```text
direct:
  原图 -> grasp

oracle_crop:
  GT mask -> crop -> 原图+GT crop -> grasp

predicted_vcot:
  stage1: 原图 + "detect {obj_name}" -> predicted bbox
  stage2: predicted bbox -> predicted crop
          原图+predicted crop + "grasp the {obj_name}" -> grasp
```

`predicted_vcot` 的 stage1/stage2 使用同一个 VCoT checkpoint。
`gt_bbox/gt_crop_box` 只用于记录和 bbox IoU 指标，不会替代 predicted crop。

eval result JSON 契约：

```text
outputs[]:
  grasp_id
  obj_name
  split
  pred_norm
  raw model answer / parse status
  crop_box 或 pred_crop_box / gt_crop_box
  pred_bbox_iou 等 pipeline 相关诊断字段

summary:
  checkpoint
  loaded_vcot_config
  evaluation_mode
  target_coordinate_frame
  bbox_edge_expand
  min_bbox_half_size
  target_grasp_index
  bbox_ratio
```

`pred_norm` 是归一化 grasp，rescore 会按 `image_size=416` 反归一化为
`[cx, cy, w, h, angle_deg]`。如果某个样本解析失败，`pred_norm` 为 `None`，会计入
all 分母，但不会进入 valid 分母。

单独评估 crop/VCoT hparam checkpoint 时，建议同时传对应实验的私有 manifest 根目录：

```bash
WORK_DIR=InternVL/internvl_chat/work_dirs/internvl_chat_v2_5/grasp_vcot_hparams/{experiment} \
DATASET_ROOT=data/vcot_grasp/vcot_hparams/{experiment} \
OUT_DIR=result/vcot_grasp_vcot/hparams/{experiment} \
conda run --no-capture-output -n 260513-internvl bash eval/eval_grasp_vcot_lmdb_lora.sh
```

统一 rescore/analysis：

```bash
conda run --no-capture-output -n 260513-internvl \
  bash analysis/rescore_existing_grasp_results.sh
```

输出目录：

```text
rescore_result/all_methods_direct_grasp_oracle_crop_predicted_vcot/
  summary.csv
  checkpoint_manifest.csv
  analysis/README.md
  analysis/overview/main_summary.csv
  analysis/methods/direct_grasp/*.csv
  analysis/methods/direct_grasp/sweeps/*.csv
  analysis/methods/oracle_crop/*.csv
  analysis/methods/oracle_crop/sweeps/*.csv
  analysis/methods/oracle_crop/diagnostics/*.csv
  analysis/methods/predicted_vcot/*.csv
  analysis/methods/predicted_vcot/sweeps/*.csv
  analysis/methods/predicted_vcot/diagnostics/*.csv
  analysis/comparisons/direct_vs_oracle_crop.csv
  analysis/comparisons/direct_vs_predicted_vcot.csv
  analysis/manifest.json

rescore_result/task_direct_grasp_original_image/
  summary.csv
  checkpoint_manifest.csv
  analysis/README.md
  analysis/overview/main_summary.csv
  analysis/methods/direct_grasp/*.csv
  analysis/methods/direct_grasp/sweeps/*.csv
  analysis/manifest.json

rescore_result/task_oracle_crop_gt_mask_crop/
  summary.csv
  checkpoint_manifest.csv
  analysis/README.md
  analysis/overview/main_summary.csv
  analysis/methods/oracle_crop/*.csv
  analysis/methods/oracle_crop/sweeps/*.csv
  analysis/methods/oracle_crop/diagnostics/*.csv
  analysis/manifest.json

rescore_result/task_predicted_vcot_two_stage_predicted_crop/
  summary.csv
  checkpoint_manifest.csv
  analysis/README.md
  analysis/overview/main_summary.csv
  analysis/methods/predicted_vcot/*.csv
  analysis/methods/predicted_vcot/sweeps/*.csv
  analysis/methods/predicted_vcot/diagnostics/*.csv
  analysis/manifest.json
```

目录语义：

```text
all_methods_direct_grasp_oracle_crop_predicted_vcot:
  全量跨任务汇总，包含 direct/oracle crop/predicted VCoT 以及跨任务对比表。

task_direct_grasp_original_image:
  只包含 direct_grasp，也就是原图直接预测 grasp。

task_oracle_crop_gt_mask_crop:
  只包含 oracle_crop，也就是 GT mask crop upper bound。

task_predicted_vcot_two_stage_predicted_crop:
  只包含 predicted_vcot，也就是预测 bbox -> predicted crop -> grasp 的二阶段闭环。
```

`analysis/rescore_existing_grasp_results.sh` 默认会刷新全量跨任务 analysis、
三个 task-specific analysis，并为 oracle crop 生成 crop-frame 常量先验诊断。
可用这些开关控制：

```bash
RUN_TASK_DIRECT=False
RUN_TASK_ORACLE_CROP=False
RUN_TASK_PREDICTED_VCOT=False
RUN_CROP_FRAME_PRIOR=False
CROP_FRAME_PRIOR_SAMPLE_LIMIT=0  # 0 表示使用完整 train annotation 估计常量先验；默认 5000
```

rescore 参数契约：

```text
rescore 的本质:
  不做模型推理
  不加载 InternVL
  不重新生成 answer
  只读取已有 result JSON 的 outputs[].pred_norm / outputs[].grasp_id
  从 summary.checkpoint / summary.loaded_vcot_config 追溯 checkpoint 和配置
  从 LMDB 读取 GT positive grasp labels 后统一重算指标

默认指标阈值:
  vcot_iou_threshold = 0.25
  vcot_angle_threshold = 30.0
  image_size = 416

默认不改写原 result JSON:
  不传 --write
  不传 --out-dir

checkpoint_manifest.csv:
  每一行来自 result JSON 的 summary.checkpoint 和 summary.loaded_vcot_config。
  如果 checkpoint 或 config 不存在，manifest 生成会直接失败。

coverage:
  all_methods_direct_grasp_oracle_crop_predicted_vcot 覆盖 direct/crop/vcot 下所有 JSON。
  task_direct_grasp_original_image 只覆盖 result/vcot_grasp_direct 下的 JSON。
  task_oracle_crop_gt_mask_crop 只覆盖 result/vcot_grasp_crop 下的 JSON。
  task_predicted_vcot_two_stage_predicted_crop 只覆盖 result/vcot_grasp_vcot 下的 JSON。
```

`analysis/score_vcot_grasp_results.py` 逐文件逻辑：

```text
1. 读取一个或多个 eval result JSON。
2. 读取 outputs，并过滤 pred_norm is not None 的有效预测。
3. 根据 grasp_id 从 grasp_label_positive LMDB 读取该样本所有 GT positive grasp labels。
4. 将 pred_norm 反归一化到 416 尺度:
     cx/cy/w/h = int(norm * 416)
     angle = norm_angle * 180
5. 对每个 GT label 计算 rotated rectangle IoU 和 circular angle diff。
6. 写入或汇总:
     vcot_success
     vcot_success_rate_all
     vcot_success_rate_valid
     vcot_top1_success
     vcot_top1_success_rate_all
     vcot_top1_success_rate_valid
     vcot_joint_iou_mean
     vcot_joint_angle_diff_mean
     vcot_best_iou_mean
     vcot_best_angle_diff_mean
     target_label_count_mean
     target_label_count_max
```

official success 定义：

```text
任意 GT label 同时满足:
  rotated IoU >= 0.25
  circular angle diff <= 30 deg
```

top1 success 定义：

```text
只看第一个 GT label，同时满足:
  rotated IoU >= 0.25
  circular angle diff <= 30 deg
```

`score_vcot_grasp_results.py` 的输出控制：

```text
--summary-csv:
  输出 result 级汇总表，不输出样本级逐条明细。

--analysis-out-dir:
  继续调用 analyze_grasp_results.run_analysis()，生成汇总型 analysis CSV。

--write:
  将重算指标写回原 result JSON。

--out-dir:
  在指定目录写 rescored JSON copy。

默认 rescore_existing_grasp_results.sh 不传 --write / --out-dir，
因此不会改写原 result JSON。
```

`analysis/analyze_grasp_results.py` 输出语义：

```text
overview/main_summary.csv:
  当前 analysis 输入内每个 result 的主指标，包含 official/top1/strict/medium/loose/error mean 等。

methods/{direct_grasp,oracle_crop,predicted_vcot}/main_summary.csv:
  按方法拆分后的主指标。

methods/{direct_grasp,oracle_crop,predicted_vcot}/error_stats.csv:
  各误差指标的 mean/median/p75/p90。

methods/{direct_grasp,oracle_crop,predicted_vcot}/sweeps/geometry.csv:
  center/width-height/angle 阈值扫描。

methods/{direct_grasp,oracle_crop,predicted_vcot}/sweeps/iou.csv:
  IoU/angle 阈值扫描。

comparisons/direct_vs_oracle_crop.csv:
  direct/crop 在同 split 可配对样本上的交叉对比。

comparisons/direct_vs_predicted_vcot.csv:
  direct/predicted VCoT 在同 split 可配对样本上的交叉对比，包含 direct 成功
  predicted 失败、direct 失败 predicted 成功、rescued/broken/net gain 等计数和比例。

methods/oracle_crop/diagnostics/crop_quality_summary.csv:
  crop 质量指标按 all/official_success/official_fail 聚合。

methods/oracle_crop/diagnostics/crop_frame_prior_summary.csv:
  oracle crop 的 crop-frame 常量均值先验诊断。它从训练集 crop-frame target grasp
  估计一个常量均值向量，把该常量作为测试集预测反变换回原图后重新计算 official/top1，
  用于验证 crop_image 模型是否只是利用固定局部坐标先验。

methods/predicted_vcot/diagnostics/pipeline_summary.csv:
  predicted VCoT 的 bbox/crop/object_coverage/good crop/bad crop 局部区域诊断。
  当前会覆盖所有 pred_vcot result，包括最新 frame 配置
  `vcot_frame_lora16_lr8e-5_ep1_patch8_edge10_half40_bbox0.5` 和
  `vcot_frame_lora16_lr8e-5_ep1_patch6_edge5_half40_bbox0.5`。

README.md:
  当前 analysis 目录的指标和布局说明，会随 analysis 自动生成。

manifest.json:
  analysis 阈值、image_size、result_count 和输出文件清单。
```

analysis 每次从空输出目录重建，不保留或迁移旧版 analysis 文件；如果只输入单任务
results，跨方法互相比对目录不会生成。

`analysis/collect_checkpoint_manifest.py` 逻辑：

```text
从 result JSON summary 读取:
  summary.checkpoint
  summary.loaded_vcot_config
  summary.evaluation_mode

打开 loaded_vcot_config 读取:
  experiment_name
  output_dir

oracle_crop / predicted_vcot 额外要求 result summary 里已有:
  target_coordinate_frame
  bbox_edge_expand
  min_bbox_half_size
  target_grasp_index

输出:
  checkpoint_manifest.csv

失败策略:
  checkpoint 不存在则失败
  loaded_vcot_config 不存在则失败
  evaluation_mode 缺失或不支持则失败
  当前 schema 必需字段缺失则失败，不再从旧 result/config 兜底兼容
```

这个失败策略是有意设计，用来防止 analysis 引用 stale 或不存在的 checkpoint/config。

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

## Current Validation State

最后一次完整性校验结果：

```text
all_methods_direct_grasp_oracle_crop_predicted_vcot: rows=40 manifest=40 missing=0 stale=0 manifest_match=True
task_direct_grasp_original_image: rows=16 manifest=16 missing=0 stale=0 manifest_match=True
task_oracle_crop_gt_mask_crop: rows=12 manifest=12 missing=0 stale=0 manifest_match=True
task_predicted_vcot_two_stage_predicted_crop: rows=12 manifest=12 missing=0 stale=0 manifest_match=True
vcot_config_result_errors=0
predicted_vcot diagnostics: rows=12
oracle_crop crop_frame_prior: rows=12
comparisons/direct_vs_predicted_vcot: cross_rows=768
```

含义：

```text
全量 analysis 覆盖当前 40 个 result JSON。
direct/oracle crop/predicted VCoT 单任务目录分别覆盖当前 16/12/12 个 result JSON。
summary.csv 和 checkpoint_manifest.csv 的 result_path 集合一致。
VCoT result summary 与 checkpoint 内 vcot_config.json 参数一致。
predicted VCoT 局部区域诊断覆盖全部 12 个 pred_vcot result。
oracle crop 常量均值先验诊断覆盖全部 12 个 oracle_crop result。
direct-vs-predicted VCoT 交叉表覆盖 direct 与 pred_vcot 的同 split 可配对样本。
analysis 不再输出样本级逐条 CSV。
direct baseline 和 oracle crop object 的重复 result JSON 已删除，保留 hparam canonical 结果。
```

已运行通过：

```bash
python -m py_compile ...
bash -n ...
git diff --check
conda run --no-capture-output -n 260513-internvl bash analysis/rescore_existing_grasp_results.sh
```

## Current Reference Results

当前结果来自最新全量 rescore：

```text
rescore_result/all_methods_direct_grasp_oracle_crop_predicted_vcot/analysis/overview/main_summary.csv
result_count = 40
direct       = 16 result JSON
oracle_crop  = 12 result JSON
pred_vcot    = 12 result JSON
```

按 split 分别取 official/top1 最优：

| method | split | best official | official | best top1 | top1 |
| --- | --- | --- | ---: | --- | ---: |
| direct | seen | `lora16_lr8e-5_ep1_patch6` | 72.47 | `lora16_lr4e-5_ep2_patch6` | 49.73 |
| direct | unseen | `lora16_lr8e-5_ep1_patch6` | 53.87 | `lora16_lr8e-5_ep1_patch6` | 33.83 |
| oracle crop | seen | `crop_frame_lora16_lr8e-5_ep1_patch8_edge10_half40` | 77.90 | `crop_frame_lora16_lr8e-5_ep1_patch8_edge10_half40` | 59.40 |
| oracle crop | unseen | `crop_frame_lora16_lr8e-5_ep1_patch6_edge10_half40` | 68.93 | `crop_frame_lora16_lr8e-5_ep1_patch6_edge5_half40` | 49.97 |
| predicted VCoT | seen | `vcot_frame_lora16_lr8e-5_ep1_patch6_edge5_half40_bbox0.5` | 75.50 | `vcot_frame_lora16_lr8e-5_ep1_patch6_edge5_half40_bbox0.5` | 57.23 |
| predicted VCoT | unseen | `vcot_frame_lora16_lr8e-5_ep1_patch8_edge10_half40_bbox0.5` | 56.83 | `vcot_frame_lora16_lr8e-5_ep1_patch6_edge5_half40_bbox0.5` | 39.07 |

当前结论：

```text
predicted VCoT > direct:
  unseen official: 56.83 vs 53.87 (+2.96 pp)
  unseen top1:     39.07 vs 33.83 (+5.24 pp)

oracle crop 仍是 upper bound:
  unseen official: 68.93
  unseen top1:     49.97
```

## Development Checks

修改 shell/Python 后建议跑：

```bash
python -m py_compile scripts/ensure_grasp_data.py scripts/grasp_config.py \
  analysis/analyze_grasp_results.py \
  analysis/collect_checkpoint_manifest.py \
  analysis/diagnose_crop_frame_prior.py \
  analysis/score_vcot_grasp_results.py \
  data_tools/prepare_grasp_anything_direct.py \
  data_tools/prepare_grasp_anything_crop.py \
  data_tools/prepare_grasp_anything_bbox.py \
  data_tools/prepare_grasp_anything_vcot.py

bash -n scripts/grasp_run_common.sh \
  analysis/rescore_existing_grasp_results.sh \
  run_grasp_direct_hparam_sweep.sh \
  run_grasp_crop_design_sweep.sh \
  run_grasp_vcot_design_sweep.sh \
  eval/eval_grasp_direct_lmdb_lora.sh \
  eval/eval_grasp_crop_lmdb_lora.sh \
  eval/eval_grasp_vcot_lmdb_lora.sh

git diff --check
```

更详细的工程约定见 `info/grasp_pipeline_overview.md`。
