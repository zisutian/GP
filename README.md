# VCoTGrasp GP

本目录是 Grasp-Anything / VCoT-style grasp 实验的工程入口，当前目标是用
`InternVL2.5-1B` 的原生图像输入做三条 grasp pipeline 的训练、评估和汇总分析。

核心约定：

```text
原始数据位置只在根目录 grasp_settings.py 里指定：
GRASP_DATASET_ROOT -> origin_split/*.csv + lmdb/

GP 目录只保存：
data_index / meta / config / train output / eval result / analysis
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
  -> data_index (*.jsonl)
  -> internvl_meta_train.json
  -> train 写 vcot_config.json
  -> checkpoint 复制 vcot_config.json
  -> eval 从 checkpoint/父目录读取 vcot_config.json
  -> result JSON
  -> analysis CSV
```

几个名字的含义：

```text
data_index:
  *.jsonl 样本清单。每行保存 image_key / grasp_key / mask_key / obj_name / crop 参数等。
  它不是图像数据本体，也不保存 crop 图像。

internvl_meta_train.json:
  InternVL 训练入口读取的数据集配置。
  它指向 data_index，并声明 vcot_dataset、repeat_time、vcot_loss_weight、crop 参数等。

vcot_config.json:
  每个实验/checkpoint 的可回溯配置。
  eval/analysis 依赖它确认 pipeline、data_index_root、meta_path、crop 参数、bbox_ratio、bbox_loss_weight 等。
```

路径和实验名统一从根目录 `grasp_settings.py` 生成。shell 入口只 source
根目录 `grasp_paths.sh`，不再各自维护路径。`scripts/` 只保留实验元数据写入和
训练/eval/sweep 运行期 helper；数据索引生成统一在 `data_tools/`。
实验表按“实验名 + 显式参数”写在 `DIRECT_EXPERIMENTS / CROP_EXPERIMENTS /
VCOT_EXPERIMENTS` 中；脚本不会从实验名反解析参数。

配置层级原则：

```text
grasp_settings.py:
  唯一默认配置源。长期配置、实验表、GPU、路径、eval split/threshold/port 都在这里改。

run_grasp_*.sh:
  编排层。读取 GRASP_* 配置，做 data check、checkpoint/result 跳过判断，并把解析后的
  具体参数传给 train/eval。

train/eval shell:
  执行层。被 run 层调用时使用 run 层传入的具体参数；单独运行时才回退到 GRASP_* 默认值。
```

环境变量覆盖只用于一次性运行或 run -> train/eval 的层间传参；可复现实验的默认值应写回
`grasp_settings.py`。

```text
DATASET_ROOT       原始 Grasp-Anything 数据根目录
DATA_INDEX_ROOT    轻量数据索引和 internvl_meta_train.json
RUN_ROOT           训练输出、checkpoint、TensorBoard
RESULT_ROOT        推理评测 JSON
ANALYSIS_ROOT      汇总分析和按任务拆分的检测结果表
```

默认目录结构：

```text
artifacts/data_index/direct/
  train.jsonl
  test_seen.jsonl
  test_unseen.jsonl
  internvl_meta_train.json

artifacts/data_index/bbox/
  train.jsonl
  test_seen.jsonl
  test_unseen.jsonl
  internvl_meta_train.json

artifacts/data_index/crop/{experiment}/
  train.jsonl
  test_seen.jsonl
  test_unseen.jsonl
  internvl_meta_train.json

artifacts/data_index/vcot/{experiment}/
  crop/train.jsonl
  test_seen.jsonl
  test_unseen.jsonl
  internvl_meta_train.json
```

`direct` 和 `bbox` 没有 crop 设计参数，可以共享。`crop` 和 `vcot` 的
data_index 直接挂在实验名下面，避免不同 crop 设置共用同一个 `train.jsonl`。

data stage 需要显式运行；train/eval 只检查已有 data_index/meta，不会自动生成。
一次性生成当前 `grasp_settings.py` 中全部实验需要的数据索引：

```bash
bash data_tools/run_grasp_data.sh
```

也可以只生成单个任务/实验：

```bash
python data_tools/ensure_grasp_data.py direct --splits train test_seen test_unseen
python data_tools/ensure_grasp_data.py crop --splits train test_seen test_unseen
python data_tools/ensure_grasp_data.py vcot --eval-splits test_seen test_unseen
```

如果文件缺失会自动从 `origin_split/*.csv` 重建；如果已有 crop data_index
的参数和当前实验参数不匹配，会报错而不是静默覆盖。

## Important Files

公共工具：

| 文件 | 作用 |
| --- | --- |
| `grasp_settings.py` | 唯一路径和实验表配置入口 |
| `grasp_paths.sh` | shell 入口 source 后导出路径变量 |
| `scripts/grasp_experiment_metadata.py` | 写入实验目录和 checkpoint 下的 `vcot_config.json` |
| `scripts/grasp_runtime_helpers.sh` | train/eval/sweep 共用的运行期 helper：阶段显示、checkpoint/result 跳过逻辑 |

数据准备与懒加载：

| 文件 | 作用 |
| --- | --- |
| `data_tools/run_grasp_data.sh` | data stage 入口，一次性生成当前实验表需要的 data_index/meta |
| `data_tools/ensure_grasp_data.py` | 显式生成 data_index 和 `internvl_meta_train.json` |
| `data_tools/check_grasp_data.py` | 只检查 data_index/meta，不生成文件 |
| `data_tools/prepare_grasp_anything_direct.py` | 生成 direct data_index/meta |
| `data_tools/prepare_grasp_anything_crop.py` | 生成 oracle crop data_index/meta |
| `data_tools/prepare_grasp_anything_bbox.py` | 生成 bbox detection data_index/meta |
| `data_tools/prepare_grasp_anything_vcot.py` | 生成 VCoT joint train meta 与 eval data_index |
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
1. 检查 data_index/meta；缺失时报错，需先显式运行 data stage
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
  data_index_root / meta_path / crop_root / bbox_root
  use_llm_lora / learning_rate / num_train_epochs
  max_dynamic_patch / force_image_size
  bbox_ratio / bbox_loss_weight / bbox_edge_expand / min_bbox_half_size
  target_coordinate_frame / target_grasp_index

predicted_vcot 的联合训练损失写作:
  L = L_grasp + bbox_loss_weight * L_bbox

其中 `bbox_ratio` 只控制 bbox 数据集的 `repeat_time`/采样配比，`bbox_loss_weight`
才是 bbox token loss 的显式权重。VCoT 以 crop 实验中更优的 `crop_image`
坐标系为主基准：`baseline_frame_*`。当前配置围绕该 frame baseline 做
`bbox_ratio in {0.25, 0.5, 1.0}` 和 `bbox_loss_weight in {0.5, 1.0, 2.0}`
消融；`vcot_full_*` 系列只保留 `bbox_ratio` 对照，不做 lambda 调整。

train 完成后会再次写 vcot_config.json，并复制到所有 checkpoint-* 下。
eval 只需要 checkpoint；data_index_root 和 crop 参数从 checkpoint 的 vcot_config.json 读取。
analysis 以 result summary 里的 checkpoint + loaded_vcot_config 为准。
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
baseline_full_lora{r}_lr{lr}_ep{ep}_patch6_edge15_half50{g}
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
baseline_frame_lora16_lr8e-5_ep1_patch6_edge5_half40_bbox0.5
vcot_frame_lora16_lr8e-5_ep1_patch6_edge5_half40_bbox0.25
vcot_frame_lora16_lr8e-5_ep1_patch6_edge5_half40_bbox1.0
vcot_frame_lora16_lr8e-5_ep1_patch6_edge5_half40_bbox0.5_lambda0.5
vcot_frame_lora16_lr8e-5_ep1_patch6_edge5_half40_bbox0.5_lambda2.0
vcot_frame_lora16_lr8e-5_ep1_patch6_edge10_half40_bbox0.5
vcot_frame_lora16_lr8e-5_ep1_patch8_edge10_half40_bbox0.5
vcot_full_lora16_lr8e-5_ep1_patch6_edge15_half50_bbox0.25
vcot_full_lora16_lr8e-5_ep1_patch6_edge15_half50_bbox0.5
vcot_full_lora16_lr8e-5_ep1_patch6_edge15_half50_bbox1.0
```

临时运行覆盖：

```bash
RUN_EVAL=0                     # 只训练，不 eval
OVERWRITE_OUTPUT_DIR=True      # 强制重训
OVERWRITE_EVAL_RESULTS=True    # 强制重评估
EVAL_DATASETS=test_seen        # sweep 只评估一个 split
DATASETS=test_seen             # 单独 eval 只评估一个 split
CUDA_VISIBLE_DEVICES=0,1
GPUS=2
BBOX_LOSS_WEIGHT=0.5           # 单独跑 predicted_vcot lambda 消融时覆盖 bbox loss 权重
```

这些变量不是默认配置入口；需要长期保留的 GPU、split、threshold、port 或实验参数应改
`grasp_settings.py`。

单独 eval 轻量入口按 baseline 模板写死一组可改默认值，不读取 `grasp_settings.py`：

```bash
conda run --no-capture-output -n 260513-internvl bash eval/eval_grasp_direct_lmdb_lora.sh
conda run --no-capture-output -n 260513-internvl bash eval/eval_grasp_crop_lmdb_lora.sh
conda run --no-capture-output -n 260513-internvl bash eval/eval_grasp_vcot_lmdb_lora.sh
```

也可以用环境变量临时覆盖模板里的 baseline 路径：

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
  grasp_loss_weight
  bbox_loss_weight
```

`pred_norm` 是归一化 grasp，rescore 会按 `image_size=416` 反归一化为
`[cx, cy, w, h, angle_deg]`。如果某个样本解析失败，`pred_norm` 为 `None`，会计入
all 分母，但不会进入 valid 分母。

单独评估 crop/VCoT hparam checkpoint 时，只需要指定 checkpoint 或 WORK_DIR：

```bash
WORK_DIR=artifacts/runs/vcot/{experiment} \
OUT_DIR=artifacts/results/vcot/{experiment} \
conda run --no-capture-output -n 260513-internvl bash eval/eval_grasp_vcot_lmdb_lora.sh
```

## Analysis

统一 analysis：

```bash
conda run --no-capture-output -n 260513-internvl \
  bash analysis/rescore_existing_grasp_results.sh
```

如果只想分析指定结果，写一个列表文件，每行一个 result JSON 路径：

```bash
RESULT_LIST=analysis/result_list.txt \
conda run --no-capture-output -n 260513-internvl \
  bash analysis/rescore_existing_grasp_results.sh
```

analysis 只读取已有 result JSON，不做模型推理，不改写原 result JSON。默认输出到
`artifacts/analysis/`；具体 CSV/manifest 由 analysis 脚本生成，不在 README 中维护结果快照。

analysis 输出组织：

```text
artifacts/analysis/all/
  summary.csv
  checkpoint_manifest.csv
  README.md
  manifest.json
  overview/
  methods/
    direct_grasp/
    oracle_crop/
    predicted_vcot/
  comparisons/

artifacts/analysis/direct_grasp/
  summary.csv
  checkpoint_manifest.csv
  overview/
  methods/direct_grasp/

artifacts/analysis/oracle_crop/
  summary.csv
  checkpoint_manifest.csv
  overview/
  methods/oracle_crop/

artifacts/analysis/predicted_vcot/
  summary.csv
  checkpoint_manifest.csv
  overview/
  methods/predicted_vcot/
```

文件语义：

```text
summary.csv:
  每个 result JSON 一行的指标汇总。

checkpoint_manifest.csv:
  每个 result JSON 对应的 checkpoint、work_dir、data_index_root、metadata 路径。

overview/:
  当前 analysis 输入集合的总览表。

methods/:
  按 evaluation_mode 拆开的方法内 summary、误差统计、sweep、diagnostics。

comparisons/:
  跨方法的同 split 可配对样本对比；只有输入同时包含对应方法时才生成。

manifest.json:
  analysis 输入、阈值和输出文件清单。
```

可用这些开关控制分析范围：

```bash
RUN_TASK_DIRECT=False
RUN_TASK_ORACLE_CROP=False
RUN_TASK_PREDICTED_VCOT=False
RUN_CROP_FRAME_PRIOR=False
CROP_FRAME_PRIOR_SAMPLE_LIMIT=0  # 0 表示使用完整 train annotation 估计常量先验；默认 5000
```

analysis 会根据 result JSON 的 `summary.checkpoint` 和 `summary.loaded_vcot_config`
回溯 checkpoint/metadata；缺失或过期会失败，以避免把 stale result 混入统计。

## Outputs

```text
artifacts/data_index/      data_index 和 internvl_meta_train.json
artifacts/runs/direct/{experiment}/
artifacts/runs/crop/{experiment}/
artifacts/runs/vcot/{experiment}/
artifacts/results/direct/{experiment}/
artifacts/results/crop/{experiment}/
artifacts/results/vcot/{experiment}/
artifacts/analysis/        analysis 输出
```

每个 result JSON 的 summary 会记录 checkpoint 和 loaded config。rescore 只依赖这些 summary 和 `vcot_config.json`，不再从目录名猜参数。

## Development Checks

修改 shell/Python 后建议跑：

```bash
python -m py_compile grasp_settings.py data_tools/ensure_grasp_data.py data_tools/check_grasp_data.py scripts/grasp_experiment_metadata.py \
  analysis/analyze_grasp_results.py \
  analysis/collect_checkpoint_manifest.py \
  analysis/diagnose_crop_frame_prior.py \
  analysis/score_vcot_grasp_results.py \
  data_tools/prepare_grasp_anything_direct.py \
  data_tools/prepare_grasp_anything_crop.py \
  data_tools/prepare_grasp_anything_bbox.py \
  data_tools/prepare_grasp_anything_vcot.py

bash -n grasp_paths.sh scripts/grasp_runtime_helpers.sh \
  data_tools/run_grasp_data.sh \
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
