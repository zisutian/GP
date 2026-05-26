# Wrapper 说明

当前只保留 InternVL2.5 原生图像输入路线，用来做最简单的 direct grasp baseline。

目标：

```text
原图 + "grasp the {obj_name}" -> <loc....><loc....><loc....><loc....><loc....>
```

输出 5 个 `<loc>` token，对应归一化抓取框：

```text
[x, y, w, h, angle]
```

## 文件结构

```text
wrapper/
  internvl_wrapper.py
  demo_internvl.py
  plot_direct_predicted_results.py
  __init__.py

data_tools/
  prepare_grasp_anything_direct.py
  inspect_grasp_anything_direct.py
```

## InternVL 原生推理

文件：

```text
internvl_wrapper.py
demo_internvl.py
```

核心类：

```python
NativeInternVLWrapper
```

用途：

- 加载 `InternVL2_5-1B`
- 使用 InternVL 原生 `<image>` / image context token 机制
- 验证普通图像问答和后续 direct grasp 推理

运行：

```bash
cd /home/2025201095KZJ1/code/VCoTGrasp/GP/wrapper
python demo_internvl.py
```

## Direct / Predicted VCoT 结果绘图

文件：

```text
plot_direct_predicted_results.py
```

用途：

- 读取 `artifacts/analysis/all/summary.csv`
- 生成 direct 与 predicted VCoT 的成功率对比图
- 生成 predicted VCoT 的 bbox/crop 诊断图
- 从已有结果 JSON 和图像 LMDB 中抽样生成预测框可视化图

运行：

```bash
cd /home/2025201095KZJ1/code/VCoTGrasp/GP
python wrapper/plot_direct_predicted_results.py
```

默认输出：

```text
wrapper/direct_predicted_figures/
```

Pair 图可手动筛选。首次运行或加 `--refresh-pair-config` 会生成候选配置：

```text
wrapper/direct_predicted_figures/examples/paired_example_config.csv
```

编辑其中的 `show` 列即可控制哪些 pair 显示：`1` 表示显示，`0` 表示隐藏。再次运行脚本时不要加 `--refresh-pair-config`，脚本会读取你修改后的配置。当前支持的 pair 类别：

```text
random        # 从 direct/predicted 共有样本里随机抽
pred_rescue   # direct 失败，predicted VCoT 成功
direct_only   # direct 成功，predicted VCoT 失败
both_success  # 两者都成功
both_fail     # 两者都失败
```

只生成某几类候选也可以：

```bash
python wrapper/plot_direct_predicted_results.py \
  --refresh-pair-config \
  --paired-categories pred_rescue,both_success \
  --paired-per-category 4 \
  --paired-candidate-limit 30
```

生成随机样本加 direct 失败 / predicted 成功样本：

```bash
python wrapper/plot_direct_predicted_results.py \
  --refresh-pair-config \
  --paired-categories random,pred_rescue \
  --paired-per-category 4 \
  --paired-candidate-limit 50 \
  --pair-random-seed 7
```

## Grasp-Anything Direct 数据

原始数据保持不动：

```text
grasp_settings.py: GRASP_DATASET_ROOT/lmdb/
```

GP 目录只保存轻量 data_index：

```text
artifacts/data_index/direct/
  train.jsonl
  test_seen.jsonl
  test_unseen.jsonl
  internvl_meta_train.json
```

data_index 每行只保存 LMDB key 和目标名，不保存图片本体。

生成 data_index；训练和评估入口也会在缺失时自动重建默认 data_index：

```bash
cd /home/2025201095KZJ1/code/VCoTGrasp/GP
conda run -n 260513-internvl python data_tools/prepare_grasp_anything_direct.py
```

检查一条训练样本：

```bash
conda run -n 260513-internvl python data_tools/inspect_grasp_anything_direct.py \
  --manifest artifacts/data_index/direct/train.jsonl \
  --index 0
```

输出 conversation 形如：

```json
[
  {
    "from": "human",
    "value": "<image>\ngrasp the remote"
  },
  {
    "from": "gpt",
    "value": "<loc0512><loc0662><loc0391><loc0107><loc0067>"
  }
]
```

## 训练

InternVL 训练代码已加入 direct LMDB 读取分支：

```text
data_tools/vcot_direct_lmdb.py
InternVL/internvl_chat/internvl/train/internvl_chat_finetune.py
```

推荐从 hparam sweep 启动训练：

```bash
cd /home/2025201095KZJ1/code/VCoTGrasp/GP
bash run_grasp_direct_hparam_sweep.sh
```

底层脚本在 `InternVL/internvl_chat/shell/internvl2.5/2nd_finetune/` 下；根目录不再保留额外的单次 train wrapper。这条路线不使用额外视觉编码器，也不使用 crop/bbox。它就是 direct grasp baseline。
