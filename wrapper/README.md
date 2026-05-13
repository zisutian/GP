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
  __init__.py

data_tools/
  prepare_grasp_anything_direct.py
  inspect_grasp_anything_direct.py

train_grasp_direct_lmdb_lora.sh
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

## Grasp-Anything Direct 数据

原始数据保持不动：

```text
../VCoT-Grasp-self/data/grasp_anything/lmdb/
```

GP 目录只保存轻量 manifest：

```text
data/vcot_grasp/direct/
  train.jsonl
  test_seen.jsonl
  test_unseen.jsonl
  internvl_meta_train.json
```

manifest 每行只保存 LMDB key 和目标名，不保存图片本体。

生成 manifest：

```bash
cd /home/2025201095KZJ1/code/VCoTGrasp/GP
conda run -n 260513-internvl python data_tools/prepare_grasp_anything_direct.py
```

检查一条训练样本：

```bash
conda run -n 260513-internvl python data_tools/inspect_grasp_anything_direct.py \
  --manifest data/vcot_grasp/direct/train.jsonl \
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
InternVL/internvl_chat/internvl/train/vcot_direct_lmdb.py
InternVL/internvl_chat/internvl/train/internvl_chat_finetune.py
```

训练入口：

```bash
cd /home/2025201095KZJ1/code/VCoTGrasp/GP
bash train_grasp_direct_lmdb_lora.sh
```

这条路线不使用额外视觉编码器，也不使用 crop/bbox。它就是 direct grasp baseline。
