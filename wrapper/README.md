# Wrapper 说明

这个目录用于实验两种 InternVL2.5-1B 图像输入方式，并为后续引入 VCoTGrasp 的抓取训练流程做准备。

## 文件结构

```text
wrapper/
  internvl_wrapper.py
  demo_internvl.py
  siglip_embedding_wrapper.py
  demo_siglip_embedding.py
  pretrained/paligemma2-3b-mix-224/
  __init__.py
```

## 1. 原生 InternVL 路线

文件：

```text
internvl_wrapper.py
demo_internvl.py
```

核心类：

```python
NativeInternVLWrapper
```

功能：

使用 InternVL 原生的图像输入方式，也就是通过 InternVL 自带的 `<image>` / image context token 机制，把图像特征插入语言模型。

用途：

- 验证本地 `InternVL2_5-1B` 权重是否可正常加载
- 验证普通图像问答链路是否跑通
- 作为 baseline，对比后续 SigLIP prefix 路线

运行示例：

```bash
cd /home/2025201095KZJ1/code/VCoTGrasp/GP/wrapper
python demo_internvl.py
```

## 2. SigLIP Prefix 路线

文件：

```text
siglip_embedding_wrapper.py
demo_siglip_embedding.py
```

核心类：

```python
SigLIPPrefixInternVLWrapper
```

功能：

按照 VCoTGrasp 的思路，不使用 `<image>` 文本占位机制，而是显式构造图像隐向量：

```text
image
  -> PaliGemma2/SigLIP vision tower
  -> image_projector: Linear(1152 -> 896)
  -> image_embeds / sqrt(hidden_size)
  -> concat(image_embeds, text_embeds)
  -> InternVL/Qwen language_model.generate(inputs_embeds=...)
```

当前训练策略：

```text
SigLIP 视觉塔：始终冻结
image_projector：始终训练
InternVL/Qwen language_model：默认冻结，可通过参数解冻
```

注意：

`image_projector` 目前是新建的 `1152 -> 896` 线性层。它还没有训练，所以这条路线的推理结果不一定可靠。后续需要用 VCoTGrasp 数据训练 projector，必要时再解冻 Qwen/InternVL。

运行示例：

```bash
cd /home/2025201095KZJ1/code/VCoTGrasp/GP/wrapper
python demo_siglip_embedding.py
```

如果后续需要解冻语言模型：

```bash
python demo_siglip_embedding.py --train-language-model
```

## 预训练权重

### InternVL2.5-1B

默认路径：

```text
../InternVL/pretrained/OpenGVLab/InternVL2_5-1B
```

下载命令：

```bash
cd /home/2025201095KZJ1/code/VCoTGrasp/GP/InternVL
huggingface-cli download OpenGVLab/InternVL2_5-1B \
  --local-dir pretrained/OpenGVLab/InternVL2_5-1B \
  --local-dir-use-symlinks False
```

### PaliGemma2/SigLIP Vision Tower

默认路径：

```text
pretrained/paligemma2-3b-mix-224
```

这个目录内保存从 VCoTGrasp/PaliGemma2 搬运过来的 SigLIP 视觉塔权重。当前只使用其中的：

```text
vision_tower.*
```

不使用 PaliGemma2 的语言模型，也不直接复用它的 `multi_modal_projector`，因为 PaliGemma2 projector 是：

```text
1152 -> 2304
```

而 InternVL2.5-1B 的 Qwen hidden size 是：

```text
896
```

所以当前重新建立：

```text
1152 -> 896
```

的 projector。

## 后续计划

下一步建议在 `SigLIPPrefixInternVLWrapper` 基础上加入 VCoTGrasp 训练逻辑：

```text
image + "detect {obj_name}" -> bbox
image/crop + "grasp the {obj_name}" -> grasp
```

但训练前需要先准备数据格式，并训练 `image_projector`，否则模型还不能稳定理解 SigLIP prefix 图像嵌入。
