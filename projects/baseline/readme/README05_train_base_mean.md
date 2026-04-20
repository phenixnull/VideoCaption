
# 训练脚本使用说明（`train_base_mean.py`，Mean-Pooling → TransformerDecoder → LM Head）

本文档介绍如何使用 **mean pooling + 3 层 Transformer 解码器** 的 Caption 基线完成训练（仅交叉熵监督）。数据为 **CLIP 全局视觉特征** + 文本 **77 长度**（CLIP 标准），训练时对文本做 **右移一位** 的 teacher forcing（前 76 做输入、后 76 做监督）。

---

## 1. 环境与依赖

- Python ≥ 3.8，PyTorch（推荐 CUDA）
- 建议安装 `tqdm`, `matplotlib`, `transformers`（用于分词器）
- GPU：单卡/多卡均支持；多卡采用 **DDP（一卡一进程）**

---

## 2. 数据准备

### 2.1 视觉特征（pickle）
- 路径（示例）：`./datasets/MSRVTT/feats/ViT-B-32_k_split_ks12_features.pickle`
- 结构：字典 `video_id → (feat, mask)`
  - `feat`: `float32` 数组，形如 **[T, 512]**（T ≤ 20，已在数据准备阶段对齐/padding 到固定 T）  
  - `mask`: `int64` 数组，形如 **[T]**，其中 `1=有效帧, 0=PAD`  
- 数据集 `__getitem__` 返回（训练基线使用的关键字段）：
  - `vid_feat_tensor: [T,512]`，`vid_mask_tensor: [T]`
  - `caption_ids: [77]`，`caption_mask: [77]`（CLIP 分词，0 为 PAD） :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

### 2.2 文本标注（JSON）
- 路径（示例）：`./datasets/MSRVTT/MSRVTT_data.json`
- 训练划分：`video0 ~ video6512`；脚本会基于 `split='train'` 读取对应视频并随机抽取 caption 组成样本。:contentReference[oaicite:2]{index=2}

> **注意**：分词器默认使用 CLIP 版，句长固定 **77**；`pad_token_id=0`，`attention_mask` 与 PAD 对齐。:contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}

---

## 3. 模型与张量形状

### 3.1 前向输入
- 视频：`video_feats [B, T, 512]`，`vid_mask [B, T]`
- 文本：`captions [B, 77]`，`caption_mask [B, 77]`（1=有效，0=PAD）

### 3.2 训练时的“右移一位”
- `input_ids = captions[:, :-1]`（前 76）  
- `target_ids = captions[:, 1:]`（后 76）  
- 模型输出 `logits` 形状 **[B, 76, vocab_size]**，与 `target_ids [B,76]` 对齐用于 CE。:contentReference[oaicite:5]{index=5} :contentReference[oaicite:6]{index=6}

### 3.3 视频侧
- 利用 `vid_mask` 做 **Masked Mean Pooling**，得到单一 memory token：**[B, 1, 512]**，在 **每一层** decoder 的 cross-attention 中参与。:contentReference[oaicite:7]{index=7}

### 3.4 损失函数
- `CrossEntropy(logits, target_ids, ignore_index=0)`；其中 `0` 是 PAD（与 CLIP 分词器设置一致）。:contentReference[oaicite:8]{index=8}

---

## 4. 训练脚本用法

> 下列命令基于你补全后的 `train_base_mean.py`。若你用的训练脚本参数名不同，请以实际代码为准。

### 4.1 单卡示例
```bash
python train_base_mean.py \
  --dataset_type msrvtt \
  --batch_size 64 \
  --epochs 15 \
  --lr 1e-4 \
  --device cuda:0
````

### 4.2 多卡 DDP（单机）

使用 `torchrun` 启动，一卡一进程（更快更稳）：

```bash
# 4 卡：0,1,2,3；每卡 batch=64，accum_steps=2 → 有效批量 = 64×4×2 = 512
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train_base_mean.py \
  --ddp 1 \
  --gpus "0,1,2,3" \
  --batch_size 64 \
  --accum_steps 2 \
  --epochs 15 \
  --lr 5e-4
```

> **有效 batch** = `每卡 batch × 卡数 × 累积步数`。可按有效批量 **线性缩放学习率**。
> 建议主实验使用有效 batch 128 或 256；24G 显存常用配置：`B=64, accum=2 → 128` 或 `B=64, accum=4 → 256`。

---

## 5. 训练输出与可视化

训练过程中会在 `--out_dir`（默认 `./runs/base_mean/...`）生成如下文件：

* `loss_curve.png`：逐 step 训练损失曲线
* `loss_curve_ma{W}.png`：**移动平均**（窗口 W，默认 200）后的损失曲线
* `metrics.jsonl`：逐 step 指标（loss、acc、ppl、tokens、time）
* `epoch_metrics.csv`：逐 epoch 汇总（avg\_loss、ppl、step tokens）
* `checkpoints/epoch_XXX.pt`：模型、优化器、AMP scaler 与训练状态快照（可断点续训）

> `acc` 为 **token-level 准确率**（忽略 PAD），`ppl = exp(loss)`，与上面的 CE 对齐（target 为后 76）。

---

## 6. 常见启动参数（示例）

* `--dataset_type {msrvtt, msvd}`：选择数据集（默认 msrvtt）
* `--clip_global_vision_feats_path`：视觉特征 pickle 路径
* `--annotations_path`：JSON 标注路径
* `--batch_size`：**每卡** batch
* `--accum_steps`：梯度累积步数（将多次小 batch 的梯度相加，等效更大的有效 batch）
* `--epochs` / `--lr` / `--weight_decay`
* `--ddp {0/1}`：是否启用分布式训练
* `--gpus "0,1,2,3"`：可见 GPU 列表（字符串）
* `--device cuda:{id}`：单卡模式的设备号
* `--out_dir`：日志与模型输出根目录
* `--ma_window`：移动平均窗口（绘图）

---

## 7. 关键对齐检查清单

1. **输入维度**：视觉特征最后维度必须等于 `d_model=512`；否则需在模型外加 `Linear(Din→512)` 再送入。
2. **Caption 长度**：固定 **77**；训练时取 **前 76 / 后 76**。
3. **Loss 忽略**：`ignore_index=0`（与分词时 PAD=0 保持一致）。
4. **分布式采样**：DDP 训练时使用 `DistributedSampler` 且每个 epoch 调 `set_epoch(epoch)`（已在脚手架实现）。
5. **学习率/步数**：按 **有效 batch** 缩放学习率；scheduler/warmup 建议以 **优化器更新次数** 为单位（累积会减少每 epoch 的更新次数）。

---

## 8. 典型训练日志期望

* `loss` 平稳下降，`ppl` 同步下降；
* `acc`（token-level）上升；
* `loss_curve_ma{W}.png` 比原始曲线更平滑，便于观察趋势。

---

## 9. 附：张量形状速查（训练时）

* `video_feats`：`[B, T, 512]`
* `vid_mask`：`[B, T]`（1=有效）
* `captions`：`[B, 77]` ⇒ `input_ids=[B,76]`，`target_ids=[B,76]`
* `logits`：`[B, 76, vocab_size]`
* `CE(ignore_index=0)` 对齐 `target_ids`（后 76）

（教师强迫与输出对齐的细节请参考模型 forward 的实现注释。）

---

## 10. 参考实现片段

* **Dataset（MSRVTT）**：返回 `(vid_feat, vid_mask, caption_ids, caption_mask, caption_str, vid, sen_id)`；字幕编码使用 CLIP 分词器、固定 77 长度。&#x20;
* **Model Forward**：前 76 → 输入；后 76 → 监督；输出 `[B,76,V]`。&#x20;

---

如需我把 **评估/推理脚本** 或 **TensorBoard 可视化** 也补上，告诉我你的偏好（单文件 vs. 分模块），我直接给到可运行版本。

```
```
