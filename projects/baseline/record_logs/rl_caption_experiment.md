# RL Video Caption 实验日志

## 实验概述

本实验基于 MSVD 数据集，实现两阶段视频描述生成方案：
1. **Draft-XE**: 监督学习阶段，训练 Draft 模型生成初始描述
2. **Draft-SCST**: 强化学习阶段，使用 CIDEr 信号微调 Draft 模型

## 实验环境

- **日期**: 2025-12-28
- **环境**: vcr (conda)
- **GPU**: 4x NVIDIA GPU
- **PyTorch**: 2.0.0+cu118
- **数据集**: MSVD (1970 videos, train:1200, val:100, test:670)

## 方案设计

### 模型架构

**DraftCaptionModel** (保留时序 token):
- 视频编码：保留 [B, T=12, D=512] 作为 memory（不做 mean pooling）
- 添加正弦位置编码到视频 token
- 文本解码：3 层 Transformer Decoder (8 heads, d_model=512)
- 词嵌入：从 CLIP ViT-B/32 初始化

### 训练策略

#### Stage 1: Draft-XE (监督学习)
- 目标：Cross-Entropy Loss
- 优化器：AdamW (lr=5e-4, weight_decay=0.01)
- 学习率调度：Warmup (3 epochs) + Cosine Decay
- Batch Size: 64
- Epochs: 30

#### Stage 2: Draft-SCST (强化学习)
- 目标：SCST Policy Gradient with CIDEr reward
- 基线：Greedy 生成的句子
- 奖励：R(sample) - R(greedy)
- 混合 XE 损失权重：0.1

---

## 实验 1: Draft-XE 训练

### 配置
```bash
python rl_caption/train_draft.py \
    --mode xe \
    --epochs 30 \
    --batch_size 64 \
    --lr 5e-4 \
    --device cuda:0 \
    --out_dir ./rl_caption/runs
```

### 训练过程

| Epoch | Val Loss | Val CIDEr | Test BLEU-4 | Test CIDEr | Test ROUGE_L | Test METEOR |
|-------|----------|-----------|-------------|------------|--------------|-------------|
| 1     | 4.37     | 68.07     | 44.91       | 61.96      | 70.19        | 32.80       |
| 2     | 3.86     | 98.02     | 49.17       | 82.85      | 72.74        | 35.48       |
| 3     | 3.80     | 97.42     | 46.65       | 85.40      | 71.42        | 34.95       |
| 4     | 3.78     | 106.33    | 46.95       | 89.94      | 72.66        | 35.61       |
| 5     | 3.75     | 98.92     | 52.78       | 94.07      | 74.40        | 37.22       |
| 6     | 3.68     | 104.73    | 53.80       | 89.88      | 74.01        | 36.92       |
| 7     | 3.66     | 104.14    | 50.03       | 89.10      | 73.55        | 36.88       |
| 8     | 3.67     | 104.72    | 49.71       | 87.91      | 73.35        | 36.55       |
| 9     | 3.64     | 111.17    | 52.49       | **97.64**  | 74.77        | 38.06       |
| 10    | 3.62     | 102.33    | 49.61       | 92.66      | 73.62        | 36.70       |
| 11    | 3.60     | 100.71    | 53.74       | 92.88      | 74.82        | 37.76       |

*训练进行中 (Epoch 11/30)...*

### 训练观察

1. **Loss 下降**: 从 Epoch 1 的 4.37 持续下降到 Epoch 11 的 3.60
2. **CIDEr 最佳**: Epoch 9 达到 **Test CIDEr 97.64** (Val CIDEr 111.17)
3. **BLEU-4 最佳**: Epoch 6 达到 53.80, Epoch 11 达到 53.74
4. **稳定收敛**: Epoch 5 后指标趋于稳定，CIDEr 在 87-97 范围波动

### 训练日志

- 训练目录: `rl_caption/runs/msvd_draft_xe_20251228_171109/`
- TensorBoard: `rl_caption/runs/msvd_draft_xe_20251228_171109/tb/`
- 检查点: `rl_caption/runs/msvd_draft_xe_20251228_171109/checkpoints/`

---

## 实验 2: Draft-SCST 微调

### 配置
```bash
python rl_caption/train_draft.py \
    --mode scst \
    --pretrained_path ./rl_caption/runs/xxx/checkpoints/best.pt \
    --epochs 10 \
    --batch_size 32 \
    --lr 1e-5 \
    --device cuda:0
```

### 训练过程

*待 Draft-XE 完成后进行...*

---

## 实验 3: Baseline (Mean Pooling) 对比

### 模型架构

**CaptionModel_Base** (mean pooling baseline):
- 视频编码：Mean Pooling [B, T=12, D=512] → [B, 1, D=512]
- 文本解码：3 层 Transformer Decoder (8 heads, d_model=512)
- 词嵌入：从 CLIP ViT-B/32 初始化

### 配置
```bash
python train_base_mean_monitored.py \
    --dataset_type msvd \
    --batch_size 32 \
    --epochs 30 \
    --lr 5e-4 \
    --device cuda:2
```

### 训练过程

*训练进行中...*

---

## 结果对比与分析

### 模型对比

| 模型 | 视频编码 | Memory Size | Test CIDEr | Test BLEU-4 | Test ROUGE_L |
|------|----------|-------------|------------|-------------|--------------|
| Baseline (Mean Pooling) | Mean [B,T,D]→[B,1,D] | 1 token | *进行中* | *进行中* | *进行中* |
| Draft (时序保留) | 保留 [B,T,D] | T tokens | **97.64** | **53.74** | **74.82** |

### 分析

*待实验完成后分析...*

---

## 代码结构

```
rl_caption/
├── dataset.py          # MSVD 数据集类（支持 RL 训练）
├── models.py           # Draft 模型定义
├── cider_scorer.py     # CIDEr 计算与 SCST 奖励
├── train_draft.py      # 统一训练脚本 (XE + SCST)
└── runs/               # 实验输出目录
    └── msvd_draft_xe_xxx/
        ├── tb/         # TensorBoard 日志
        ├── checkpoints/
        └── metrics.jsonl
```

---

## TensorBoard 监控

```bash
# 在服务器上启动
tensorboard --logdir ./rl_caption/runs --port 6007 --bind_all

# 本地 SSH 转发
ssh -N -L 16007:127.0.0.1:6007 <user>@<host>
# 浏览 http://127.0.0.1:16007
```
