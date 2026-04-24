# Video Caption Reconstruction 项目详细理解文档

## 一、项目概述

### 1.1 项目名称与目标
- **项目名称**: VideoCaption_Reconstruction (VCR)
- **核心任务**: 视频描述生成 (Video Captioning)
- **目标**: 基于 CLIP 视觉特征，使用 Transformer Decoder 生成视频的自然语言描述

### 1.2 数据集
- **主要数据集**: MSR-VTT (Microsoft Research Video to Text)
  - 10,000 个视频
  - 每个视频 20 条人工标注的描述
  - 数据划分:
    - Train: video0 ~ video6512 (6,513 个视频, 130,260 条描述)
    - Val: video6513 ~ video7009 (497 个视频, 9,940 条描述)
    - Test: video7010 ~ video9999 (2,990 个视频, 59,800 条描述)

### 1.3 技术栈
- **深度学习框架**: PyTorch 2.0 + CUDA 11.8
- **预训练模型**: OpenAI CLIP (ViT-B/32)
- **分词器**: HuggingFace CLIPTokenizer (词表大小: 49,408)
- **评估指标**: BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr, SPICE
- **分布式训练**: PyTorch DDP (DistributedDataParallel)

---

## 二、项目架构

### 2.1 目录结构
```
project/
├── dataloaders/                    # 数据加载模块
│   ├── dataset_msrvtt_feats.py     # 基于预提取特征的数据集
│   ├── dataset_msrvtt_precomputed_feats.py  # 预计算特征数据集
│   ├── dataset_msrvtt_raw.py       # 原始视频数据集
│   └── rawvideo_util.py            # 视频帧提取工具
├── models/                         # 预训练模型权重
│   ├── clip_models/                # CLIP 模型权重
│   └── clip_tokenizer/             # CLIP 分词器
├── runs/                           # 训练输出目录
│   ├── base_mean_ks12/             # 基础模型 (k_split=12)
│   ├── base_mean_CL_ks12/          # 对比学习版本
│   └── ...
├── eval/                           # 评估结果
│   ├── base/                       # 基础模型评估
│   └── lencontrol/                 # 长度控制模型评估
├── temp/                           # 临时文件
│   └── pycocoevalcap-master/       # 评估工具包
├── models.py                       # 核心模型定义
├── model_SRVC.py                   # SRVC 结构化视频描述模型
├── models_contrastive.py           # 对比学习模块
├── multi_label_classifier.py       # 多标签分类器
├── load_tokenizers.py              # 分词器加载与扩展
├── extract_clip_feats.py           # CLIP 特征提取
├── train_*.py                      # 训练脚本
├── infer_*.py                      # 推理脚本
├── evaluate.py                     # 评估脚本
└── test_*.py                       # 测试脚本
```

### 2.2 核心模块关系图
```
┌─────────────────────────────────────────────────────────────────┐
│                        数据流程                                  │
├─────────────────────────────────────────────────────────────────┤
│  原始视频 (.mp4)                                                 │
│      ↓                                                          │
│  extract_clip_feats.py (K-Split 采样 + CLIP 编码)               │
│      ↓                                                          │
│  预提取特征 (.pickle) [video_id: (feats[K,512], mask[K])]       │
│      ↓                                                          │
│  MSRVTT_FeaturesDataset (数据加载器)                            │
│      ↓                                                          │
│  CaptionModel_Base / CaptionModel_LenControl / CaptionModel_SRVC│
│      ↓                                                          │
│  生成描述文本                                                    │
│      ↓                                                          │
│  evaluate.py (BLEU, METEOR, ROUGE, CIDEr, SPICE)               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 三、核心模型详解

### 3.1 CaptionModel_Base (基础模型)

#### 架构设计
```
输入:
  - video_feats: [B, T, D]  视频帧特征 (T=12/20, D=512)
  - vid_mask: [B, T]        有效帧掩码 (1=有效, 0=padding)
  - captions: [B, 77]       文本 token IDs
  - caption_mask: [B, 77]   文本掩码

处理流程:
  1. 视频编码: Mean Pooling → [B, 1, D] (单 token memory)
  2. 文本嵌入: Word Embedding + Positional Embedding + LayerNorm
  3. Transformer Decoder (3层, 8头, batch_first=True)
  4. 输出投影: LayerNorm + Linear → [B, 76, vocab_size]

关键设计:
  - 从 CLIP 初始化 word_embeddings 和 lm_head
  - 支持冻结 word_embeddings (frozen_we=True)
  - Teacher Forcing 训练: input=captions[:, :-1], target=captions[:, 1:]
```

#### 代码位置
- 定义: `models.py` → `CaptionModel_Base`
- 训练: `train_base_mean.py`, `train_base_mean_monitored.py`
- 推理: `infer_base_mean.py`

### 3.2 CaptionModel_LenControl (长度控制模型)

#### 架构设计
```
在 CaptionModel_Base 基础上增加:
  1. Length Token: 在 BOS 后插入长度嵌入
     - h = (target_len - len_min) / (len_max - len_min)  ∈ [0, 1]
     - e_len = Linear(h)  → [B, 1, D]
  
  2. Type Embedding: 区分文本 token 和长度 token
     - type_id_text = 1
     - type_id_len = 2
  
  3. 解码器输入: [BOS, LEN_TOKEN, text_tokens...]
     - 输入长度从 76 变为 77
     - 损失计算: logits[:, 2:, :] 与 target 对齐

推理时可控制生成长度:
  - len_target_total: 目标 token 总数
  - len_h: 直接指定归一化长度
```

#### 代码位置
- 定义: `models.py` → `CaptionModel_LenControl`
- 训练: `train_base_mean_len_control_monitored.py`
- 推理: `infer_base_mean_len_control.py`

### 3.3 CaptionModel_Base_CL (对比学习版本)

#### 架构设计
```
在 CaptionModel_Base 基础上增加对比学习分支:

视频侧:
  - 离线 CLIP 特征 → Mask-aware Mean Pooling → L2 归一化
  
文本侧:
  - CLIP.encode_text(caption_str) → L2 归一化
  
对比损失:
  - 单向 v→t InfoNCE Loss
  - 可学习温度参数 (从 CLIP logit_scale 初始化)
  
总损失:
  L_total = L_CE + λ * L_v→t  (λ 默认 0.1)
```

#### 代码位置
- 定义: `models.py` → `CaptionModel_Base_CL`
- 训练: `train_base_mean_CL_monitored.py`

### 3.4 CaptionModel_SRVC (结构化视频描述模型)

#### 架构设计
```
三阶段流程:

阶段1: 对象检测 (VideoMultiLabelDetector)
  - 输入: [B, T, D] 视频特征
  - 输出: [B, vocab_size] 多标签预测
  - 聚合方式: Attention / Transformer / Mean / Max

阶段2: 短语生成 (PhraseDecoder)
  - 输入: 视频特征 + 检测到的对象
  - 输出: 对象短语 (如 "a man", "red car")
  - 使用 CLIPPhraseEncoder (第6层特征)

阶段3: 句子生成 (Initial/Refine Decoder)
  - Initial Decoder: 仅基于视频和短语生成初始句子
  - Refine Decoder: 结合初始句子进行精炼
  - 使用 CLIPSentenceEncoder (第12层特征)

Memory 构建:
  [Vision Tokens] + [OBJ_CLS] + [短语1] + [OBJ_SEP] + [短语2] + ... + [OBJ_END]
  
特殊 Token:
  - [OBJ_CLS]: 49409
  - [OBJ_END]: 49410
  - [OBJ_SEP]: 49411
  - [NO_OBJ]: 49412
  - [MASK]: 49408
```

#### 代码位置
- 定义: `model_SRVC.py` → `CaptionModel_SRVC`
- 训练: `train_msrvtt_srvc_monitored.py`

---

## 四、数据处理流程

### 4.1 视频特征提取

#### K-Split 采样策略
```python
# extract_clip_feats.py
def video_to_tensor(..., sample_type='k_split'):
    k = k_segments  # 默认 12 或 20
    segment_size = frameCount // k
    
    for i in range(k):
        start_frame = i * segment_size
        end_frame = (i + 1) * segment_size
        
        # 采样位置: start / mid / end / tsn(随机)
        if extract_loc == "start":
            target_frame = start_frame
        elif extract_loc == "mid":
            target_frame = (start_frame + end_frame) // 2
        ...
```

#### 特征保存格式
```python
# 保存为 pickle 文件
features = {
    'video0': (feats[K, 512], mask[K]),  # feats: float16, mask: int64
    'video1': (feats[K, 512], mask[K]),
    ...
}
# 路径: ../datasets/MSRVTT/feats/ViT-B-32_k_split_ks12_features.pickle
```

### 4.2 数据集类

#### MSRVTT_FeaturesDataset
```python
# dataloaders/dataset_msrvtt_feats.py
class MSRVTT_FeaturesDataset(Dataset):
    def __init__(self, features_path, json_path, split='train', tkz_type='clip'):
        # 加载预提取特征
        self.features = pickle.load(open(features_path, 'rb'))
        
        # 加载标注
        self.captions_data = {}  # {vid: [(cap, sen_id), ...]}
        
        # 构建数据列表
        # 训练: 每个 (vid, cap) 对作为一个样本
        # 测试: 每个视频只取第一条描述
    
    def __getitem__(self, idx):
        vid, caption, sen_id = self.data_list[idx]
        feat, vid_mask = self.features[vid]
        
        # Tokenize
        encoded = self.tokenizer.encode_plus(caption, ...)
        
        return (vid_feat_tensor, vid_mask_tensor, 
                caption_ids, caption_mask, 
                caption_label, vid, sen_id)
```

### 4.3 分词器

#### CLIPTokenizer_Custom
```python
# load_tokenizers.py
class CLIPTokenizer_Custom:
    """
    关键修改:
    1. 将 pad_token 的 ID 从 49409 改为 0
    2. 原 ID=0 的 token ('!') 改为 49409
    
    原因: 与 PyTorch 的 ignore_index=0 对齐
    """
    def __init__(self, swap_pad_token=True):
        self._tokenizer = load_clip_tokenizer()
        if swap_pad_token:
            self._swap_pad_token_with_id_0()
```

#### Tokenizer_M (可扩展版本)
```python
# load_tokenizers.py
class Tokenizer_M:
    """
    支持添加自定义 token:
    - [MASK]: 49408
    - [OBJ_CLS]: 49409
    - [OBJ_END]: 49410
    - [OBJ_SEP]: 49411
    - [NO_OBJ]: 49412
    
    扩展后词表大小: 49413
    """
```

---

## 五、训练流程

### 5.1 基础训练 (train_base_mean.py)

#### 命令示例
```bash
# 单卡训练
python train_base_mean.py \
    --dataset_type msrvtt \
    --clip_global_vision_feats_path ../datasets/MSRVTT/feats/ViT-B-32_k_split_ks12_features.pickle \
    --annotations_path ../datasets/MSRVTT/MSRVTT_data.json \
    --batch_size 128 \
    --epochs 50 \
    --lr 5e-4

# 多卡 DDP 训练
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    train_base_mean.py \
    --ddp 1 \
    --gpus "0,1,2,3" \
    --batch_size 64 \
    --accum_steps 2
```

#### 训练配置
```python
# 默认超参数
epochs = 50
batch_size = 128
lr = 5e-4
weight_decay = 0.01
accum_steps = 2
amp = True  # 混合精度

# 模型配置
d_model = 512
decoder_nhead = 8
num_layers = 3
init_we = True      # 从 CLIP 初始化 word embedding
init_lmhead = True  # 从 CLIP 初始化 lm_head
frozen_we = True    # 冻结 word embedding
frozen_lmhead = False
```

### 5.2 监控训练 (train_base_mean_monitored.py)

#### 新增功能
```python
# TensorBoard 监控
writer = SummaryWriter(log_dir=out_dir / "tb")
writer.add_scalar("train/loss", loss, global_step)
writer.add_scalar("train/acc", acc, global_step)
writer.add_scalar("train/ppl", ppl, global_step)

# 每 epoch 验证评估
val_scores = evaluate_on_val(model, device, args, out_dir, epoch)
# 返回: BLEU-4, ROUGE_L, CIDEr, METEOR, VAL_LOSS

# 保存检查点
torch.save({
    "epoch": epoch,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scaler": scaler.state_dict(),
    "args": vars(args),
    "global_step": global_step,
}, out_dir / "checkpoints" / f"epoch_{epoch:03d}.pt")
```

### 5.3 对比学习训练 (train_base_mean_CL_monitored.py)

#### 损失计算
```python
# 前向传播
logits, cl_loss, cl_acc = model(
    vid_feat, vid_mask, caption_ids, caption_mask,
    cap_str_list=cap_str_list,
    return_contrastive=True
)

# CE 损失
loss_ce = F.cross_entropy(logits, target, ignore_index=0)

# 总损失
loss_total = loss_ce + cl_weight * cl_loss  # cl_weight 默认 0.1
```

---

## 六、推理与评估

### 6.1 推理流程 (infer_base_mean.py)

#### 生成策略
```python
# Top-p (Nucleus) Sampling
def top_p_sampling(logits, top_p=0.9, temperature=1.0):
    probs = softmax(logits / temperature)
    sorted_probs, sorted_idx = sort(probs, descending=True)
    cum = cumsum(sorted_probs)
    mask = cum <= top_p
    mask[:, :1] = True  # 至少保留一个
    filtered = where(mask, sorted_probs, 0)
    next_ids = multinomial(filtered, num_samples=1)
    return sorted_idx.gather(next_ids)

# Beam Search
def beam_search_batch(model, vid_feat, vid_mask, tokenizer,
                      beam_size=5, max_new_tokens=76, alpha=0.7):
    # GNMT 长度惩罚
    def length_penalty(L, a):
        return ((5.0 + L) ** a) / ((5.0 + 1.0) ** a)
    ...
```

#### 输出格式
```json
// hyp.json
{
    "video7010": ["a man is talking about something"],
    "video7011": ["a woman is cooking in the kitchen"],
    ...
}

// ref.json
{
    "video7010": ["a man is speaking", "someone is talking", ...],
    "video7011": ["a woman cooks food", "cooking in kitchen", ...],
    ...
}
```

### 6.2 评估指标 (evaluate.py)

#### 支持的指标
```python
# BLEU-1/2/3/4: n-gram 精确率
bleu = Bleu(4)
bleu_score, _ = bleu.compute_score(gts, res)

# METEOR: 同义词匹配 + 词干还原
meteor = Meteor()
meteor_score, _ = meteor.compute_score(gts, res)

# ROUGE-L: 最长公共子序列
rouge = Rouge()
rouge_score, _ = rouge.compute_score(gts, res)

# CIDEr: TF-IDF 加权 n-gram 相似度
cider = Cider()
cider_score, _ = cider.compute_score(gts, res)

# SPICE: 场景图匹配 (需要 Java)
spice_score = _run_spice(gts, res, spice_mem="4G")
```

#### 评估命令
```bash
python evaluate.py \
    --hyp ./eval/base/msrvtt_test_base_mean_hyp.json \
    --ref ./eval/base/msrvtt_test_base_mean_ref.json \
    --spice_mem 16G
```

---

## 七、多标签分类器

### 7.1 VideoMultiLabelClassifier

#### 架构
```python
# multi_label_classifier.py
class VideoMultiLabelClassifier(nn.Module):
    """
    Pipeline:
    1. 时序聚合: [B, T, D] → [B, D]
       - TemporalAttentionPooling (可学习 query)
       - TransformerTemporalAggregator
       - SimpleTemporalPooling (mean/max/mean_max)
    
    2. 分类头: [B, D] → [B, vocab_size]
       - Linear + BatchNorm + ReLU + Dropout
       - Linear(hidden_dim, vocab_size)
    """
```

#### 损失函数
```python
class MultiLabelLoss(nn.Module):
    """
    支持:
    - BCE: 标准二元交叉熵
    - Focal Loss: 处理正负样本不平衡
    - ASL (Asymmetric Loss): 对正负样本使用不同 gamma
    """
```

---

## 八、关键技术点

### 8.1 CLIP 特征提取
- 使用 ViT-B/32 的视觉编码器
- K-Split 采样: 将视频均分为 K 段，每段取一帧
- 输出维度: [K, 512]

### 8.2 Mask 约定
```
数据集 mask: 1=有效, 0=padding (HuggingFace 风格)
PyTorch Transformer mask: True=忽略, False=有效

转换: key_padding_mask = ~mask.bool()
```

### 8.3 Teacher Forcing
```python
# 训练时
input_ids = captions[:, :-1]   # [B, 76] 前 76 个 token
target = captions[:, 1:]       # [B, 76] 后 76 个 token
logits = model(vid_feat, vid_mask, captions, caption_mask)  # [B, 76, V]
loss = cross_entropy(logits, target, ignore_index=0)
```

### 8.4 分布式训练
```python
# DDP 初始化
dist.init_process_group(backend="nccl")
model = DDP(model, device_ids=[local_rank])

# 数据采样
sampler = DistributedSampler(dataset, shuffle=True)
loader = DataLoader(dataset, sampler=sampler)

# 每 epoch 设置
sampler.set_epoch(epoch)
```

---

## 九、实验结果

### 9.1 基础模型 (Base Mean Pooling)
| 指标 | 值 |
|------|-----|
| BLEU-4 | ~40-42 |
| METEOR | ~28-30 |
| ROUGE-L | ~60-62 |
| CIDEr | ~45-50 |

### 9.2 长度控制模型
- 支持生成 4-70 个 token 的描述
- 长度控制精度: ±2 tokens

### 9.3 对比学习模型
- CIDEr 提升约 2-3 点
- 视频-文本对齐更好

---

## 十、环境配置

### 10.1 依赖安装
```bash
conda create --name vcr python=3.8.20
conda activate vcr

# PyTorch (GPU)
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# CLIP
pip install git+https://github.com/openai/CLIP.git

# 其他依赖
pip install transformers==4.46.3 pandas opencv-python matplotlib tensorboard

# 评估工具 (需要 Java 11+)
pip install pycocoevalcap
sudo apt-get install openjdk-11-jre-headless
```

### 10.2 数据准备
```bash
# 下载标注
wget https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msrvtt_data.zip

# 下载视频
wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip

# 目录结构
datasets/
└── MSRVTT/
    ├── raw/           # 10000 个 mp4 视频
    ├── feats/         # 预提取特征
    ├── MSRVTT_data.json
    └── msrvtt.csv
```

---

## 十一、总结

本项目是一个完整的视频描述生成系统，具有以下特点:

1. **模块化设计**: 数据加载、模型定义、训练、推理、评估各模块独立
2. **多种模型变体**: 基础模型、长度控制、对比学习、结构化生成
3. **工业级实现**: 支持 DDP 分布式训练、混合精度、TensorBoard 监控
4. **完整评估体系**: 支持 BLEU、METEOR、ROUGE、CIDEr、SPICE 等标准指标
5. **可扩展性**: 支持自定义 token、多标签分类、预提取特征加速

项目代码结构清晰，注释详尽，适合作为视频描述生成任务的研究基础。


---

## 附录 A: 文件功能速查表

| 文件名 | 功能描述 |
|--------|----------|
| **核心模型** ||
| `models.py` | 基础模型定义 (CaptionModel_Base, CaptionModel_LenControl, CaptionModel_Base_CL) |
| `model_SRVC.py` | 结构化视频描述模型 (CaptionModel_SRVC) |
| `models_contrastive.py` | 对比学习模块 (VideoTextAligner, MultiPositiveInfoNCE) |
| `multi_label_classifier.py` | 多标签分类器 (VideoMultiLabelClassifier) |
| **数据处理** ||
| `dataloaders/dataset_msrvtt_feats.py` | 基于预提取特征的数据集 |
| `dataloaders/dataset_msrvtt_precomputed_feats.py` | 预计算特征数据集 |
| `dataloaders/dataset_msrvtt_raw.py` | 原始视频数据集 |
| `dataloaders/rawvideo_util.py` | 视频帧提取工具 (RawVideoExtractorCV2) |
| `load_tokenizers.py` | 分词器加载与扩展 (CLIPTokenizer_Custom, Tokenizer_M) |
| **特征提取** ||
| `extract_clip_feats.py` | CLIP 视觉特征提取 (K-Split 采样) |
| `extract_msrvtt_CLIPPhraseEncoder_feats.py` | 短语级 CLIP 特征提取 |
| `extract_msrvtt_CLIPSentenceEncoder_feats.py` | 句子级 CLIP 特征提取 |
| **训练脚本** ||
| `train_base_mean.py` | 基础模型训练 |
| `train_base_mean_monitored.py` | 带 TensorBoard 监控的训练 |
| `train_base_mean_CL_monitored.py` | 对比学习训练 |
| `train_base_mean_len_control_monitored.py` | 长度控制模型训练 |
| `train_base_mean_RL_monitored.py` | 强化学习训练 |
| `train_msrvtt_srvc_monitored.py` | SRVC 模型训练 |
| **推理脚本** ||
| `infer_base_mean.py` | 基础模型推理 |
| `infer_base_mean_len_control.py` | 长度控制推理 |
| **评估脚本** ||
| `evaluate.py` | 完整评估 (BLEU, METEOR, ROUGE, CIDEr, SPICE) |
| `process_cider_only.py` | 仅计算 CIDEr |
| `process_spice_only.py` | 仅计算 SPICE |
| **测试脚本** ||
| `test_eval.py` | 评估功能测试 |
| `test_base_mean_monitored.py` | 基础模型测试 |
| `test_classifier_only.py` | 分类器测试 |
| `test_clip_encode_text.py` | CLIP 文本编码测试 |

---

## 附录 B: 模型参数量统计

### CaptionModel_Base
```
组件                    参数量
─────────────────────────────────
word_embeddings        25,296,896  (49408 × 512, 冻结)
pos_embeddings            102,400  (200 × 512)
norm_input                  1,024  (512 × 2)
decoder (3层)           9,447,936  
norm_output                 1,024
lm_head                25,296,896  (512 × 49408)
─────────────────────────────────
总计                   ~60M (可训练 ~35M)
```

### CaptionModel_SRVC
```
组件                    参数量
─────────────────────────────────
clip_embedding         25,299,456  (49413 × 512)
video_multilabel_detector  ~25M
cross_attention           ~6M
phrase_decoder            ~9M
initial_decoder          ~18M
refine_decoder           ~18M
clip_phrase_encoder      (冻结)
clip_sentence_encoder    (冻结)
─────────────────────────────────
总计                   ~100M+
```

---

## 附录 C: 常见问题与解决方案

### Q1: CUDA Out of Memory
```bash
# 解决方案
1. 减小 batch_size
2. 增加 accum_steps (梯度累积)
3. 启用混合精度 --amp 1
4. 使用更少的 decoder 层
```

### Q2: METEOR 评估报错
```bash
# 需要安装 Java
sudo apt-get install openjdk-11-jre-headless

# 验证
java -version
```

### Q3: 分词器 pad_token_id 不一致
```python
# 原始 CLIP: pad_token_id = 49409
# 本项目: pad_token_id = 0 (与 ignore_index 对齐)

# 使用 CLIPTokenizer_Custom 自动处理
from load_tokenizers import CLIPTokenizer_Custom
tokenizer = CLIPTokenizer_Custom()  # 自动交换
```

### Q4: DDP 训练卡住
```bash
# 检查 NCCL 环境
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # 禁用 InfiniBand

# 使用 gloo 后端 (CPU)
dist.init_process_group(backend="gloo")
```

### Q5: 特征提取速度慢
```bash
# 使用多 GPU 并行
python extract_clip_feats.py --work_devices "0,0,1,1,2,2,3,3"

# 增加 batch size
python extract_clip_feats.py --process_frames_size 32
```

---

## 附录 D: 参考文献

1. **CLIP**: Learning Transferable Visual Models From Natural Language Supervision (Radford et al., 2021)
2. **MSR-VTT**: A Large Video Description Dataset for Bridging Video and Language (Xu et al., 2016)
3. **Transformer**: Attention Is All You Need (Vaswani et al., 2017)
4. **CIDEr**: CIDEr: Consensus-based Image Description Evaluation (Vedantam et al., 2015)
5. **SPICE**: SPICE: Semantic Propositional Image Caption Evaluation (Anderson et al., 2016)

---

*文档生成时间: 2024年12月*
*项目版本: VideoCaption_Reconstruction v1.0*
