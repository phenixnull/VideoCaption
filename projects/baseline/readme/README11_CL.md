太好了，那我给你一份**“单向 v→t 对比 + Caption CE”**的完整方案（只给公式与工程细节，不写代码）。读完你就能直接把它嵌进现有 `CaptionModel_Base` 的训练循环里。

# 1) 批次数据与基线前向

* 来自 `MSRVTT_FeaturesDataset` 的一个 batch 形如
  ((\mathbf{F},\mathbf{m},\texttt{ids},\texttt{mask},\text{cap_str},\text{vid},\text{sen_id}))，
  其中 (\mathbf{F}\in\mathbb{R}^{B\times T\times512}) 是离线 CLIP 视频帧特征，(\mathbf{m}\in{0,1}^{B\times T}) 是有效帧 mask；你的实现中每视频通常取 (T=12) 帧（按 ks12 特征文件），但方案对任意 (T) 都成立。
* `CaptionModel_Base` 前向：对视频做 **mask 感知均值池化** 得到单 token memory，再用解码器做 teacher-forcing，输出 token 级 logits。
  交叉熵（单向条件建模）：
  [
  \mathcal{L}*{CE}
  = -\frac{1}{B}\sum*{i=1}^B \sum_{t=1}^{L-1}\log p_\theta!\left(y^{(i)}*{t},\middle|,y^{(i)}*{<t},,\mathbf{F}^{(i)},\mathbf{m}^{(i)}\right),
  ]
  其中 (L) 为文本序列总长（如固定 77，训练时对后 76 个位置计 loss）。

# 2) 对比分支（单向 v→t）

**(a) 视频嵌入**（不额外视频编码，直接用离线帧特征 + 均值池化）
对第 (i) 条样本：
[
\tilde{\mathbf{v}}*i ;=; \frac{1}{\sum*{t} m_{it}}\sum_{t=1}^{T} m_{it},\mathbf{F}_{it}\in\mathbb{R}^{512},\qquad
\mathbf{v}_i ;=; \frac{\tilde{\mathbf{v}}_i}{|\tilde{\mathbf{v}}_i|_2}.
]
这就是你模型里用到的 **SequenceMeanPooling（mask 感知）** 的向量版做法。

**(b) 句子嵌入**（CLIP 默认文本编码）
使用你项目里的 `load_clip_model(...).encode_text` 与 CLIP 官方 tokenizer 对 `cap_str` 编码，并做 L2 归一化：
[
\mathbf{s}_i ;=; \frac{\texttt{CLIP.encode_text}\big(\texttt{clip_tokenize}(\text{cap_str}_i)\big)}{\left|\texttt{CLIP.encode_text}\big(\texttt{clip_tokenize}(\text{cap_str}_i)\big)\right|_2}\in\mathbb{R}^{512}.
]
工程注意：**不要**复用 CE 分支的 token ids；`encode_text` 期望的是 CLIP 的 77 上限格式（你推理/训练脚本里已用 CLIP 自定义 tokenizer）。

**(c) 相似度矩阵与温度**
令 batch 大小 (B)，用余弦（归一化后点积）与温度 (\tau)：
[
Z_{ij} ;=; \frac{\mathbf{v}_i^\top \mathbf{s}_j}{\tau},\qquad \tau=\exp(-\alpha).
]
(\alpha) 初值采用 CLIP 的 `logit_scale`（可选地把 (\alpha) 设为**可学习**，其余编码器全冻结）。

**(d) 单向 v→t 的 InfoNCE**
只保留“给定视频找到自己的句子”的方向：
[
\mathcal{L}*{v\to t} ;=; -\frac{1}{B}\sum*{i=1}^B
\log \frac{\exp(Z_{ii})}{\sum_{j=1}^B \exp(Z_{ij})}.
]

> DDP：建议 `all_gather` 扩展 ({\mathbf{v}},{\mathbf{s}}) 到全局 batch (B'!=!B\times\text{world})，上式分母把 (B) 换 (B')，正样本仍用**同 rank**对应的配对索引。

# 3) 总损失与调度

* **联合训练目标**（只单向对比）：
  [
  \mathcal{L}*{total} ;=; \mathcal{L}*{CE};+;\lambda,\mathcal{L}_{v\to t}.
  ]
* (\lambda) 建议从 0 **线性 warmup** 到目标值（如 0.1～0.5），与现有训练脚本的混合、AMP、梯度累积保持一致（你的训练/推理脚本里前向签名与张量形状已固定为 `[B,76,V]` 的输出）。

# 4) 工程落地要点（与你现有风格对齐）

1. **数据流**

   * Loader 输出已包含 (\mathbf{F},\mathbf{m},\text{cap_str}) 等；`MSRVTT_FeaturesDataset` 负责把离线 **ViT-B/32 ks12** 特征与标注对齐。
2. **CE 分支保持不变**

   * 前向：视频均值池化→单 token memory→解码器→logits→CE（PAD 忽略，未来 mask）。
3. **对比分支实现细节**

   * **视频侧**：直接对 (\mathbf{F}) 做 mask-aware mean（与基线同一实现保证对齐）；L2 归一化。
   * **文本侧**：用 CLIP 的 tokenizer & `encode_text`；得到 (\mathbf{s}_i) 后 L2。
   * **梯度**：冻结 CLIP 文本编码器与离线视频特征；**只**对 caption 模型参数（以及可选的温度 (\alpha)）反传。
   * **稳定性**：相似度矩阵计算前可 `detach()` 文本侧以确保不反传到 CLIP；温度 (\tau) 用 `clamp` 限界。
   * **分布式**：可选 `all_gather` 拼负样本；注意去重本 rank 的正样索引。
4. **日志与监控**

   * 记录 `acc@1`（批内检索准确率） = (\frac{1}{B}\sum_i \mathbb{1}[\arg\max_j Z_{ij}=i])，同时跟踪 CE perplexity；二者此消彼长时适度调 (\lambda)。
5. **推理阶段**

   * 生成保持你现有的 nucleus/beam/greedy 路线；对比分支在推理**不参与**（只在训练期起到正则/Aux 对齐的作用）。

# 5) 关键超参与建议

* (\lambda)：0.1 起步，线性 3–5 epoch 升到 0.3（视 CE 曲线与验证集 CIDEr 调整）。
* 温度 (\tau)：用 CLIP 默认 `logit_scale` 初始化；是否学习看稳定性。
* Batch size：对比分支效果依赖负样本数；DDP 下全局 batch 越大越稳。
* 归一化：务必对 (\mathbf{v},\mathbf{s}) **L2** 归一化后再点积，等价于余弦相似度。

