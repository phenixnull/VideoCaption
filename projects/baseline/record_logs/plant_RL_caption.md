下面把我前面给你的“两阶段 draft→refine（vision→caption→better_caption）+ CIDEr 信号（监督 + 强化学习）”方案，用**完整符号、公式、shape、mask 语义、数据构造与优化策略**系统写一遍，并且把你提到的 **扩展词表 vocab=49412** 一并纳入（CLIP 原始 49408 + 4 个新 token）。

---

# 1. 任务与符号定义（含 shape）

## 1.1 数据与输入

* Batch size：(B)
* 视频 token 数（你目前预提取）：(T=12)
* 隐空间维度：(D=512)
* 词表大小：(V=49412)
* 最大文本长度（你固定）：(L_{\max}=77)

**视频特征（已离线提取）**

* (\mathbf{V}\in\mathbb{R}^{B\times T\times D})
  其中 (\mathbf{V}_{b,t}\in\mathbb{R}^{D}) 表示第 (b) 个视频的第 (t) 个片段/帧 CLS token。
* 视频 mask：(\mathbf{M}_v\in{0,1}^{B\times T})，1 表示有效 token，0 表示 pad。

**文本**

* 目标 token：(\mathbf{y}\in\mathbb{N}^{B\times L_{\max}})，包含 BOS/EOS/PAD。
* 文本 mask：(\mathbf{M}*y\in{0,1}^{B\times L*{\max}})，1 有效，0 pad。

> 注意：PyTorch `*_key_padding_mask` 语义是 **True=忽略**；因此要用
> [
> \mathbf{K}_v = (\mathbf{M}_v = 0)\in{0,1}^{B\times T},\quad
> \mathbf{K}_y = (\mathbf{M}_y = 0)\in{0,1}^{B\times L}
> ]

---

# 2. 模型结构（与你当前实现对齐，但为 refine 做必要扩展）

## 2.1 词嵌入与位置嵌入

* token embedding：(\mathbf{E}\in\mathbb{R}^{V\times D})
* 对输入 token 序列 (\mathbf{x}\in\mathbb{N}^{B\times L})：
  [
  \mathrm{Emb}(\mathbf{x}) = \mathbf{E}[\mathbf{x}] \in \mathbb{R}^{B\times L\times D}
  ]
* learned positional embedding：(\mathbf{P}\in\mathbb{R}^{L_{\max}\times D})
  [
  \mathbf{H}*0 = \mathrm{LN}\big(\mathbf{E}[\mathbf{x}] + \mathbf{P}*{0:L-1}\big)\in\mathbb{R}^{B\times L\times D}
  ]

## 2.2 视频 memory（强烈建议：不要 mean pool 成 1 token）

你现在 mean pooling 得到 ([B,1,D]) 会成为主要瓶颈。refine 方案也更需要保留时序 token。

**建议 memory：**
[
\mathbf{H}_v = \mathbf{V} + \mathbf{P}^{(temp)}\in\mathbb{R}^{B\times T\times D}
]
其中 (\mathbf{P}^{(temp)}\in\mathbb{R}^{T\times D}) 为可学习 temporal pos emb（可选，但推荐）。

对应 padding mask：
[
\mathbf{K}_v = (\mathbf{M}_v=0)\in{0,1}^{B\times T}
]

## 2.3 Transformer Decoder（与你代码一致）

* decoder 输入：(\mathbf{H}_0\in\mathbb{R}^{B\times L\times D})
* memory：(\mathbf{M}\in\mathbb{R}^{B\times S\times D})
* causal mask（禁止看未来）：
  [
  \mathbf{C}\in{0,1}^{L\times L},\quad \mathbf{C}_{i,j} = \mathbb{1}[j>i]
  ]
* 输出 hidden：
  [
  \mathbf{H} = \mathrm{Decoder}(\mathbf{H}_0,\ \mathbf{M};\ \mathbf{C},\mathbf{K}_y,\mathbf{K}_m)\in\mathbb{R}^{B\times L\times D}
  ]
* 输出 logits：
  [
  \mathbf{Z}=\mathbf{H}\mathbf{W}^\top+\mathbf{b},\quad \mathbf{W}\in\mathbb{R}^{V\times D},\ \mathbf{b}\in\mathbb{R}^{V}
  ]
  [
  \mathbf{Z}\in\mathbb{R}^{B\times L\times V}
  ]

---

# 3. 两阶段生成：Draft 与 Refine 的条件形式（核心方案）

## 3.1 Stage-1 Draft：vision → caption

**概率分解**
[
p_{\theta_1}(\mathbf{y}\mid \mathbf{V})=\prod_{t=1}^{L^*}p_{\theta_1}(y_t\mid y_{<t},\mathbf{V})
]

**训练 teacher forcing：右移**

* 输入：(\mathbf{y}^{in} = \mathbf{y}_{[:,0:L-2]}\in\mathbb{N}^{B\times (L-1)})
* 目标：(\mathbf{y}^{tgt} = \mathbf{y}_{[:,1:L-1]}\in\mathbb{N}^{B\times (L-1)})

**memory 取视频 tokens：**
[
\mathbf{M}^{(1)}=\mathbf{H}_v\in\mathbb{R}^{B\times T\times D},\quad \mathbf{K}_m^{(1)}=\mathbf{K}_v
]

输出：

* logits：(\mathbf{Z}^{(1)}\in\mathbb{R}^{B\times (L-1)\times V})
* hidden：(\mathbf{H}^{(1)}\in\mathbb{R}^{B\times (L-1)\times D})（你 `return_hidden=True`）

---

## 3.2 Stage-2 Refine：vision + draft_hidden → better_caption

你希望“既能 vision 直接生成（stage1），也能基于 vision + 上一句 decoder hidden 生成新句子（stage2）”。最稳的条件方式是：

1. Draft 先生成一条句子 (\hat{\mathbf{y}}^{(1)})（推理输出）。
2. 把 (\hat{\mathbf{y}}^{(1)}) 再走一遍 teacher forcing（或至少前缀）得到其 token-level hidden：(\hat{\mathbf{H}}^{(1)}\in\mathbb{R}^{B\times S\times D})，其中 (S=\mathrm{len}(\hat{\mathbf{y}}^{(1)})-1)（不含最后预测位）。

然后 refine 的 memory 定义为拼接：
[
\mathbf{M}^{(2)} = \mathrm{Concat}\big(\mathbf{H}_v,\ \mathrm{Proj}(\hat{\mathbf{H}}^{(1)})\big)\in\mathbb{R}^{B\times (T+S)\times D}
]
其中 (\mathrm{Proj}(\cdot)) 可选（线性层），用于让模型区分“视频 token”与“文本 hidden token”（也可加 type embedding）。

对应 mask：
[
\mathbf{K}*m^{(2)} = \mathrm{Concat}(\mathbf{K}*v,\ \mathbf{K}*{\hat{y}})\in{0,1}^{B\times (T+S)}
]
其中 (\mathbf{K}*{\hat{y}}\in{0,1}^{B\times S}) 来自 (\hat{\mathbf{y}}^{(1)}) 的 padding（若你固定长度则按 PAD 位置置 True）。

Refine 的生成分解：
[
p_{\theta_2}(\mathbf{y}^{(2)}\mid \mathbf{V},\hat{\mathbf{y}}^{(1)})=\prod_{t=1}^{L^*}p_{\theta_2}(y^{(2)}*t\mid y^{(2)}*{<t},\mathbf{V},\hat{\mathbf{H}}^{(1)})
]

---

# 4. “必须学会 refine 而不是重写”的目标函数设计

你的关键诉求是：**refine 必须倾向在 y1 基础上改进**，如果改差了要惩罚。

我给你两层约束：一层是“结构性（模型会用 y1）”，一层是“奖励 shaping（改差就罚）”。

---

## 4.1 结构性：Copy/Edit 偏置（推荐）

Refine 阶段对 draft hidden token 有 cross-attention 权重 (\alpha_{t,s})（第 (t) 个生成步对 draft token (s) 的注意力）。

构造 copy 分布（把注意力聚合到词表上）：
[
p_{\text{copy}}(w)=\sum_{s:\hat{y}^{(1)}*s=w}\alpha*{t,s}
]

生成分布来自 lm head：
[
p_{\text{gen}}(w)=\mathrm{softmax}(\mathbf{z}_t)_w
]

引入 gate（标量）：
[
g_t = \sigma(\mathbf{u}^\top \mathbf{h}_t)\in(0,1)
]

最终输出：
[
p(w)=g_t p_{\text{gen}}(w)+(1-g_t)p_{\text{copy}}(w)
]

直观效果：模型会“能抄就抄，必要时改”，天然更像 refinement。

---

## 4.2 奖励 shaping：显式要求“比上一句更好，否则惩罚”

定义 CIDEr 分数（相对于该视频参考集合 (\mathcal{R})）：
[
R(\mathbf{y}) = \mathrm{CIDEr}(\mathbf{y},\mathcal{R})
]

定义提升量：
[
\Delta R = R(\mathbf{y}^{(2)}) - R(\mathbf{y}^{(1)})
]

定义编辑代价（越改越大）：(C(\mathbf{y}^{(2)},\mathbf{y}^{(1)})\ge 0)。工程上你可以用：

* token-level Levenshtein distance（编辑距离）
* 或 1-gram/2-gram 差异比例

构造 refine 的 shaped reward：
[
\tilde{R}(\mathbf{y}^{(2)};\mathbf{y}^{(1)})=\Delta R - \lambda C(\mathbf{y}^{(2)},\mathbf{y}^{(1)})
]

若你想更“硬”的惩罚（改差就强罚），可以用 hinge：
[
\tilde{R} = \max(\Delta R - \lambda C,\ -m)
]
其中 (m>0) 是最小惩罚幅度。

---

# 5. 训练数据构造：如何得到 (y1 → y2) 与每句 CIDEr

你说“每个 caption 都有对应 CIDEr 分数”——严格落地需要指定它相对于哪个参考集合。

对每个视频 (i) 有参考集合 (\mathcal{R}_i)。则任意候选句 (\mathbf{y}) 都可以计算：
[
s(\mathbf{y}) = \mathrm{CIDEr}(\mathbf{y},\mathcal{R}_i)
]

## 5.1 最稳的 refine pair 构造（推荐）

对每个训练视频：

1. 用 Draft 模型生成 (N) 个候选（beam 或 sampling）：
   [
   {\hat{\mathbf{y}}^{(1,n)}}_{n=1..N}
   ]
2. 计算每个候选分数 (s_n=s(\hat{\mathbf{y}}^{(1,n)}))
3. 选一个作为 y1（例如从中位数附近抽样，避免总是极差导致训练不稳定）：
   [
   \mathbf{y}^{(1)}\leftarrow \hat{\mathbf{y}}^{(1,n)}
   ]
4. 目标 y2 有两种选择：

* **参考句的共识代表** (r^\star)：在 (\mathcal{R}_i) 里选“最共识”的一句
* 或 **候选集合的高分句**（让模型学“从一般到更好”）：
  [
  \mathbf{y}^{(2)}\leftarrow \arg\max_n s_n
  ]

最终得到训练对 ((\mathbf{V}_i,\ \mathbf{y}^{(1)})\to \mathbf{y}^{(2)})，并且有 (s(\mathbf{y}^{(1)}), s(\mathbf{y}^{(2)}))。

---

# 6. 优化目标：监督（XE）+ 强化学习（SCST）完整公式

## 6.1 Stage-1 Draft 的监督学习（XE）

[
\mathcal{L}^{(1)}*{\text{XE}}
= -\sum*{b=1}^B\sum_{t=1}^{L_b-1}\log p_{\theta_1}(y_{b,t}^{tgt}\mid y_{b,<t}^{in},\mathbf{V}_b)
]
其中对 PAD 位置用 ignore（或用 mask 乘掉）。

---

## 6.2 Stage-2 Refine 的监督学习（XE + 可选 copy）

[
\mathcal{L}^{(2)}*{\text{XE}}
= -\sum*{b=1}^B\sum_{t=1}^{L_b-1}\log p_{\theta_2}(y^{(2),tgt}*{b,t}\mid y^{(2),in}*{b,<t},\mathbf{V}_b,\hat{\mathbf{H}}^{(1)}_b)
]

若引入 copy mixture，则 (p_{\theta_2}) 用上面的混合分布。

---

## 6.3 强化学习（SCST）用于 CIDEr（Draft 或 Refine 都可用）

### 6.3.1 Draft 的 SCST（可选但常用）

对每个样本：

* 从当前策略采样一句 (\mathbf{y}^s\sim p_{\theta_1}(\cdot\mid\mathbf{V}))
* greedy baseline：(\mathbf{y}^g=\mathrm{Greedy}(p_{\theta_1}(\cdot\mid\mathbf{V})))

优势：
[
A = R(\mathbf{y}^s) - R(\mathbf{y}^g)
]

损失：
[
\mathcal{L}^{(1)}*{\text{RL}}
= -\sum*{b=1}^B A_b \sum_{t}\log p_{\theta_1}(y^s_{b,t}\mid y^s_{b,<t},\mathbf{V}_b)
]

常见做法：与 XE 混合稳定训练：
[
\mathcal{L}^{(1)} = \mathcal{L}^{(1)}*{\text{RL}} + \eta \mathcal{L}^{(1)}*{\text{XE}}
]
(\eta) 通常取 0.05~0.2。

---

### 6.3.2 Refine 的 SCST（核心，用 shaped reward）

Refine 情况下，reward 用 (\tilde{R}=\Delta R - \lambda C)。

对每个样本：

* 输入条件（固定）：((\mathbf{V}, \mathbf{y}^{(1)}))
* 采样 refine 输出：(\mathbf{y}^{(2),s}\sim p_{\theta_2}(\cdot\mid \mathbf{V},\mathbf{y}^{(1)}))
* greedy baseline：(\mathbf{y}^{(2),g}=\mathrm{Greedy}(p_{\theta_2}(\cdot\mid \mathbf{V},\mathbf{y}^{(1)})))

优势：
[
\tilde{A} = \tilde{R}(\mathbf{y}^{(2),s};\mathbf{y}^{(1)}) - \tilde{R}(\mathbf{y}^{(2),g};\mathbf{y}^{(1)})
]

损失：
[
\mathcal{L}^{(2)}*{\text{RL}}
= -\sum*{b=1}^B \tilde{A}*b \sum_t \log p*{\theta_2}(y^{(2),s}*{b,t}\mid y^{(2),s}*{b,<t},\mathbf{V}_b,\mathbf{y}^{(1)}_b)
]

同样建议混合 XE：
[
\mathcal{L}^{(2)} = \mathcal{L}^{(2)}*{\text{RL}} + \eta \mathcal{L}^{(2)}*{\text{XE}}
]

---

# 7. 推理（测试）时的流程（不依赖 CIDEr）

给一个最可复现的 inference pipeline：

1. Draft：(\hat{\mathbf{y}}^{(1)}=\mathrm{Greedy/Beam}(p_{\theta_1}(\cdot\mid \mathbf{V})))
2. 获取 draft hidden：(\hat{\mathbf{H}}^{(1)}=\mathrm{DecoderHidden}(\mathbf{V},\hat{\mathbf{y}}^{(1)}))
3. Refine：(\hat{\mathbf{y}}^{(2)}=\mathrm{Greedy/Beam}(p_{\theta_2}(\cdot\mid \mathbf{V},\hat{\mathbf{H}}^{(1)})))
4. 输出：(\hat{\mathbf{y}}^{(2)})

（可选更稳）加一个质量预测头 (\widehat{s}(\cdot)) 做 gate：若 (\widehat{s}(\hat{\mathbf{y}}^{(2)}) < \widehat{s}(\hat{\mathbf{y}}^{(1)}))，就输出 y1。

---

# 8. 关键 shape 总表（避免实现时搞混）

| 符号                       | 含义                      | shape          |
| ------------------------ | ----------------------- | -------------- |
| (\mathbf{V})             | 视频 token 特征             | ([B,T,D])      |
| (\mathbf{M}_v)           | 视频有效 mask（1有效）          | ([B,T])        |
| (\mathbf{K}_v)           | 视频 padding mask（True忽略） | ([B,T])        |
| (\mathbf{y})             | token 序列                | ([B,L_{\max}]) |
| (\mathbf{M}_y)           | 文本有效 mask（1有效）          | ([B,L_{\max}]) |
| (\mathbf{y}^{in})        | 右移输入                    | ([B,L-1])      |
| (\mathbf{y}^{tgt})       | 右移目标                    | ([B,L-1])      |
| (\mathbf{H}_v)           | 视频 memory               | ([B,T,D])      |
| (\hat{\mathbf{H}}^{(1)}) | draft token hidden      | ([B,S,D])      |
| (\mathbf{M}^{(2)})       | refine memory concat    | ([B,T!+!S,D])  |
| (\mathbf{Z})             | logits                  | ([B,L-1,V])    |

---

# 9. 词表扩展到 49412：初始化与冻结的“正确公式化处理”

CLIP 原始词表 (V_0=49408)，你当前 (V=49412=V_0+4)。

## 9.1 embedding / lm_head 初始化（partial copy）

令 CLIP 的 token embedding：
[
\mathbf{E}_0\in\mathbb{R}^{V_0\times D}
]

你的模型 embedding：
[
\mathbf{E}\in\mathbb{R}^{V\times D}
]

**拷贝前 (V_0) 行：**
[
\mathbf{E}_{0:V_0-1}\leftarrow \mathbf{E}_0
]

**新增 4 行初始化：**（匹配 CLIP embedding 的均值/方差以保持尺度）
[
\mu=\frac{1}{V_0}\sum_{i=1}^{V_0}\mathbf{E}*0[i],\quad
\sigma=\sqrt{\frac{1}{V_0}\sum*{i=1}^{V_0}(\mathbf{E}_0[i]-\mu)^2}
]
[
\mathbf{E}[V_0:V-1]\sim \mathcal{N}(\mu,\ \mathrm{diag}(\sigma^2))
]

lm_head 权重同理（或直接 weight tying：(\mathbf{W}=\mathbf{E})）。

## 9.2 冻结策略（必须让新增 token 可训练）

你如果“整块冻结 embedding”，新增 token 永远学不会。正确做法是：

* 冻结 (\mathbf{E}[0:V_0-1])（CLIP 原始部分）
* 放开 (\mathbf{E}[V_0:V-1])（新增 4 行）

工程上最直接的方法是对梯度做行级 mask：
[
\nabla \mathbf{E}[0:V_0-1] \leftarrow 0
]

同理 lm_head（如果不 weight tying）。

---

# 10. 推荐的训练日程（MSVD 规模下的可执行版本）

给你一个最实用的顺序（每一步都能产出可对比的 ablation）：

1. **Draft-XE**：先把 memory 改成 ([B,T,D])，训练到收敛。
2. **Draft-SCST（可选）**：用 CIDEr 做 SCST 微调，获得更强 draft。
3. **构造 refine pairs**：用当前 draft 对 train 集生成候选并缓存 CIDEr。
4. **Refine-XE**：训练 ((\mathbf{V},\mathbf{y}^{(1)})\to \mathbf{y}^{(2)})。
5. **Refine-SCST（核心）**：用 shaped reward (\tilde{R}=\Delta R-\lambda C) 做 SCST，确保“改差就罚、少改更稳”。

---

# 11. 你这个 baseline 需要的最小接口改动（概念级）

为了让“vision-only”和“refine”都能跑，建议把 forward 抽象成：

* `encode_video(V, Mv) -> Hv, Kv`
* `decode(memory, Kmem, captions, My) -> logits, hidden`

然后：

* Draft：`memory = Hv`
* Refine：`memory = concat(Hv, proj(H1))`

这能保证你的训练和推理路径完全一致，减少 bug。

具体的clip embedding要参考我当前已有的这个项目怎么做的，然后结合上述有效的方案再去做实验验证
