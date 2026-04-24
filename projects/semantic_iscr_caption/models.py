# models.py
# -*- coding: utf-8 -*-
import math
import sys
import os
import torch
import torch.nn as nn
from clip import clip
from pathlib import Path
from typing import Optional, List, Tuple
import torch.nn.functional as F
from load_tokenizers import CLIPTokenizer_Custom

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CLIP_MODEL_DIR = PROJECT_ROOT / "models" / "clip_models"

# =========================================================
# CLIP 加载（保持原函数名与签名）
# =========================================================
def load_clip_model(
    model_name: str = "ViT-B/32",
    device: str = "cuda",
    local_files_only: bool = True,
    cache_dir: str = str(DEFAULT_CLIP_MODEL_DIR),
):
    """
    加载CLIP模型并指定缓存目录

    Args:
        model_name: 模型名称，默认为"ViT-B/32"
        device: 设备，默认为"cuda"
        local_files_only: 是否仅使用本地文件，默认为True
        cache_dir: 缓存目录的路径

    Returns:
        clip_model, preprocessor: 加载好的模型和预处理器
    """
    CLIP_PRETRAINED_MODELS_NAMES = [
        'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64',
        'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'
    ]

    if sys.platform.startswith("win"):  # 或者用 os.name == "nt"
        cache_dir = r"D:\Users\Administrator\Desktop\DeepLearningProjects_SELF\VideoCaption_Reconstruction\project\models\clip_models"

    # 确保缓存目录存在
    os.makedirs(cache_dir, exist_ok=True)

    if local_files_only is True:
        print("加载本地CLIP模型 ...")
        try:
            clip_model, preprocessor = clip.load(
                model_name,
                device=device,
                download_root=cache_dir  # 指定缓存目录
            )
        except Exception as e:
            print(f"本地加载失败: {e}")
            print("尝试从网络下载...")
            clip_model, preprocessor = clip.load(
                model_name,
                device=device,
                download_root=cache_dir
            )
    else:
        print("从网络加载CLIP模型...")
        clip_model, preprocessor = clip.load(
            model_name,
            device=device,
            download_root=cache_dir  # 指定缓存目录
        )

    print(f"CLIP模型缓存位置: {cache_dir}")
    return clip_model, preprocessor


# =========================================================
# 位置编码：正弦型（保留原类，修注释）
# =========================================================
class SinusoidalPositionalEncoding(nn.Module):
    """
    固定的正弦-余弦位置编码（不可学习）。
    公式（d_model = D，位置 p，通道索引 i 从 0 开始）：
        PE[p, 2i]   = sin(p / (10000^(2i/D)))
        PE[p, 2i+1] = cos(p / (10000^(2i/D)))

    用法：
        pe = SinusoidalPositionalEncoding(d_model=512, max_len=200, dropout=0.1)
        x = pe(x)  # x: [B, T, D]，返回 x + PE

    参数：
        d_model: 特征维度 D
        max_len: 预生成的最大序列长度上限
        dropout: 加性后可选的 dropout
        scale_x: 是否对输入 x 进行 sqrt(D) 缩放
    """
    def __init__(
        self,
        d_model: int = 512,
        max_len: int = 200,
        dropout: float = 0.0,
        scale_x: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.scale_x = scale_x

        pe = torch.zeros(max_len, d_model)  # [max_len, D]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维
        pe = pe.unsqueeze(0)  # [1, max_len, D]
        self.register_buffer("pe", pe, persistent=False)

    def forward(
        self,
        x: torch.Tensor,               # [B, T, D]
        position_ids: Optional[torch.Tensor] = None,  # [B, T]
    ) -> torch.Tensor:
        B, T, D = x.shape
        assert D == self.d_model, f"d_model mismatch: expected {self.d_model}, got {D}"

        if self.scale_x:
            x = x * math.sqrt(self.d_model)

        if position_ids is None:
            position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)  # [B, T]

        # 从 buffer 取对应位置
        pe_t = self.pe.to(dtype=x.dtype)  # [1, max_len, D]
        pos_emb = pe_t[0].index_select(0, position_ids.reshape(-1)).view(B, T, D)
        return self.dropout(x + pos_emb)


# =========================================================
# 位置编码：可学习型（保留原类，修注释）
# =========================================================
class LearnedPositionalEmbedding(nn.Module):
    """
    可学习的绝对位置嵌入（nn.Embedding 版）。
    常见于 BERT/GPT/ViT 等模型的"绝对位置编码"。

    用法：
        pos = LearnedPositionalEmbedding(num_positions=200, embedding_dim=512, dropout=0.0, padding_idx=0)
        x = pos(x, position_ids=pos_ids, attention_mask=attn_mask)  # 返回 x + pos

    参数：
        num_positions: 支持的最大序列长度上限
        embedding_dim: 特征维度 D（需与输入匹配）
        dropout: 加性后可选的 dropout
        padding_idx: 若提供，并传入 attention_mask，可让被 mask 的位置使用 padding_idx（一般为 0）
    """
    def __init__(
        self,
        num_positions: int = 200,
        embedding_dim: int = 512,
        dropout: float = 0.0,
        padding_idx: Optional[int] = 0,
    ):
        super().__init__()
        self.num_positions = num_positions
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        self.weight = nn.Embedding(num_positions, embedding_dim, padding_idx=padding_idx)
        nn.init.normal_(self.weight.weight, mean=0.0, std=0.02)
        if padding_idx is not None:
            with torch.no_grad():
                self.weight.weight[padding_idx].zero_()

        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    @torch.no_grad()
    def _positions_from_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        由 attention_mask 生成逐位置的绝对位置索引（跳过 pad 的递增编号）。
        attention_mask: [B, T]，有效为 1，pad 为 0
        返回：position_ids [B, T]，pad 位置置为 padding_idx（若未设置则置 0）
        """
        pos = attention_mask.long().cumsum(dim=1) * attention_mask.long() - attention_mask.long()
        if self.padding_idx is not None:
            pos = torch.where(attention_mask.bool(), pos, torch.full_like(pos, self.padding_idx))
        return pos.clamp_(min=0, max=self.num_positions - 1)

    def forward(
        self,
        x: torch.Tensor,                                # [B, T, D]
        position_ids: Optional[torch.Tensor] = None,    # [B, T]
        attention_mask: Optional[torch.Tensor] = None,  # [B, T] (1=有效, 0=pad)
    ) -> torch.Tensor:
        B, T, D = x.shape
        assert D == self.embedding_dim, f"embedding_dim mismatch: expected {self.embedding_dim}, got {D}"
        assert T <= self.num_positions, f"sequence length {T} exceeds num_positions={self.num_positions}"

        if position_ids is None:
            if attention_mask is not None:
                position_ids = self._positions_from_mask(attention_mask)
            else:
                position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)

        pos_emb = self.weight(position_ids).to(dtype=x.dtype)  # [B, T, D]
        return self.dropout(x + pos_emb)


def test_positional_encodings(device: str = None) -> None:
    """
    单测：同时覆盖 SinusoidalPositionalEncoding 与 LearnedPositionalEmbedding
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 基本设置
    B, T, D = 2, 10, 32
    x = torch.randn(B, T, D, device=device, requires_grad=True)

    # 1) 正弦位置编码
    pe = SinusoidalPositionalEncoding(d_model=D, max_len=512, dropout=0.0, scale_x=True).to(device)
    y1 = pe(x)
    assert y1.shape == x.shape

    pos_ids = torch.arange(T, device=device).flip(0).unsqueeze(0).expand(B, T)
    y1_rev = pe(x.detach().clone().requires_grad_(True), position_ids=pos_ids)
    assert not torch.allclose(y1, y1_rev)

    x.grad = None
    (y1.sum()).backward(retain_graph=True)
    expected = math.sqrt(D)
    mean_abs_grad = x.grad.abs().mean().item()
    assert abs(mean_abs_grad - expected) < 1e-4

    # 2) 可学习位置嵌入
    attn = torch.ones(B, T, dtype=torch.long, device=device)
    attn[0, -3:] = 0
    lp = LearnedPositionalEmbedding(num_positions=512, embedding_dim=D, dropout=0.0, padding_idx=0).to(device)
    x2 = torch.randn(B, T, D, device=device, requires_grad=True)
    y2 = lp(x2, attention_mask=attn)
    assert y2.shape == x2.shape
    assert torch.allclose(y2[0, -3:], x2[0, -3:], atol=1e-6)

    lp.zero_grad(set_to_none=True)
    x2.grad = None
    (y2 * attn.unsqueeze(-1)).sum().backward()
    assert lp.weight.weight.grad is not None
    assert lp.weight.weight.grad[lp.padding_idx].abs().max().item() == 0.0
    assert lp.weight.weight.grad[1:].abs().sum().item() > 0

    pos_ids2 = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    x3 = torch.randn(B, T, D, device=device)
    y3 = lp(x3, position_ids=pos_ids2)
    assert y3.shape == x3.shape

    print("✅ 所有位置编码测试通过（Sinusoidal / Learned）。")


# =========================================================
# 类型嵌入（保留原类，修注释）
# =========================================================
class TypeEmbedding(nn.Module):
    """
    可学习的"类型嵌入"（token-type / modality embedding）
    典型用途：为不同模态或片段（如 video / text）提供可学习的偏置向量，并与输入特征逐位置相加。

    约定：
    - 输入 x 形状为 [B, T, D]
    - type_ids 可为 [B, T]，也支持更简便的 [B] / [T] / [1]
    - 若提供 attention_mask（[B, T]，1=有效、0=PAD）且设置了 padding_idx，则 PAD 位置自动使用 padding_idx 行

    参数:
        num_types:      类型总数（包含 padding_idx 占用的行）
        embedding_dim:  嵌入维度 D（需与 x 的最后一维一致）
        dropout:        加性后的 dropout 概率
        padding_idx:    若指定（常设为 0），该行向量在初始化后会被置零，且不会被更新
        init_std:       正态初始化的标准差
    """
    def __init__(
        self,
        num_types: int,
        embedding_dim: int,
        dropout: float = 0.0,
        padding_idx: Optional[int] = 0,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.num_types = num_types
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        self.weight = nn.Embedding(num_types, embedding_dim, padding_idx=padding_idx)
        nn.init.normal_(self.weight.weight, mean=0.0, std=init_std)
        if padding_idx is not None:
            with torch.no_grad():
                self.weight.weight[padding_idx].zero_()

        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def _expand_type_ids(self, type_ids: torch.Tensor, B: int, T: int) -> torch.Tensor:
        if type_ids.dim() == 2:
            assert type_ids.shape == (B, T), f"type_ids 应为 [B,T]，实际 {tuple(type_ids.shape)}"
            return type_ids
        if type_ids.dim() == 1:
            n = type_ids.size(0)
            if n == B:
                return type_ids.view(B, 1).expand(B, T)
            if n == T:
                return type_ids.view(1, T).expand(B, T)
            if n == 1:
                return type_ids.view(1, 1).expand(B, T)
            raise ValueError(f"不支持的 type_ids 形状 [n]={n}，期望 B 或 T 或 1")
        raise ValueError(f"不支持的 type_ids 维度 {type_ids.dim()}，期望 1 或 2")

    def forward(
        self,
        x: torch.Tensor,                         # [B, T, D]
        type_ids: Optional[torch.Tensor] = None, # 可为 [B,T] / [B] / [T] / [1]
        attention_mask: Optional[torch.Tensor] = None,  # [B, T]，1=有效，0=PAD
        default_type_id: int = 1,                # 当未提供 type_ids 时的默认类型 ID
    ) -> torch.Tensor:
        B, T, D = x.shape
        if D != self.embedding_dim:
            raise AssertionError(f"embedding_dim 不匹配：期望 {self.embedding_dim}，实际 {D}")
        if type_ids is None:
            type_ids = torch.full((B, 1), default_type_id, dtype=torch.long, device=x.device)
        type_ids = self._expand_type_ids(type_ids.to(x.device), B, T)

        if attention_mask is not None and self.padding_idx is not None:
            type_ids = torch.where(
                attention_mask.to(dtype=torch.bool, device=x.device),
                type_ids,
                torch.full_like(type_ids, self.padding_idx),
            )

        emb = self.weight(type_ids).to(dtype=x.dtype)  # [B, T, D]

        if attention_mask is not None and self.padding_idx is None:
            emb = emb * attention_mask.to(dtype=torch.float32, device=x.device).unsqueeze(-1).to(dtype=x.dtype)

        return self.dropout(x + emb)


# =========================================================
# 序列均值池化（保留原类，修注释）
# =========================================================
class SequenceMeanPooling(nn.Module):
    """
    序列均值池化（mask 感知版）
    - 输入 x: [B, T, D] 或 [B, T]
    - attention_mask: [B, T]（1=有效，0=PAD）
    - lengths: [B]（每条样本有效长度；与 attention_mask 二选一，优先使用 attention_mask）
    """
    def __init__(self, dim: int = 1, keepdim: bool = False, eps: float = 1e-12):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,                               # [B, T, D] 或 [B, T]
        attention_mask: Optional[torch.Tensor] = None, # [B, T] (0/1)
        lengths: Optional[torch.Tensor] = None,        # [B]
    ) -> torch.Tensor:
        assert x.dim() in (2, 3), f"x 需为 [B,T] 或 [B,T,D]，但得到 {tuple(x.shape)}"
        B = x.size(0)

        if attention_mask is None and lengths is not None:
            T = x.size(1)
            device = x.device
            ar = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # [B,T]
            attention_mask = (ar < lengths.view(B, 1)).to(torch.long)

        if attention_mask is None:
            return x.mean(dim=self.dim, keepdim=self.keepdim)

        assert attention_mask.shape == (x.size(0), x.size(1)), f"attention_mask 需为 [B,T]，实际 {tuple(attention_mask.shape)}"
        mask = attention_mask.to(dtype=x.dtype, device=x.device)

        if x.dim() == 3 and self.dim == 1:
            mask = mask.unsqueeze(-1)  # [B,T,1]

        sum_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        x_sum = (x.to(sum_dtype) * mask.to(sum_dtype)).sum(dim=self.dim, keepdim=self.keepdim)
        count = mask.sum(dim=self.dim, keepdim=self.keepdim).to(sum_dtype).clamp(min=self.eps)
        out = (x_sum / count).to(dtype=x.dtype)
        return out


def test_sequence_mean_pooling(device: str = None) -> None:
    """
    单测：针对 SequenceMeanPooling 的全面测试
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    B, T, D = 3, 7, 5
    x = torch.randn(B, T, D, device=device, requires_grad=True)

    attn = torch.ones(B, T, dtype=torch.long, device=device)
    attn[0, -2:] = 0
    attn[1, -4:] = 0

    pool = SequenceMeanPooling(dim=1, keepdim=False, eps=1e-12)

    out_plain = pool(x)
    exp_plain = x.mean(dim=1)
    assert torch.allclose(out_plain, exp_plain, atol=1e-6)

    out_masked = pool(x, attention_mask=attn)
    masked_sum = (x * attn.unsqueeze(-1)).sum(dim=1)
    counts = attn.sum(dim=1).clamp(min=1).unsqueeze(-1)
    exp_masked = masked_sum / counts
    assert torch.allclose(out_masked, exp_masked, atol=1e-6)

    pool.zero_grad(set_to_none=True)
    if x.grad is not None:
        x.grad.zero_()
    out_masked.sum().backward()
    grad_masked_positions = x.grad[attn == 0]
    if grad_masked_positions.numel() > 0:
        assert torch.allclose(grad_masked_positions, torch.zeros_like(grad_masked_positions), atol=1e-8)

    lengths = attn.sum(dim=1)
    out_len = pool(x.detach(), lengths=lengths)
    assert torch.allclose(out_len, exp_masked, atol=1e-6)

    pool_kd = SequenceMeanPooling(dim=1, keepdim=True, eps=1e-12)
    out_kd = pool_kd(x.detach(), attention_mask=attn)
    assert out_kd.shape == (B, 1, D)
    assert torch.allclose(out_kd.squeeze(1), exp_masked, atol=1e-6)

    x2d = torch.randn(B, T, device=device)
    out2d = pool(x2d, attention_mask=attn)
    exp2d = (x2d * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1)
    assert out2d.shape == (B,)
    assert torch.allclose(out2d, exp2d, atol=1e-6)

    if device == "cuda":
        x_fp16 = x.detach().half()
        out_fp16 = pool(x_fp16, attention_mask=attn)
        assert torch.allclose(out_fp16.float(), exp_masked.float(), atol=5e-3)

    print("✅ SequenceMeanPooling 单测通过。")


# =========================================================
# Caption 基线模型（保留原类名与接口，按你的需求最小侵入修复）
# =========================================================
class CaptionModel_Base(nn.Module):
    def __init__(self,
                 vocab_size: int = 49408,
                 decoder_nhead: int = 8,
                 d_model: int = 512,
                 deocder_layer_nums: int = 3,
                 pretrained_clip_name: str = "ViT-B/32",
                 init_we: bool = True,
                 init_lmhead: bool = True,
                 frozen_we = True,
                 frozen_lmhead: bool = False,
                 pad_token_id: int = 0,
                 bos_token_id: int = 49406,  # <BOS>/<CLS>
                 eos_token_id: int = 49407,  # <EOS>/<SEP>
                 ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # 1) 视频侧：mask 感知 mean pooling（输出 [B,1,D]）
        self.mean_pooling = SequenceMeanPooling(dim=1, keepdim=True, eps=1e-12)

        # 2) 文本侧：词嵌入 + 可学习位置 + 输入层 LN
        self.word_embeddings = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_embeddings = LearnedPositionalEmbedding(num_positions=200, embedding_dim=d_model, dropout=0.0, padding_idx=pad_token_id)
        self.norm_input = nn.LayerNorm(d_model)

        # 3) 解码器：**不注册原型层**；batch_first=True（避免转置，DDP 更稳）
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=decoder_nhead,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,  # 关键：直接使用 [B,L,D]
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=deocder_layer_nums)

        # 4) 输出：LN + 词表投影（保留原 bias=True）
        self.norm_output = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=True)

        # 5) 从 CLIP 初始化（仅用于权重拷贝）
        self.pretrained_clip_model, _ = load_clip_model(pretrained_clip_name)
        self.pretrained_clip_model.eval()
        for p in self.pretrained_clip_model.parameters():
            p.requires_grad = False
        self._init_from_clip_embeddings(init_we=init_we, init_lmhead=init_lmhead)
        # 可选：释放引用（不再使用），避免占用显存
        # self.pretrained_clip_model = None
        if frozen_we:
            for p in self.word_embeddings.parameters():
                p.requires_grad = False
        if frozen_lmhead:
            for p in self.lm_head.parameters():
                p.requires_grad = False
    def _init_from_clip_embeddings(self, init_we: bool = True, init_lmhead: bool = True):
        """
        使用 CLIP 的 token_embedding 权重初始化：
          - word_embeddings.weight
          - lm_head.weight
        要求 vocab_size 与 d_model 与 CLIP 一致；否则抛出错误，避免静默错配。
        """
        if not (init_we or init_lmhead):
            return

        clip_token_emb = self.pretrained_clip_model.token_embedding.weight  # [V, D]
        clip_vocab, clip_dim = clip_token_emb.shape
        cur_vocab, cur_dim = self.word_embeddings.num_embeddings, self.word_embeddings.embedding_dim

        if cur_vocab != clip_vocab:
            raise ValueError(
                f"[init_from_clip] vocab_size 不一致: current={cur_vocab}, clip={clip_vocab}."
            )
        if cur_dim != clip_dim:
            raise ValueError(
                f"[init_from_clip] d_model 不一致: current={cur_dim}, clip={clip_dim}."
            )

        with torch.no_grad():
            w = clip_token_emb.detach().to(self.word_embeddings.weight.device, dtype=self.word_embeddings.weight.dtype)
            if init_we:
                self.word_embeddings.weight.copy_(w)
            if init_lmhead:
                if self.lm_head.weight.shape != w.shape:
                    raise ValueError(
                        f"[init_from_clip] lm_head.weight 形状不匹配: {self.lm_head.weight.shape} vs {w.shape}"
                    )
                self.lm_head.weight.copy_(w)

    def forward(
        self,
        video_feats: torch.Tensor,            # [B, T, D]
        vid_mask: torch.Tensor,               # [B, T]  1=有效, 0=pad（来自特征文件）
        captions: torch.Tensor,               # [B, 77]
        caption_mask: Optional[torch.Tensor] = None,  # [B, 77] 1=有效, 0=pad（HF风格）
        return_hidden: bool = False,          # 是否返回解码器隐藏状态（用于对比学习）
    ) -> torch.Tensor:
        """
        训练(teacher forcing)：
          - input_ids = captions[:, :-1]  → 前 76
          - target_ids = captions[:,  1:] → 后 76（外部计算 CE，ignore_index=pad）
        返回：
          logits: [B, 76, vocab_size]
          如果 return_hidden=True，返回 (logits, hidden_states)
        """
        if video_feats.dim() != 3:
            raise ValueError(f"[forward] video_feats 期望 [B,T,D]，得到 {tuple(video_feats.shape)}")

        B, T, Din = video_feats.shape
        D = self.word_embeddings.embedding_dim
        if Din != D:
            raise ValueError(
                f"[forward] video_feats hidden dim ({Din}) != d_model ({D})。"
            )

        # === 1) 视频：mean pooling 成单 token memory ===
        # 你的 vid_mask: 1=有效帧；SequenceMeanPooling 里 1=参与均值
        memory = self.mean_pooling(video_feats, attention_mask=vid_mask)  # [B, 1, D]
        # 因为 memory 只有 1 个 token，直接设为全 False（不忽略）
        memory_key_padding_mask = torch.zeros(B, 1, dtype=torch.bool, device=memory.device)  # [B,1]

        # === 2) 文本：前 76 做输入 ===
        if captions.size(1) < 2:
            raise ValueError("captions 长度需 ≥ 2（如固定 77）才能做右移预测。")
        input_ids = captions[:, :-1].contiguous()  # [B, 76]
        attn_mask = caption_mask[:, :-1].contiguous() if caption_mask is not None else None  # [B, 76] 1=有效

        L = input_ids.size(1)  # 76
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)  # [B, 76]

        # Token Embedding + Positional Embedding（内部返回 x+pos）
        tgt_emb = self.word_embeddings(input_ids)  # [B, 76, D]
        tgt_emb = self.pos_embeddings(tgt_emb, position_ids=pos_ids, attention_mask=attn_mask)  # [B, 76, D]
        tgt_emb = self.norm_input(tgt_emb)  # [B, 76, D]

        # === 3) mask 规范：PyTorch 的 *_key_padding_mask -> True=忽略；tgt_mask(True=禁止看未来) ===
        tgt_mask = torch.triu(torch.ones(L, L, device=input_ids.device, dtype=torch.bool), diagonal=1)  # [L,L] bool
        tgt_key_padding_mask = (attn_mask == 0).bool() if attn_mask is not None else None               # [B,76] bool

        # === 4) 解码器（batch_first=True，无需转置）===
        hidden = self.decoder(
            tgt=tgt_emb,                                  # [B,76,D]
            memory=memory,                                # [B, 1,D]
            tgt_mask=tgt_mask,                            # [L,L]  True=屏蔽未来
            tgt_key_padding_mask=tgt_key_padding_mask,    # [B,76] True=忽略 PAD
            memory_key_padding_mask=memory_key_padding_mask,  # [B,1] False=不过滤
        )  # -> [B,76,D]

        # === 5) 输出 ===
        hidden = self.norm_output(hidden)     # [B,76,D]
        logits = self.lm_head(hidden)         # [B,76,V]
        
        if return_hidden:
            return logits, hidden  # 返回 logits 和 hidden states 用于对比学习
        return logits


import torch
import torch.nn as nn
import torch.nn.functional as F

# 依赖：CaptionModel_Base / LearnedPositionalEmbedding / TypeEmbedding / SequenceMeanPooling / load_clip_model
# 已在 models.py 中定义，本文只新增该类。

class CaptionModel_LenControl(CaptionModel_Base):
    """
    长度可控版本：
    - 在 BOS 之后插入一个 length token（由归一化长度 h 线性投影得到）
    - 为 length token 指定独立的 type id（与普通文本 token 区分）
    - 解码输入长度从 L 变为 L+1，但训练时只对后 L-1 个位置计算 CE：
        logits_for_loss = logits_full[:, 2:, :]  # 与 target = captions[:, 1:] 对齐，形状 [B,76,V]

    兼容原训练脚本的 forward 签名：forward(video_feats, vid_mask, captions, caption_mask, **kwargs)
    你也可以在推理时传入：
        - len_target_total: 目标 token 总数（含 BOS/EOS）（float/int 标量或 [B]）
        - len_h:            直接给定 h∈[0,1]（float 标量或 [B]）
    若均未提供，则训练时自动用 caption_mask 统计真实长度，推理时退化为 h=0.5。
    """
    def __init__(self,
                 vocab_size: int = 49408,
                 decoder_nhead: int = 8,
                 d_model: int = 512,
                 deocder_layer_nums: int = 3,
                 pretrained_clip_name: str = "ViT-B/32",
                 init_we: bool = True,
                 init_lmhead: bool = True,
                 frozen_we: bool = True,
                 frozen_lmhead: bool = False,
                 pad_token_id: int = 0,
                 bos_token_id: int = 49406,
                 eos_token_id: int = 49407,
                 # 长度归一化的 min-max，可按数据统计覆盖
                 len_min: int = 4,
                 len_max: int = 76,
                 # type embedding 的类型数量：0=PAD, 1=TEXT, 2=LEN
                 num_types: int = 3):
        super().__init__(vocab_size=vocab_size,
                         decoder_nhead=decoder_nhead,
                         d_model=d_model,
                         deocder_layer_nums=deocder_layer_nums,
                         pretrained_clip_name=pretrained_clip_name,
                         init_we=init_we,
                         init_lmhead=init_lmhead,
                         frozen_we=frozen_we,
                         frozen_lmhead=frozen_lmhead,
                         pad_token_id=pad_token_id,
                         bos_token_id=bos_token_id,
                         eos_token_id=eos_token_id)

        # 长度 token 的线性投影： e_len = W_l * h + b_l  (h: [B,1] ∈ [0,1])
        self.len_proj = nn.Linear(1, d_model)

        # 类型嵌入：与普通文本区分 length token
        self.type_embeddings = TypeEmbedding(num_types=num_types, embedding_dim=d_model, padding_idx=0)
        self.type_id_text = 1
        self.type_id_len  = 2

        # 记录 min-max（buffer 以便落到 model.state_dict）
        self.register_buffer("len_min_b", torch.tensor(float(len_min), dtype=torch.float32))
        self.register_buffer("len_max_b", torch.tensor(float(len_max), dtype=torch.float32))

    @torch.no_grad()
    def _norm_len_h(self,
                    batch_size: int,
                    device: torch.device,
                    caption_mask: torch.Tensor = None,      # [B,77] (1/0)
                    len_target_total=None,                   # 标量或 [B]
                    len_h=None):                             # 标量或 [B]
        """
        归一化长度 h∈[0,1] 的获取优先级：
        len_h (直接给) > len_target_total (数值→min-max 归一化) > caption_mask 统计 > 默认 0.5
        返回形状 [B,1]、dtype=float32、device 对齐。
        """
        B = batch_size
        # 1) 直接给 h
        if len_h is not None:
            h = torch.as_tensor(len_h, dtype=torch.float32, device=device)
            if h.dim() == 0:
                h = h.expand(B)
            h = h.clamp(0.0, 1.0)
            return h.view(B, 1)

        # 2) 给了目标总长度（含 BOS/EOS）
        if len_target_total is not None:
            L = torch.as_tensor(len_target_total, dtype=torch.float32, device=device)
            if L.dim() == 0:
                L = L.expand(B)
            L = torch.clamp(L, min=float(self.len_min_b.item()), max=float(self.len_max_b.item()))
            denom = max(1.0, float(self.len_max_b.item() - self.len_min_b.item()))
            h = (L - self.len_min_b) / denom
            return h.view(B, 1)

        # 3) 训练：由 caption_mask 统计真实长度
        if caption_mask is not None:
            L = caption_mask.to(torch.float32).sum(dim=1)  # [B]
            L = torch.clamp(L, min=float(self.len_min_b.item()), max=float(self.len_max_b.item()))
            denom = max(1.0, float(self.len_max_b.item() - self.len_min_b.item()))
            h = (L - self.len_min_b) / denom
            return h.view(B, 1)

        # 4) 兜底
        return torch.full((B, 1), 0.5, dtype=torch.float32, device=device)

    def forward(self,
                video_feats: torch.Tensor,            # [B, T, D]
                vid_mask: torch.Tensor,               # [B, T]  (1=valid, 0=pad)
                captions: torch.Tensor,               # [B, 77]
                caption_mask: torch.Tensor = None,    # [B, 77]
                # 可选：推理时外部控长
                len_target_total=None,                # 标量或 [B]，目标 token 总数（含 BOS/EOS）
                len_h=None):                          # 标量或 [B]，直接给 h∈[0,1]
        """
        训练 (teacher forcing)：
          input_ids = captions[:, :-1]  # [B,76]
          target    = captions[:,  1:]  # [B,76]
        解码器输入在 BOS 之后插入 length token，使得解码输入长度从 76 -> 77；
        计算损失时使用 logits_full[:, 2:, :] 与 target 对齐。
        """
        if video_feats.dim() != 3:
            raise ValueError(f"[forward] video_feats 期望 [B,T,D]，得到 {tuple(video_feats.shape)}")

        B, T, Din = video_feats.shape
        D = self.word_embeddings.embedding_dim
        if Din != D:
            raise ValueError(f"[forward] video_feats hidden dim ({Din}) != d_model ({D})。")

        # === 1) 视频 memory：mask-aware mean pooling ===
        memory = self.mean_pooling(video_feats, attention_mask=vid_mask)  # [B,1,D]
        memory_key_padding_mask = torch.zeros(B, 1, dtype=torch.bool, device=video_feats.device)

        # === 2) 文本右移 ===
        if captions.size(1) < 2:
            raise ValueError("captions 长度需 ≥ 2（如固定 77）才能做右移预测。")
        input_ids = captions[:, :-1].contiguous()                  # [B,76]
        attn_mask_in = caption_mask[:, :-1].contiguous() if caption_mask is not None else torch.ones_like(input_ids)

        L = input_ids.size(1)                                      # 76
        # 基础词嵌入（先不加位置，便于插入 length token 后再统一加）
        we = self.word_embeddings(input_ids)                       # [B,76,D]

        # === 3) 计算 h 与 length token ===
        h = self._norm_len_h(batch_size=B,
                             device=video_feats.device,
                             caption_mask=caption_mask,
                             len_target_total=len_target_total,
                             len_h=len_h)  # [B,1]
        e_len = self.len_proj(h).unsqueeze(1)  # [B,1,D] <- 关键：unsqueeze(1)

        # === 4) 在 BOS 后插入 length token ===
        x_bos = we[:, :1, :]  # [B,1,D]
        x_tail = we[:, 1:, :]  # [B,75,D]
        x = torch.cat([x_bos, e_len.to(dtype=we.dtype), x_tail], dim=1)  # [B,77,D]

        # 扩展注意力 mask：在索引 1 处插入有效位 1
        one = torch.ones(B, 1, dtype=attn_mask_in.dtype, device=attn_mask_in.device)
        attn_mask_full = torch.cat([attn_mask_in[:, :1], one, attn_mask_in[:, 1:]], dim=1)  # [B,77]

        # 位置编码：显式提供 0..76 的 position_ids，确保 LEN 占用位置 1
        pos_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(B, x.size(1))  # [B,77]
        x = self.pos_embeddings(x, position_ids=pos_ids, attention_mask=attn_mask_full)       # [B,77,D]

        # 类型嵌入：0=BOS(text), 1=LEN(len), 其余=text
        type_ids = torch.full((B, x.size(1)), self.type_id_text, dtype=torch.long, device=x.device)  # [B,77]
        type_ids[:, 1] = self.type_id_len
        x = self.type_embeddings(x, type_ids=type_ids, attention_mask=attn_mask_full)         # [B,77,D]
        x = self.norm_input(x)

        # === 5) Transformer 解码 ===
        Lp = x.size(1)  # 77
        tgt_mask = torch.triu(torch.ones(Lp, Lp, device=x.device, dtype=torch.bool), diagonal=1)   # [Lp,Lp] True=mask future
        tgt_key_padding_mask = (attn_mask_full == 0).bool()                                        # [B,Lp]

        hidden = self.decoder(
            tgt=x,                                           # [B,77,D]
            memory=memory,                                   # [B,1,D]
            tgt_mask=tgt_mask,                               # [Lp,Lp]
            tgt_key_padding_mask=tgt_key_padding_mask,       # [B,77]
            memory_key_padding_mask=memory_key_padding_mask  # [B,1]
        )                                                    # -> [B,77,D]

        hidden = self.norm_output(hidden)
        logits_full = self.lm_head(hidden)                   # [B,77,V]

        # 与 target=captions[:,1:]（长度 76）对齐：丢弃位置 0(BOS) 与 1(LEN)，取 2..76 共 76 个
        logits = logits_full[:, 1:, :].contiguous()          # [B,76,V]
        return logits



class CaptionModel_Base_NoMeanPooling(CaptionModel_Base):
    """
    去均值池化版本：
    - 不再对 video_feats 做 mean pooling
    - 在 video_feats 上添加位置编码（Sinusoidal / 可改为 Learned）
    - 直接以 [B, T, D] 形式作为 Decoder 的 memory，并传入对应的 padding mask
    - 文本侧保持与基类一致（词嵌入 + 可学习绝对位置编码 + 层归一化）
    """
    def __init__(self,
                 vocab_size: int = 49408,
                 decoder_nhead: int = 8,
                 d_model: int = 512,
                 deocder_layer_nums: int = 3,
                 pretrained_clip_name: str = "ViT-B/32",
                 init_we: bool = True,
                 init_lmhead: bool = True,
                 frozen_we: bool = True,
                 frozen_lmhead: bool = False,
                 pad_token_id: int = 0,
                 bos_token_id: int = 49406,
                 eos_token_id: int = 49407,
                 # 选择视频侧位置编码的类型：'sin' 或 'learned'
                 video_pos_type: str = "sin",
                 video_max_len: int = 256,
                 video_pos_dropout: float = 0.0):
        """
        说明：
        - 继续沿用基类中：词嵌入、文本位置嵌入、Decoder、输出层、以及 CLIP 权重初始化逻辑
        - 仅新增 video 侧的位置编码模块，并在 forward 中启用"整段 memory"路径
        """
        super().__init__(vocab_size=vocab_size,
                         decoder_nhead=decoder_nhead,
                         d_model=d_model,
                         deocder_layer_nums=deocder_layer_nums,
                         pretrained_clip_name=pretrained_clip_name,
                         init_we=init_we,
                         init_lmhead=init_lmhead,
                         frozen_we=frozen_we,
                         frozen_lmhead=frozen_lmhead,
                         pad_token_id=pad_token_id,
                         bos_token_id=bos_token_id,
                         eos_token_id=eos_token_id)

        # --- 新增：视频侧位置编码 ---
        if video_pos_type.lower() == "sin":
            self.video_pos = SinusoidalPositionalEncoding(
                d_model=d_model, max_len=video_max_len, dropout=video_pos_dropout, scale_x=True
            )
        elif video_pos_type.lower() == "learned":
            self.video_pos = LearnedPositionalEmbedding(
                num_positions=video_max_len, embedding_dim=d_model, dropout=video_pos_dropout, padding_idx=0
            )
        else:
            raise ValueError(f"Unsupported video_pos_type: {video_pos_type}, choose from ['sin','learned'].")

    def forward(
        self,
        video_feats: torch.Tensor,            # [B, T, D]
        vid_mask: torch.Tensor,               # [B, T]  1=有效, 0=pad
        captions: torch.Tensor,               # [B, 77]
        caption_mask: Optional[torch.Tensor] = None,  # [B, 77] 1=有效, 0=pad
    ) -> torch.Tensor:
        """
        训练(teacher forcing)：
          input_ids = captions[:, :-1]  # [B,76]
          target    = captions[:,  1:]  # [B,76]
        输出：
          logits: [B, 76, vocab_size]
        """
        if video_feats.dim() != 3:
            raise ValueError(f"[forward] video_feats 期望 [B,T,D]，得到 {tuple(video_feats.shape)}")

        B, T, Din = video_feats.shape
        D = self.word_embeddings.embedding_dim
        if Din != D:
            raise ValueError(f"[forward] video_feats hidden dim ({Din}) != d_model ({D})。")

        # === 1) 视频侧：添加位置编码，整段作为 memory ===
        # 对于 learned 版本：若提供 attention_mask，可令 pad 位置使用 padding_idx 的 0 向量
        if isinstance(self.video_pos, LearnedPositionalEmbedding):
            memory = self.video_pos(video_feats, attention_mask=vid_mask)  # [B,T,D]
        else:
            # Sinusoidal：按 0..T-1 生成 position_ids
            pos_ids_v = torch.arange(T, device=video_feats.device).unsqueeze(0).expand(B, T)  # [B,T]
            memory = self.video_pos(video_feats, position_ids=pos_ids_v)                      # [B,T,D]

        # 注意：PyTorch decoder 的 *_key_padding_mask 约定 True=忽略
        memory_key_padding_mask = (vid_mask == 0).bool()  # [B,T]

        # === 2) 文本右移，嵌入 + 文本位置编码 ===
        if captions.size(1) < 2:
            raise ValueError("captions 长度需 ≥ 2（如固定 77）才能做右移预测。")
        input_ids = captions[:, :-1].contiguous()  # [B,76]
        attn_mask = caption_mask[:, :-1].contiguous() if caption_mask is not None else None  # [B,76]

        L = input_ids.size(1)
        pos_ids_t = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)      # [B,76]
        tgt_emb = self.word_embeddings(input_ids)                                           # [B,76,D]
        tgt_emb = self.pos_embeddings(tgt_emb, position_ids=pos_ids_t, attention_mask=attn_mask)  # [B,76,D]
        tgt_emb = self.norm_input(tgt_emb)                                                  # [B,76,D]

        # === 3) 构造 mask 并解码 ===
        tgt_mask = torch.triu(torch.ones(L, L, device=input_ids.device, dtype=torch.bool), diagonal=1)  # [L,L]
        tgt_key_padding_mask = (attn_mask == 0).bool() if attn_mask is not None else None               # [B,76]

        hidden = self.decoder(
            tgt=tgt_emb,                                  # [B,76,D]
            memory=memory,                                # [B,T,D]
            tgt_mask=tgt_mask,                            # [L,L]  True=屏蔽未来
            tgt_key_padding_mask=tgt_key_padding_mask,    # [B,76] True=忽略 PAD
            memory_key_padding_mask=memory_key_padding_mask,  # [B,T]  True=忽略 PAD
        )  # -> [B,76,D]

        hidden = self.norm_output(hidden)                 # [B,76,D]
        logits = self.lm_head(hidden)                     # [B,76,V]
        return logits

# === 新增：CaptionModel_Base_TemporalMeanPooling =========================================
# 放到 models.py 里（与其它 CaptionModel_* 并列即可）

class CaptionModel_Base_TemporalMeanPooling(CaptionModel_Base):
    """
    在 mean pooling 前增加"时序 Transformer Encoder"：
    video_feats -> (视频位置编码) -> TransformerEncoder -> 均值池化(按 mask) -> 单 token memory
    再与文本侧解码流程对接（与 base 完全一致的 right-shift 训练 / logits 对齐）。
    """
    def __init__(self,
                 vocab_size: int = 49408,
                 decoder_nhead: int = 8,
                 d_model: int = 512,
                 deocder_layer_nums: int = 3,
                 pretrained_clip_name: str = "ViT-B/32",
                 init_we: bool = True,
                 init_lmhead: bool = True,
                 frozen_we: bool = True,
                 frozen_lmhead: bool = False,
                 pad_token_id: int = 0,
                 bos_token_id: int = 49406,
                 eos_token_id: int = 49407,
                 # --- 新增：视频侧时序建模参数 ---
                 video_pos_type: str = "sin",      # 'sin' or 'learned'
                 video_max_len: int = 256,
                 video_pos_dropout: float = 0.1,
                 video_nhead: int = 8,
                 video_layer_nums: int = 2,
                 video_ffn_dim: int = 2048,
                 ):
        super().__init__(vocab_size=vocab_size,
                         decoder_nhead=decoder_nhead,
                         d_model=d_model,
                         deocder_layer_nums=deocder_layer_nums,
                         pretrained_clip_name=pretrained_clip_name,
                         init_we=init_we,
                         init_lmhead=init_lmhead,
                         frozen_we=frozen_we,
                         frozen_lmhead=frozen_lmhead,
                         pad_token_id=pad_token_id,
                         bos_token_id=bos_token_id,
                         eos_token_id=eos_token_id)

        # 1) 视频侧位置编码
        if str(video_pos_type).lower() == "sin":
            self.video_pos = SinusoidalPositionalEncoding(
                d_model=d_model, max_len=video_max_len, dropout=video_pos_dropout, scale_x=True
            )
        elif str(video_pos_type).lower() == "learned":
            self.video_pos = LearnedPositionalEmbedding(
                num_positions=video_max_len, embedding_dim=d_model, dropout=video_pos_dropout, padding_idx=0
            )
        else:
            raise ValueError(f"Unsupported video_pos_type: {video_pos_type}, choose from ['sin','learned'].")

        # 2) 视频侧 Transformer Encoder（batch_first=True）
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=video_nhead,
            dim_feedforward=video_ffn_dim,
            dropout=0.1,
            batch_first=True,
        )
        self.video_encoder = nn.TransformerEncoder(enc_layer, num_layers=video_layer_nums)

    def forward(
        self,
        video_feats: torch.Tensor,            # [B, T, D]
        vid_mask: torch.Tensor,               # [B, T]  1=有效, 0=pad
        captions: torch.Tensor,               # [B, 77]
        caption_mask: Optional[torch.Tensor] = None,  # [B, 77]
    ) -> torch.Tensor:
        """
        与 base 一致的 teacher-forcing：input = 前 76，target = 后 76；返回 logits[B,76,V]
        """
        if video_feats.dim() != 3:
            raise ValueError(f"[forward] video_feats 期望 [B,T,D]，得到 {tuple(video_feats.shape)}")

        B, T, Din = video_feats.shape
        D = self.word_embeddings.embedding_dim
        if Din != D:
            raise ValueError(f"[forward] video_feats hidden dim ({Din}) != d_model ({D})。")

        # === (A) 视频侧：位置编码 -> TransformerEncoder ===
        if isinstance(self.video_pos, LearnedPositionalEmbedding):
            v = self.video_pos(video_feats, attention_mask=vid_mask)           # [B,T,D]
        else:
            pos_ids_v = torch.arange(T, device=video_feats.device).unsqueeze(0).expand(B, T)
            v = self.video_pos(video_feats, position_ids=pos_ids_v)            # [B,T,D]

        src_key_padding_mask = (vid_mask == 0).bool()                           # True = 忽略
        v_enc = self.video_encoder(v, src_key_padding_mask=src_key_padding_mask)  # [B,T,D]

        # === (B) 均值池化到单 token memory（按 mask 感知） ===
        memory = self.mean_pooling(v_enc, attention_mask=vid_mask)              # [B,1,D]
        memory_key_padding_mask = torch.zeros(B, 1, dtype=torch.bool, device=video_feats.device)

        # === (C) 文本侧（与 base 完全一致） ===
        if captions.size(1) < 2:
            raise ValueError("captions 长度需 ≥ 2（如固定 77）才能做右移预测。")
        input_ids = captions[:, :-1].contiguous()                                # [B,76]
        attn_mask = caption_mask[:, :-1].contiguous() if caption_mask is not None else None

        L = input_ids.size(1)
        pos_ids_t = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        tgt_emb = self.word_embeddings(input_ids)                                 # [B,76,D]
        tgt_emb = self.pos_embeddings(tgt_emb, position_ids=pos_ids_t, attention_mask=attn_mask)
        tgt_emb = self.norm_input(tgt_emb)

        # mask 规范（True=屏蔽/忽略）
        tgt_mask = torch.triu(torch.ones(L, L, device=input_ids.device, dtype=torch.bool), diagonal=1)
        tgt_key_padding_mask = (attn_mask == 0).bool() if attn_mask is not None else None

        hidden = self.decoder(
            tgt=tgt_emb,
            memory=memory,                               # 单 token memory
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )                                               # [B,76,D]

        hidden = self.norm_output(hidden)
        logits = self.lm_head(hidden)                   # [B,76,V]
        return logits
# === 结束：新增类 =========================================================================





# ===== Contrastive Learning Model Built on Base (Academic-Grade Implementation) =====

class CaptionModel_BaseCL(CaptionModel_Base):
    """
    基于 CaptionModel_Base 的对比学习版本（方案2：解码器隐藏状态）：
    - 继承 Base 的所有结构和功能
    - 新增视频-句子对比学习机制
    - 视频编码：使用 mean pooling 获得全局表示
    - 文本编码：使用解码器隐藏状态的平均池化（梯度可回传）✅
    - 对比学习损失：双向 InfoNCE (video-to-text + text-to-video)
    - 梯度流：CL_loss → text_projection → decoder → 主任务参数 ✅
    
    【学术优势】
    - 对比学习梯度直接优化解码器参数
    - 无需额外的文本编码器（CLIP冻结问题已解决）
    - 端到端训练，视频-文本对齐更紧密
    """
    def __init__(self,
                 vocab_size: int = 49408,
                 decoder_nhead: int = 8,
                 d_model: int = 512,
                 deocder_layer_nums: int = 3,
                 pretrained_clip_name: str = "ViT-B/32",
                 init_we: bool = True,
                 init_lmhead: bool = True,
                 frozen_we: bool = True,
                 frozen_lmhead: bool = False,
                 pad_token_id: int = 0,
                 bos_token_id: int = 49406,
                 eos_token_id: int = 49407,
                 # 对比学习参数
                 cl_temperature: float = 0.07,
                 cl_weight: float = 0.1,
                 cl_projection_dim: int = 256,
                 ):
        super().__init__(vocab_size=vocab_size,
                         decoder_nhead=decoder_nhead,
                         d_model=d_model,
                         deocder_layer_nums=deocder_layer_nums,
                         pretrained_clip_name=pretrained_clip_name,
                         init_we=init_we,
                         init_lmhead=init_lmhead,
                         frozen_we=frozen_we,
                         frozen_lmhead=frozen_lmhead,
                         pad_token_id=pad_token_id,
                         bos_token_id=bos_token_id,
                         eos_token_id=eos_token_id)
        
        # 对比学习参数
        self.cl_temperature = cl_temperature
        self.cl_weight = cl_weight
        
        # 投影层：将视频和文本特征投影到对比学习空间
        self.video_projection = nn.Sequential(
            nn.Linear(d_model, cl_projection_dim),
            nn.ReLU(),
            nn.Linear(cl_projection_dim, cl_projection_dim)
        )
        
        self.text_projection = nn.Sequential(
            nn.Linear(d_model, cl_projection_dim),
            nn.ReLU(),
            nn.Linear(cl_projection_dim, cl_projection_dim)
        )
        
        # 初始化投影层
        self._init_contrastive_projections()
        
    def _init_contrastive_projections(self):
        """初始化对比学习投影层"""
        for proj in [self.video_projection, self.text_projection]:
            for layer in proj:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def encode_video(self, video_feats: torch.Tensor, vid_mask: torch.Tensor) -> torch.Tensor:
        """
        编码视频特征为全局表示
        Args:
            video_feats: [B, T, D] 视频特征
            vid_mask: [B, T] 视频mask
        Returns:
            video_global: [B, D] 全局视频表示
        """
        # 使用 mean pooling 获得全局视频表示
        video_global = self.mean_pooling(video_feats, attention_mask=vid_mask).squeeze(1)  # [B, D]
        return video_global
    
    def forward(self,
                video_feats: torch.Tensor,            # [B, T, D]
                vid_mask: torch.Tensor,               # [B, T]  1=有效, 0=pad
                captions: torch.Tensor,               # [B, 77]
                caption_mask: Optional[torch.Tensor] = None,  # [B, 77] 1=有效, 0=pad
                return_contrastive: bool = False,     # 是否返回对比学习特征
                ) -> torch.Tensor:
        """
        前向传播，支持对比学习（方案2：解码器隐藏状态）
        Args:
            video_feats: 视频特征
            vid_mask: 视频mask
            captions: 句子token ids
            caption_mask: 句子mask
            return_contrastive: 是否返回对比学习特征
        Returns:
            logits: [B, 76, vocab_size] 语言模型logits
            如果 return_contrastive=True，额外返回 (video_cls, text_cls) 用于对比学习
        """
        if return_contrastive:
            # 1. 执行基类前向传播，获取 logits 和 hidden states
            logits, hidden = super().forward(
                video_feats, vid_mask, captions, caption_mask, 
                return_hidden=True  # 关键：获取解码器隐藏状态
            )
            
            # 2. 视频表示：mean pooling（与之前一致）
            video_global = self.encode_video(video_feats, vid_mask)  # [B, D]
            video_cls = self.video_projection(video_global)  # [B, cl_projection_dim]
            video_cls = F.normalize(video_cls, p=2, dim=1)
            
            # 3. 文本表示：使用解码器输出的隐藏状态
            # A1策略：平均池化所有有效token的hidden states
            caption_mask_76 = caption_mask[:, :-1] if caption_mask is not None else None  # [B, 76]
            text_repr = self.mean_pooling(hidden, attention_mask=caption_mask_76)  # [B, 1, D]
            text_repr = text_repr.squeeze(1)  # [B, D]
            text_cls = self.text_projection(text_repr)  # [B, cl_projection_dim]
            text_cls = F.normalize(text_cls, p=2, dim=1)
            
            return logits, (video_cls, text_cls)
        else:
            # 推理时不需要对比学习特征
            return super().forward(video_feats, vid_mask, captions, caption_mask)
    
    def compute_contrastive_loss(self, video_cls: torch.Tensor, text_cls: torch.Tensor) -> torch.Tensor:
        """
        计算对比学习损失
        Args:
            video_cls: [B, D] 视频特征
            text_cls: [B, D] 文本特征
        Returns:
            contrastive_loss: 标量损失
        """
        # 确保特征已经归一化
        video_cls = F.normalize(video_cls, p=2, dim=1)
        text_cls = F.normalize(text_cls, p=2, dim=1)
        
        # 计算相似度矩阵
        logits = torch.matmul(video_cls, text_cls.T) / self.cl_temperature  # [B, B]
        
        # 标签：对角线为正样本
        labels = torch.arange(logits.size(0), device=logits.device)
        
        # 双向对比学习损失
        loss_v2t = F.cross_entropy(logits, labels)
        loss_t2v = F.cross_entropy(logits.T, labels)
        contrastive_loss = (loss_v2t + loss_t2v) / 2
        
        return contrastive_loss


# ===== 单向对比学习模型（README11_CL方案）=====

class CaptionModel_Base_CL(CaptionModel_Base):
    """
    单向对比学习版本（基于README11_CL方案）：
    
    【核心设计】
    - 视频嵌入：离线CLIP特征 + mask感知均值池化 + L2归一化
    - 文本嵌入：CLIP.encode_text(cap_str) + L2归一化（冻结CLIP）
    - 对比损失：单向v→t的InfoNCE
    - 温度参数：可学习，从CLIP logit_scale初始化
    
    【训练目标】
    L_total = L_CE + λ * L_v→t
    
    其中：
    - L_CE: 原始Caption交叉熵损失（与Base一致）
    - L_v→t: 单向视频到文本的InfoNCE对比损失
    - λ: 对比学习权重（建议0.1～0.5）
    
    【使用方式】
    # 训练时：
    logits, cl_loss, cl_acc = model(
        video_feats, vid_mask, captions, caption_mask,
        cap_str_list=batch_cap_strs,  # 原始caption字符串列表
        return_contrastive=True
    )
    loss_ce = CE_loss(logits, targets)
    loss_total = loss_ce + model.cl_weight * cl_loss
    
    # 推理时：
    logits = model(video_feats, vid_mask, captions, caption_mask)
    """
    
    def __init__(
        self,
        vocab_size: int = 49408,
        decoder_nhead: int = 8,
        d_model: int = 512,
        deocder_layer_nums: int = 3,
        pretrained_clip_name: str = "ViT-B/32",
        init_we: bool = True,
        init_lmhead: bool = True,
        frozen_we: bool = True,
        frozen_lmhead: bool = False,
        pad_token_id: int = 0,
        bos_token_id: int = 49406,
        eos_token_id: int = 49407,
        # 对比学习专用参数
        cl_weight: float = 0.1,
        learnable_temperature: bool = True,
    ):
        """
        初始化单向对比学习Caption模型
        
        Args:
            cl_weight: 对比学习损失权重λ，建议0.1～0.5
            learnable_temperature: 是否学习温度参数，默认True
            其余参数同CaptionModel_Base
        """
        # 继承Base的所有功能
        super().__init__(
            vocab_size=vocab_size,
            decoder_nhead=decoder_nhead,
            d_model=d_model,
            deocder_layer_nums=deocder_layer_nums,
            pretrained_clip_name=pretrained_clip_name,
            init_we=init_we,
            init_lmhead=init_lmhead,
            frozen_we=frozen_we,
            frozen_lmhead=frozen_lmhead,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )
        
        # 存储对比学习超参数
        self.cl_weight = cl_weight
        self.learnable_temperature = learnable_temperature
        
        # 加载CLIP模型用于文本编码（注意：这里复用Base里已经加载的模型）
        # Base的__init__已经加载了self.pretrained_clip_model
        # 我们直接使用它，不需要重新加载
        self.clip_model = self.pretrained_clip_model
        
        # 确保CLIP模型冻结（已在Base中冻结，这里再次确认）
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.clip_model.eval()
        
        # 加载CLIP tokenizer（使用项目中的CLIPTokenizer_Custom）
        from load_tokenizers import CLIPTokenizer_Custom
        self.clip_tokenizer = CLIPTokenizer_Custom(swap_pad_token=True)
        
        # 初始化可学习温度参数
        # CLIP的logit_scale初始值约为ln(1/0.07) ≈ 2.659
        # 我们从CLIP的logit_scale初始化
        if hasattr(self.clip_model, 'logit_scale'):
            init_logit_scale = self.clip_model.logit_scale.item()
        else:
            init_logit_scale = math.log(1.0 / 0.07)  # 约2.659
        
        if learnable_temperature:
            # 可学习的温度参数（存储logit_scale，实际温度为exp(-logit_scale)）
            self.logit_scale = nn.Parameter(torch.tensor([init_logit_scale]))
        else:
            # 固定温度
            self.register_buffer('logit_scale', torch.tensor([init_logit_scale]))
        
        print(f"[CaptionModel_Base_CL] 初始化完成:")
        print(f"  - 对比学习权重: {cl_weight}")
        print(f"  - 温度参数可学习: {learnable_temperature}")
        print(f"  - 初始logit_scale: {init_logit_scale:.4f} (温度: {math.exp(-init_logit_scale):.4f})")
    
    def encode_video(
        self,
        video_feats: torch.Tensor,  # [B, T, D]
        vid_mask: torch.Tensor,     # [B, T] 1=有效, 0=pad
    ) -> torch.Tensor:
        """
        编码视频为归一化的嵌入向量
        
        流程：
        1. 使用mask感知均值池化（与Base一致）
        2. L2归一化
        
        Args:
            video_feats: 离线CLIP视频特征 [B, T, D]
            vid_mask: 有效帧掩码 [B, T]
        
        Returns:
            video_emb: 归一化后的视频嵌入 [B, D]
        """
        # 1) mask感知均值池化（复用Base的mean_pooling）
        video_pooled = self.mean_pooling(video_feats, attention_mask=vid_mask)  # [B, 1, D]
        video_pooled = video_pooled.squeeze(1)  # [B, D]
        
        # 2) L2归一化
        video_emb = F.normalize(video_pooled, p=2, dim=-1)  # [B, D]
        
        return video_emb
    
    def encode_text(
        self,
        cap_str_list: List[str],  # 批次中的原始caption字符串
    ) -> torch.Tensor:
        """
        编码文本为归一化的嵌入向量
        
        流程：
        1. 使用CLIPTokenizer_Custom处理文本
        2. 调用CLIP.encode_text获取句子嵌入
        3. L2归一化
        
        Args:
            cap_str_list: 原始caption字符串列表，长度为B
        
        Returns:
            text_emb: 归一化后的文本嵌入 [B, D]
        """
        # 使用CLIPTokenizer_Custom进行批量编码
        # 与训练脚本保持一致：padding='max_length', max_length=77, truncation=True
        encoded = self.clip_tokenizer._tokenizer.batch_encode_plus(
            cap_str_list,
            padding='max_length',
            max_length=77,
            truncation=True,
            return_tensors='pt'
        )
        
        # 获取input_ids并转移到设备
        text_tokens = encoded['input_ids'].to(
            next(self.clip_model.parameters()).device
        )  # [B, 77]
        
        # 使用CLIP编码文本（冻结模式，不计算梯度）
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)  # [B, D]
        
        # L2归一化
        text_emb = F.normalize(text_features, p=2, dim=-1)  # [B, D]
        
        return text_emb
    
    def compute_contrastive_loss_v2t(
        self,
        video_emb: torch.Tensor,  # [B, D] 已归一化
        text_emb: torch.Tensor,   # [B, D] 已归一化
        use_all_gather: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算单向v→t的InfoNCE对比损失
        
        公式：
        Z_ij = (v_i · s_j) / τ
        L_v→t = -1/B Σ_i log( exp(Z_ii) / Σ_j exp(Z_ij) )
        
        Args:
            video_emb: 归一化后的视频嵌入 [B, D]
            text_emb: 归一化后的文本嵌入 [B, D]
            use_all_gather: 是否使用DDP的all_gather扩展负样本
        
        Returns:
            loss: 对比损失标量
            acc: 批内检索准确率@1
        """
        B = video_emb.shape[0]
        device = video_emb.device
        
        # === DDP支持（可选）===
        if use_all_gather and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            if world_size > 1:
                # 收集所有GPU的嵌入
                video_emb_list = [torch.zeros_like(video_emb) for _ in range(world_size)]
                text_emb_list = [torch.zeros_like(text_emb) for _ in range(world_size)]
                
                torch.distributed.all_gather(video_emb_list, video_emb)
                torch.distributed.all_gather(text_emb_list, text_emb)
                
                # 拼接成全局batch
                video_emb_global = torch.cat(video_emb_list, dim=0)  # [B*world, D]
                text_emb_global = torch.cat(text_emb_list, dim=0)    # [B*world, D]
                
                # 计算相似度矩阵（当前rank的video vs 全局text）
                logits = video_emb @ text_emb_global.T  # [B, B*world]
                
                # 正样本索引偏移
                rank = torch.distributed.get_rank()
                labels = torch.arange(B, device=device) + rank * B  # [B]
            else:
                # 单GPU，直接计算
                logits = video_emb @ text_emb.T  # [B, B]
                labels = torch.arange(B, device=device)
        else:
            # 不使用all_gather
            logits = video_emb @ text_emb.T  # [B, B]
            labels = torch.arange(B, device=device)
        
        # === 应用温度参数 ===
        # τ = exp(-logit_scale)，相当于 logits / τ = logits * exp(logit_scale)
        temperature = torch.clamp(self.logit_scale.exp(), min=1e-6, max=100.0)
        logits = logits / temperature
        
        # === 计算单向v→t的InfoNCE损失 ===
        loss_v2t = F.cross_entropy(logits, labels)
        
        # === 计算acc@1（用于监控）===
        with torch.no_grad():
            pred = logits.argmax(dim=1)  # [B]
            acc = (pred == labels).float().mean()
        
        return loss_v2t, acc
    
    def forward(
        self,
        video_feats: torch.Tensor,                       # [B, T, D]
        vid_mask: torch.Tensor,                          # [B, T]
        captions: torch.Tensor,                          # [B, 77]
        caption_mask: Optional[torch.Tensor] = None,     # [B, 77]
        cap_str_list: Optional[List[str]] = None,        # 原始caption字符串列表
        return_contrastive: bool = False,                # 是否返回对比学习损失
        use_all_gather: bool = False,                    # 是否使用DDP all_gather
    ):
        """
        前向传播，支持对比学习
        
        训练模式（return_contrastive=True）：
            返回 (logits, cl_loss, cl_acc)
        推理模式（return_contrastive=False）：
            返回 logits
        
        Args:
            video_feats: 视频特征 [B, T, D]
            vid_mask: 视频mask [B, T]
            captions: token ids [B, 77]
            caption_mask: caption mask [B, 77]
            cap_str_list: 原始caption字符串列表（训练时需要）
            return_contrastive: 是否计算并返回对比学习损失
            use_all_gather: 是否使用DDP扩展负样本
        
        Returns:
            如果return_contrastive=True:
                (logits, cl_loss, cl_acc)
            否则:
                logits
        """
        # === 1) 基础Caption分支（CE损失） ===
        logits = super().forward(video_feats, vid_mask, captions, caption_mask)  # [B, 76, V]
        
        # === 2) 对比学习分支（可选） ===
        if return_contrastive:
            if cap_str_list is None:
                raise ValueError(
                    "[CaptionModel_Base_CL] 开启对比学习时需要提供cap_str_list（原始caption字符串）"
                )
            
            # 编码视频和文本
            video_emb = self.encode_video(video_feats, vid_mask)  # [B, D]
            text_emb = self.encode_text(cap_str_list)              # [B, D]
            
            # 计算对比损失
            cl_loss, cl_acc = self.compute_contrastive_loss_v2t(
                video_emb, text_emb, use_all_gather=use_all_gather
            )
            
            return logits, cl_loss, cl_acc
        
        else:
            # 推理模式，只返回logits
            return logits


# ===== RL model built on base (signature-aligned) =====


#-----------------------------------------------------------------
# Based on CaptionModel_Base
class CaptionModel_RL(CaptionModel_Base):
    """
    基于 Base 的 RL 版：
    - 结构与 Base 完全一致；SCST 的采样/奖励/损失都在训练脚本完成；
    - forward 的签名与基类完全一致（video_feats, vid_mask, captions, caption_mask=None）；
    - 暴露 pad/bos/eos 三个 token id；提供 step_logits() 便于逐步采样。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_token_id = getattr(self, "pad_token_id", kwargs.get("pad_token_id", 0))
        self.bos_token_id = getattr(self, "bos_token_id", kwargs.get("bos_token_id", 49406))
        self.eos_token_id = getattr(self, "eos_token_id", kwargs.get("eos_token_id", 49407))

    def forward(
        self,
        video_feats: torch.Tensor,           # [B, T, D]
        vid_mask: torch.Tensor,              # [B, T]
        captions: torch.Tensor,              # [B, 77]
        caption_mask: torch.Tensor = None,   # [B, 77]
    ) -> torch.Tensor:
        # 与 Base 完全一致，保持权重/行为统一
        return super().forward(video_feats, vid_mask, captions, caption_mask)

    @torch.no_grad()
    def step_logits(
        self,
        video_feats: torch.Tensor,
        vid_mask: torch.Tensor,
        prefix_ids: torch.Tensor,
        prefix_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        便捷函数：给定已生成前缀，返回下一步 logits（最后一位置）。
        与推理时构造 (prefix + 占位) 的用法一致。
        """
        logits = super().forward(video_feats, vid_mask, prefix_ids, prefix_mask)  # [B, L, V]
        return logits[:, -1, :]  # [B, V]



# ===== New: Router for length selection =====================================

class LenRouter(nn.Module):
    """
    输入：video_feats [B,T,D], vid_mask [B,T]
    输出：每个候选长度的打分 logits [B,K]
    """
    def __init__(self, d_model: int = 512, hidden: int = 512, num_layers: int = 2, num_candidates: int = 4, dropout: float = 0.1):
        super().__init__()
        self.pool = SequenceMeanPooling(dim=1, keepdim=False, eps=1e-12)  # 与基线一致的 mask-aware mean
        layers = []
        in_dim = d_model
        for i in range(num_layers - 1):
            layers += [nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            in_dim = hidden
        layers += [nn.LayerNorm(in_dim), nn.Linear(in_dim, num_candidates)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, video_feats: torch.Tensor, vid_mask: torch.Tensor) -> torch.Tensor:
        """
        video_feats: [B,T,D], vid_mask: [B,T] (1=valid,0=pad)
        return: logits [B,K]
        """
        v = self.pool(video_feats, attention_mask=vid_mask)  # [B,D]
        logits = self.mlp(v)  # [B,K]
        return logits


class CaptionModel_LenControl_Router(nn.Module):
    """
    组合：冻结的 CaptionModel_LenControl + 可训练的 LenRouter
    - backbone: 用于实际生成（推理/构造监督标签时）
    - router:   输入视频特征，输出候选长度的打分
    """
    def __init__(self,
                 # backbone (LenControl) 超参：
                 vocab_size: int = 49408, decoder_nhead: int = 8, d_model: int = 512, deocder_layer_nums: int = 3,
                 init_we: bool = True, init_lmhead: bool = True,
                 frozen_we: bool = True, frozen_lmhead: bool = False,
                 pad_token_id: int = 0, bos_token_id: int = 49406, eos_token_id: int = 49407,
                 len_min: int = 4, len_max: int = 76, num_types: int = 3,

                 # router 超参：
                 candidate_lengths: list = (8, 10, 12, 14),
                 router_hidden: int = 512, router_layers: int = 2, router_dropout: float = 0.1):
        super().__init__()
        # 1) backbone（LenControl）
        self.backbone = CaptionModel_LenControl(
            vocab_size=vocab_size, decoder_nhead=decoder_nhead, d_model=d_model, deocder_layer_nums=deocder_layer_nums,
            init_we=init_we, init_lmhead=init_lmhead, frozen_we=frozen_we, frozen_lmhead=frozen_lmhead,
            pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id,
            len_min=len_min, len_max=len_max, num_types=num_types
        )
        # 2) router
        self.candidate_lengths = list(candidate_lengths)
        self.router = LenRouter(d_model=d_model, hidden=router_hidden, num_layers=router_layers,
                                num_candidates=len(self.candidate_lengths), dropout=router_dropout)

        # 默认冻结 backbone（Router 训练只更新 router）
        for p in self.backbone.parameters():
            p.requires_grad = False

    # 便于外部加载 backbone 的权重
    def load_backbone_state_dict(self, state_dict, strict: bool = True):
        self.backbone.load_state_dict(state_dict, strict=strict)

    def forward(self, video_feats: torch.Tensor, vid_mask: torch.Tensor):
        """
        只做路由打分：
        return: logits [B,K]
        """
        return self.router(video_feats, vid_mask)

    @torch.no_grad()
    def pick_lengths(self, video_feats: torch.Tensor, vid_mask: torch.Tensor) -> torch.Tensor:
        """
        返回每条样本选中的"总 token 数"（含BOS/EOS），shape [B]，来自 router argmax→映射到候选长度。
        """
        logits = self.forward(video_feats, vid_mask)                 # [B,K]
        idx = torch.argmax(logits, dim=-1)                           # [B]
        lengths = torch.tensor([self.candidate_lengths[i] for i in idx.tolist()],
                               device=video_feats.device, dtype=torch.long)  # [B]
        return lengths








#-----------------------------------------------------------------


# ============================================================================
# Multi-Modal Embedding - 多模态嵌入模块
# ============================================================================
class MultiModalEmbedding(nn.Module):
    """
    多模态嵌入模块（Token Embedding + Modality Embedding）
    
    设计参考：
    - BERT: Token + Segment + Position Embedding
    - ViLT: Token + Modality Embedding
    - CLIP: Token + Position Embedding
    
    本模块：Token + Modality Embedding
    
    模态定义（9种）：
    0: VISION/OBJECT - 对象词（从Classifier识别的）
    1: CLS - 句子开始标记
    2: OBJ_CLS - 对象序列开始
    3: OBJECT - 对象及其修饰词（如 "a", "man", "in", "black"）
    4: OBJ_SEP - 对象分隔符
    5: OBJ_END - 对象序列结束
    6: SEP - 句子分隔/结束
    7: WORD - Caption中的常规词
    8: PADDING - Padding（不参与计算，embedding会被置零）
    
    特殊处理：
    - MASK (49408): 不添加模态embedding（保持中性）
    - PAD (0): 模态embedding置零（会被attention mask掉）
    
    使用示例：
    ```python
    # 完整序列（拼接）
    input_ids = [
        49406, 49409, 320, 786, 287, 1602, 49411, 320, 736, 1615, 49410, 49407,  # 对象序列
        320, 786, 476, 1870, 320, 1615, 49407,  # caption序列
        0, 0, 0, ...  # padding
    ]
    
    modality_ids = [
        1, 2, 3, 3, 3, 3, 4, 3, 3, 3, 5, 6,  # 对象序列模态
        7, 7, 7, 7, 7, 7, 6,  # caption序列模态
        8, 8, 8, ...  # padding模态
    ]
    
    # 嵌入
    embeddings = multi_modal_emb(input_ids, modality_ids)  # [B, L, 512]
    ```
    """
    
    def __init__(self, 
                 vocab_size=49413, 
                 d_model=512,
                 num_modalities=9,
                 dropout=0.1,
                 use_layernorm=True):
        """
        Args:
            vocab_size: 词表大小（扩展后的CLIP词表）
            d_model: embedding 维度
            num_modalities: 模态数量（默认9）
            dropout: dropout 比例
            use_layernorm: 是否使用 LayerNorm
        """
        super().__init__()
        
        # Token Embedding（可以加载预训练的CLIP embedding）
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Modality Embedding（从头训练）
        self.modality_embedding = nn.Embedding(num_modalities, d_model)
        
        # LayerNorm（标准做法）
        if use_layernorm:
            self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        else:
            self.LayerNorm = None
        
        # Dropout（防止过拟合）
        self.dropout = nn.Dropout(dropout)
        
        # 特殊 token IDs
        self.PAD_ID = 0
        self.MASK_ID = 49408
        
        # 配置
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_modalities = num_modalities
        
        print(f"[INFO] 初始化 MultiModalEmbedding")
        print(f"  vocab_size: {vocab_size}")
        print(f"  d_model: {d_model}")
        print(f"  num_modalities: {num_modalities}")
        print(f"  use_layernorm: {use_layernorm}")
        print(f"  dropout: {dropout}")
    
    def load_pretrained_token_embedding(self, pretrained_embedding):
        """
        加载预训练的 token embedding（如扩展后的CLIP embedding）
        
        Args:
            pretrained_embedding: nn.Embedding 或 torch.Tensor
        """
        if isinstance(pretrained_embedding, nn.Embedding):
            weight = pretrained_embedding.weight
        else:
            weight = pretrained_embedding
        
        assert weight.shape == self.token_embedding.weight.shape, \
            f"Shape mismatch: {weight.shape} vs {self.token_embedding.weight.shape}"
        
        with torch.no_grad():
            self.token_embedding.weight.copy_(weight)
        
        print(f"[INFO] 已加载预训练的 token embedding")
        print(f"  Shape: {weight.shape}")
    
    def freeze_token_embedding(self, freeze_old_tokens=True, old_vocab_size=49408):
        """
        冻结 token embedding
        
        Args:
            freeze_old_tokens: 是否冻结旧token（保留预训练知识）
            old_vocab_size: 旧词表大小（49408）
        """
        if freeze_old_tokens:
            # 冻结旧行，新行可训练
            def freeze_old_rows_hook(grad):
                if grad is None:
                    return grad
                grad[:old_vocab_size] = 0  # 冻结前49408行
                return grad
            
            self.token_embedding.weight.register_hook(freeze_old_rows_hook)
            print(f"[INFO] Token Embedding: 旧行(0-{old_vocab_size-1})冻结，新行可训练")
        else:
            # 全部冻结
            self.token_embedding.weight.requires_grad = False
            print(f"[INFO] Token Embedding: 全部冻结")
    
    def forward(self, input_ids, modality_ids):
        """
        前向传播（支持两种模式）
        
        模式1 - 整体加模态（每个位置不同模态）：
            Args:
                input_ids: [B, L] token ids
                modality_ids: [B, L] 或 [L] 每个位置的模态标识
            
            示例：
                input_ids:    [49406, 49409, 320, 786, ..., 49407, 0, 0]  [1, 77]
                modality_ids: [1,     2,     3,   3,   ..., 6,     8, 8]  [1, 77] 或 [77]
        
        模式2 - 单独加模态（整个序列同一模态）：
            Args:
                input_ids: [B, L] token ids
                modality_ids: int 或 [B] 标量模态ID
            
            示例：
                input_ids:    [320, 786, 287, 1602, ...]  [1, 5] - 短语 "a man in black"
                modality_ids: 3  (标量) - 整个序列都是模态3（OBJECT）
                
                或
                modality_ids: [3, 7]  [2] - batch中每个样本不同模态
        
        Returns:
            embeddings: [B, L, d_model] 多模态嵌入
        """
        B, L = input_ids.shape
        
        # 自动判断模式
        if isinstance(modality_ids, int):
            # 模式2a: 标量 - 整个batch同一模态
            modality_ids = torch.full((B, L), modality_ids, 
                                     dtype=torch.long, device=input_ids.device)
            mode = "单独加模态（标量）"
            
        elif modality_ids.dim() == 1:
            # 可能是两种情况
            if modality_ids.shape[0] == B:
                # 模式2b: [B] - 每个样本一个模态
                modality_ids = modality_ids.unsqueeze(1).expand(B, L)
                mode = "单独加模态（batch）"
            elif modality_ids.shape[0] == L:
                # 模式1: [L] - 广播到所有样本
                modality_ids = modality_ids.unsqueeze(0).expand(B, L)
                mode = "整体加模态（广播）"
            else:
                raise ValueError(f"modality_ids shape {modality_ids.shape} 无法匹配 input_ids {input_ids.shape}")
        
        elif modality_ids.dim() == 2:
            # 模式1: [B, L] - 每个位置都有模态
            assert modality_ids.shape == input_ids.shape, \
                f"modality_ids shape {modality_ids.shape} 必须匹配 input_ids {input_ids.shape}"
            mode = "整体加模态"
        
        else:
            raise ValueError(f"不支持的 modality_ids 维度: {modality_ids.dim()}")
        
        # Step 1: Token Embedding
        token_emb = self.token_embedding(input_ids)  # [B, L, d_model]
        
        # Step 2: Modality Embedding
        modality_emb = self.modality_embedding(modality_ids)  # [B, L, d_model]
        
        # Step 3: 特殊处理 - PAD 和 MASK 的模态embedding置零
        # PAD 位置
        pad_mask = (input_ids == self.PAD_ID).unsqueeze(-1)  # [B, L, 1]
        
        # MASK 位置（保持中性，不添加模态bias）
        mask_mask = (input_ids == self.MASK_ID).unsqueeze(-1)  # [B, L, 1]
        
        # 组合mask
        special_mask = pad_mask | mask_mask  # [B, L, 1]
        
        # 置零模态embedding
        modality_emb = modality_emb * (~special_mask).float()
        
        # Step 4: 组合
        embeddings = token_emb + modality_emb  # [B, L, d_model]
        
        # Step 5: LayerNorm（可选，推荐）
        if self.LayerNorm is not None:
            embeddings = self.LayerNorm(embeddings)
        
        # Step 6: Dropout
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
    def get_token_embedding_only(self, input_ids):
        """
        只返回 token embedding（不加模态）
        
        用途：某些场景只需要词嵌入
        """
        return self.token_embedding(input_ids)
    
    def get_modality_embedding_only(self, modality_ids):
        """
        只返回 modality embedding
        
        用途：分析不同模态的表示
        """
        return self.modality_embedding(modality_ids)


#-----------------------------------------------------------------


# ============================================================================
# CLIP Phrase Encoder - 使用 CLIP Text Encoder 编码带上下文的短语
# ============================================================================
class CLIPPhraseEncoder(nn.Module):
    """
    使用 CLIP Text Encoder 的第6层输出编码短语（短语级上下文）
    
    功能：
    - 加载本地 CLIP ViT-B/32 text encoder
    - 替换为扩展后的 word embedding（49413）
    - 冻结除新增 token 外的所有参数
    - 提取第6层的输出（短语级，局部上下文充分）
    
    与 CLIPSentenceEncoder 的区别：
    - CLIPPhraseEncoder: 第6层（短语级，局部上下文）⭐
    - CLIPSentenceEncoder: 第12层（句子级，全局语义）
    
    输入：
    - phrase_ids: [B, 77] tokenizer 后的 id 序列
      格式：[CLS, word1, word2, ..., SEP, PAD, PAD, ...]
    
    输出：
    - all_mask: [B, 77] 包括 CLS 和 SEP，PAD 位置为 0
    - embedding: [B, 77, 512] 第6层的输出（短语级特征）
    - phrase_mask: [B, 77] 不包括 CLS 和 SEP，只有实际短语为 1
    
    示例：
    输入: "a big car in red"
    序列: [CLS] a big car in red [SEP] [PAD] [PAD] ...
    
    all_mask:    [1, 1, 1, 1, 1, 1, 1, 0, 0, ...]  ← CLS到SEP都是1
    phrase_mask: [0, 1, 1, 1, 1, 1, 0, 0, 0, ...]  ← 只有实际短语是1
    """
    
    def __init__(self, 
                 layer_idx=6,  # 默认使用第6层（短语级）
                 freeze_encoder=True,
                 updated_tokenizer_dir='./models/clip_tokenizer/models--openai--clip-vit-base-patch32/updated_with_obj_tokens',
                 clip_weights_dir='./models/clip_models'):
        """
        Args:
            layer_idx: 使用 CLIP TE 的哪一层（默认6，范围1-12）
                      推荐6：短语级特征，局部上下文充分
                      与句子编码器(layer_idx=12)形成对比
            freeze_encoder: 是否冻结 CLIP encoder（除新增token外）
            updated_tokenizer_dir: 扩展后的 tokenizer 目录
            clip_weights_dir: CLIP 权重目录
        """
        super().__init__()
        
        print("=" * 80)
        print("[INFO] 初始化 CLIPPhraseEncoder")
        print("=" * 80)
        
        self.layer_idx = layer_idx
        self.d_model = 512  # ViT-B/32
        
        # Step 1: 加载 CLIP Text Encoder
        print(f"\n[Step 1] 加载 CLIP Text Encoder（ViT-B/32）")
        self.text_encoder = self._load_clip_text_encoder(clip_weights_dir)
        
        # Step 2: 替换为扩展后的 word embedding
        print(f"\n[Step 2] 替换 word embedding")
        self._replace_word_embedding(updated_tokenizer_dir)
        
        # Step 3: 冻结策略
        print(f"\n[Step 3] 设置冻结策略")
        if freeze_encoder:
            self._freeze_encoder_except_new_tokens()
        
        # Step 4: 配置
        print(f"\n[Step 4] 配置")
        print(f"  使用层: 第 {layer_idx} 层")
        print(f"  输出维度: {self.d_model}")
        
        # 兼容不同版本
        try:
            vocab_size = self.text_encoder.text_model.embeddings.token_embedding.num_embeddings
        except AttributeError:
            vocab_size = self.text_encoder.token_embedding.num_embeddings
        print(f"  词表大小: {vocab_size}")
        
        print("=" * 80)
    
    def _load_clip_text_encoder(self, clip_weights_dir):
        """加载 CLIP Text Encoder"""
        try:
            # 方法1: 尝试从 HuggingFace transformers 加载
            from transformers import CLIPTextModel
            
            text_encoder = CLIPTextModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                cache_dir="./models/clip_cache",
                local_files_only=True
            )
            print(f"  ✅ 从 HuggingFace 加载成功")
            return text_encoder
            
        except Exception as e:
            print(f"  ⚠️  HuggingFace 加载失败: {e}")
            print(f"  尝试从 OpenAI CLIP 提取...")
            
            # 方法2: 从 OpenAI CLIP 提取 text encoder
            import clip
            
            # 查找权重文件
            candidate_pt = ["ViT-B-32.pt", "ViT-B-16.pt"]
            ckpt = None
            for f in candidate_pt:
                p = os.path.join(clip_weights_dir, f)
                if os.path.isfile(p):
                    ckpt = p
                    break
            
            if ckpt is None:
                raise FileNotFoundError(f"未找到 CLIP 权重文件")
            
            print(f"  从 OpenAI CLIP 加载: {ckpt}")
            model, _ = clip.load("ViT-B/32", device="cpu", jit=False, 
                                download_root=clip_weights_dir)
            
            # OpenAI CLIP 需要保存整个 model（不只是 transformer）
            # 因为 positional_embedding 在 model 上
            print(f"  ✅ 从 OpenAI CLIP 提取成功")
            
            return model  # 返回整个 model，不只是 transformer
    
    def _replace_word_embedding(self, updated_tokenizer_dir):
        """替换为扩展后的 word embedding"""
        import clip
        
        # 加载扩展后的 embedding
        # 方法：从 OpenAI CLIP 加载并手动扩展
        print(f"  加载扩展后的 word embedding...")
        
        # 1. 加载原始 CLIP embedding
        clip_weights_dir = './models/clip_models'
        model, _ = clip.load("ViT-B/32", device="cpu", jit=False,
                            download_root=clip_weights_dir)
        old_emb = model.token_embedding
        
        # 2. 扩展到 49413
        new_emb = nn.Embedding(49413, 512)
        with torch.no_grad():
            new_emb.weight[:49408].copy_(old_emb.weight)
            # 新行已在之前的脚本中初始化，这里简单初始化
            mean = old_emb.weight.mean().item()
            std = old_emb.weight.std().item()
            new_emb.weight[49408:].normal_(mean=mean, std=std*0.5)
        
        # 3. 替换到 text encoder
        # 根据加载方式不同，路径可能不同
        is_hf = hasattr(self.text_encoder, 'text_model')
        
        if is_hf:
            # HuggingFace 版本
            self.text_encoder.text_model.embeddings.token_embedding = new_emb
            print(f"  ✅ 已替换 word embedding (HuggingFace)")
        else:
            # OpenAI CLIP 版本 - 直接替换 model.token_embedding
            self.text_encoder.token_embedding = new_emb
            print(f"  ✅ 已替换 word embedding (OpenAI CLIP)")
        
        print(f"  词表大小: 49408 → 49413")
    
    def _freeze_encoder_except_new_tokens(self):
        """冻结 encoder，但保留新增 token 的梯度"""
        # 解冻新增 token 的 embedding（49408-49412）
        # 注意：先解冻，再冻结其他，避免被覆盖
        is_hf = hasattr(self.text_encoder, 'text_model')
        
        if is_hf:
            # HuggingFace 版本
            word_emb = self.text_encoder.text_model.embeddings.token_embedding
        else:
            # OpenAI CLIP 版本
            word_emb = self.text_encoder.token_embedding
        
        # 先确保新增token可训练
        word_emb.weight.requires_grad = True
        
        # 然后冻结其他所有参数
        for name, param in self.text_encoder.named_parameters():
            # 跳过 word embedding（稍后单独处理）
            if 'token_embedding' not in name:
                param.requires_grad = False
        
        # 冻结旧的 token embedding 行（0-49407）
        # 使用 hook 方式
        def freeze_old_rows_hook(grad):
            if grad is None:
                return grad
            # 冻结前49408行
            grad[:49408] = 0
            return grad
        
        word_emb.weight.register_hook(freeze_old_rows_hook)
        
        print(f"  ✅ CLIP Encoder 已冻结")
        print(f"  ✅ 新增 token embedding (49408-49412) 可训练")
        
        # 统计参数
        total = sum(p.numel() for p in self.text_encoder.parameters())
        trainable = sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad)
        print(f"  总参数: {total/1e6:.2f}M")
        print(f"  可训练参数: {trainable/1e3:.2f}K")
        print(f"    (实际可更新: 新增5个token × 512维 = 2.56K)")
    
    def forward(self, phrase_ids):
        """
        编码短语，提取第8层的带上下文表示
        
        Args:
            phrase_ids: [B, 77] tokenizer 后的 id 序列
                       格式: [CLS, word1, word2, ..., SEP, PAD, PAD, ...]
        
        Returns:
            all_mask: [B, 77] 包括 CLS 和 SEP，PAD 位置为 0
            embedding: [B, 77, 512] 第8层的输出
            phrase_mask: [B, 77] 不包括 CLS 和 SEP，只有实际短语为 1
        
        示例：
            输入 phrase_ids: [49406, 320, 1124, 1615, 287, 1842, 49407, 0, 0, ...]
                            ([CLS]   a    big   car   in   red  [SEP] [PAD][PAD]...)
            
            输出 all_mask:    [1, 1, 1, 1, 1, 1, 1, 0, 0, ...]
            输出 phrase_mask: [0, 1, 1, 1, 1, 1, 0, 0, 0, ...]
            输出 embedding:   [B, 77, 512]
        """
        B, L = phrase_ids.shape
        assert L == 77, f"输入长度必须是77，当前为{L}"
        
        # Step 1: 生成 all_mask（CLS 到 SEP 都是 1）
        # PAD 的 id 是 0
        all_mask = (phrase_ids != 0).long()  # [B, 77]
        
        # Step 2: 通过 CLIP Text Encoder 编码
        # 判断使用哪种版本
        is_hf = hasattr(self.text_encoder, 'text_model')
        
        if is_hf:
            # HuggingFace 版本
            outputs = self.text_encoder(
                input_ids=phrase_ids,
                output_hidden_states=True,
                return_dict=True
            )
            # outputs.hidden_states: tuple of 13个元素
            # [0]: embedding 层输出
            # [1-12]: transformer 层输出
            layer_output = outputs.hidden_states[self.layer_idx]  # [B, 77, 512]
        else:
            # OpenAI CLIP 版本
            layer_output = self._forward_to_layer(phrase_ids, self.layer_idx)
        
        embedding = layer_output  # [B, 77, 512]
        
        # Step 3: 生成 phrase_mask（不包括 CLS 和 SEP）
        phrase_mask = all_mask.clone()
        
        # CLS 通常是 49406，SEP 通常是 49407
        CLS_ID = 49406
        SEP_ID = 49407
        
        # 将 CLS 和 SEP 的位置设为 0
        phrase_mask[phrase_ids == CLS_ID] = 0
        phrase_mask[phrase_ids == SEP_ID] = 0
        
        return all_mask, embedding, phrase_mask
    
    def _forward_to_layer(self, input_ids, target_layer):
        """
        OpenAI CLIP 的前向传播（到指定层）
        
        OpenAI CLIP 结构：
        - model.token_embedding
        - model.positional_embedding  
        - model.transformer.resblocks (12层)
        - model.ln_final
        """
        # Embedding
        x = self.text_encoder.token_embedding(input_ids)  # [B, 77, 512]
        x = x + self.text_encoder.positional_embedding  # 位置编码
        x = x.permute(1, 0, 2)  # [77, B, 512] - OpenAI CLIP使用seq-first
        
        # 通过指定数量的 transformer blocks
        for i in range(target_layer):
            x = self.text_encoder.transformer.resblocks[i](x)
        
        x = x.permute(1, 0, 2)  # [B, 77, 512] - 转回batch-first
        
        return x


#-----------------------------------------------------------------


# ============================================================================
# CLIP Sentence Encoder - 使用 CLIP Text Encoder 编码完整句子
# ============================================================================
class CLIPSentenceEncoder(nn.Module):
    """
    使用 CLIP Text Encoder 的最后一层（第12层）输出编码句子（全局语义）
    
    功能：
    - 加载本地 CLIP ViT-B/32 text encoder
    - 替换为扩展后的 word embedding（49413）
    - 冻结除新增 token 外的所有参数
    - 提取第12层的输出（最抽象、最全局的表示）
    
    与 CLIPPhraseEncoder 的区别：
    - CLIPPhraseEncoder: 第6层（短语级，局部上下文）
    - CLIPSentenceEncoder: 第12层（句子级，全局语义）⭐
    
    输入：
    - sentence_ids: [B, 77] tokenizer 后的 id 序列
      格式：[CLS, word1, word2, ..., SEP, PAD, PAD, ...]
    
    输出：
    - sentence_mask: [B, 77] 包括 CLS 和 SEP，PAD 位置为 0
    - embedding: [B, 77, 512] 第12层的输出（最终层，全局语义）
    - sentence_words_mask: [B, 77] 不包括 CLS 和 SEP，只有实际单词为 1
    
    示例：
    输入: "a man is driving a car"
    序列: [CLS] a man is driving a car [SEP] [PAD] [PAD] ...
    
    sentence_mask:       [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, ...]  ← CLS到SEP都是1
    sentence_words_mask: [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, ...]  ← 只有实际单词
    
    层选择对比：
    - CLIPPhraseEncoder(layer_idx=6): 短语级，中层特征
    - CLIPSentenceEncoder(layer_idx=12): 句子级，深层特征
    - 适合不同的任务复杂度
    """
    
    def __init__(self, 
                 layer_idx=12,  # 默认使用最后一层
                 freeze_encoder=True,
                 updated_tokenizer_dir='./models/clip_tokenizer/models--openai--clip-vit-base-patch32/updated_with_obj_tokens',
                 clip_weights_dir='./models/clip_models'):
        """
        Args:
            layer_idx: 使用 CLIP TE 的哪一层（默认12=最后一层，范围1-12）
                      推荐12：全局语义，适合完整句子
                      与短语编码器(layer_idx=6)形成对比
            freeze_encoder: 是否冻结 CLIP encoder（除新增token外）
            updated_tokenizer_dir: 扩展后的 tokenizer 目录
            clip_weights_dir: CLIP 权重目录
        """
        super().__init__()
        
        print("=" * 80)
        print("[INFO] 初始化 CLIPSentenceEncoder")
        print("=" * 80)
        print(f"  设计: 使用第{layer_idx}层（句子级全局语义）")
        
        self.layer_idx = layer_idx
        self.d_model = 512  # ViT-B/32
        
        # Step 1: 加载 CLIP Text Encoder
        print(f"\n[Step 1] 加载 CLIP Text Encoder（ViT-B/32）")
        self.text_encoder = self._load_clip_text_encoder(clip_weights_dir)
        
        # Step 2: 替换为扩展后的 word embedding
        print(f"\n[Step 2] 替换 word embedding")
        self._replace_word_embedding(updated_tokenizer_dir)
        
        # Step 3: 冻结策略
        print(f"\n[Step 3] 设置冻结策略")
        if freeze_encoder:
            self._freeze_encoder_except_new_tokens()
        
        # Step 4: 配置
        print(f"\n[Step 4] 配置")
        print(f"  使用层: 第 {layer_idx} 层")
        print(f"  输出维度: {self.d_model}")
        
        # 兼容不同版本
        try:
            vocab_size = self.text_encoder.text_model.embeddings.token_embedding.num_embeddings
        except AttributeError:
            vocab_size = self.text_encoder.token_embedding.num_embeddings
        print(f"  词表大小: {vocab_size}")
        
        print("=" * 80)
    
    def _load_clip_text_encoder(self, clip_weights_dir):
        """加载 CLIP Text Encoder"""
        try:
            # 方法1: 尝试从 HuggingFace transformers 加载
            from transformers import CLIPTextModel
            
            text_encoder = CLIPTextModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                cache_dir="./models/clip_cache",
                local_files_only=True
            )
            print(f"  ✅ 从 HuggingFace 加载成功")
            return text_encoder
            
        except Exception as e:
            print(f"  ⚠️  HuggingFace 加载失败: {e}")
            print(f"  尝试从 OpenAI CLIP 提取...")
            
            # 方法2: 从 OpenAI CLIP 提取 text encoder
            import clip
            
            # 查找权重文件
            candidate_pt = ["ViT-B-32.pt", "ViT-B-16.pt"]
            ckpt = None
            for f in candidate_pt:
                p = os.path.join(clip_weights_dir, f)
                if os.path.isfile(p):
                    ckpt = p
                    break
            
            if ckpt is None:
                raise FileNotFoundError(f"未找到 CLIP 权重文件")
            
            print(f"  从 OpenAI CLIP 加载: {ckpt}")
            model, _ = clip.load("ViT-B/32", device="cpu", jit=False, 
                                download_root=clip_weights_dir)
            
            # OpenAI CLIP 需要保存整个 model（不只是 transformer）
            # 因为 positional_embedding 在 model 上
            print(f"  ✅ 从 OpenAI CLIP 提取成功")
            
            return model  # 返回整个 model，不只是 transformer
    
    def _replace_word_embedding(self, updated_tokenizer_dir):
        """替换为扩展后的 word embedding"""
        import clip
        
        # 加载扩展后的 embedding
        print(f"  加载扩展后的 word embedding...")
        
        # 1. 加载原始 CLIP embedding
        clip_weights_dir = './models/clip_models'
        model, _ = clip.load("ViT-B/32", device="cpu", jit=False,
                            download_root=clip_weights_dir)
        old_emb = model.token_embedding
        
        # 2. 扩展到 49413
        new_emb = nn.Embedding(49413, 512)
        with torch.no_grad():
            new_emb.weight[:49408].copy_(old_emb.weight)
            # 新行已在之前的脚本中初始化，这里简单初始化
            mean = old_emb.weight.mean().item()
            std = old_emb.weight.std().item()
            new_emb.weight[49408:].normal_(mean=mean, std=std*0.5)
        
        # 3. 替换到 text encoder
        is_hf = hasattr(self.text_encoder, 'text_model')
        
        if is_hf:
            # HuggingFace 版本
            self.text_encoder.text_model.embeddings.token_embedding = new_emb
            print(f"  ✅ 已替换 word embedding (HuggingFace)")
        else:
            # OpenAI CLIP 版本 - 直接替换 model.token_embedding
            self.text_encoder.token_embedding = new_emb
            print(f"  ✅ 已替换 word embedding (OpenAI CLIP)")
        
        print(f"  词表大小: 49408 → 49413")
    
    def _freeze_encoder_except_new_tokens(self):
        """冻结 encoder，但保留新增 token 的梯度"""
        is_hf = hasattr(self.text_encoder, 'text_model')
        
        if is_hf:
            # HuggingFace 版本
            word_emb = self.text_encoder.text_model.embeddings.token_embedding
        else:
            # OpenAI CLIP 版本
            word_emb = self.text_encoder.token_embedding
        
        # 先确保新增token可训练
        word_emb.weight.requires_grad = True
        
        # 然后冻结其他所有参数
        for name, param in self.text_encoder.named_parameters():
            # 跳过 word embedding（稍后单独处理）
            if 'token_embedding' not in name:
                param.requires_grad = False
        
        # 冻结旧的 token embedding 行（0-49407）
        # 使用 hook 方式
        def freeze_old_rows_hook(grad):
            if grad is None:
                return grad
            # 冻结前49408行
            grad[:49408] = 0
            return grad
        
        word_emb.weight.register_hook(freeze_old_rows_hook)
        
        print(f"  ✅ CLIP Encoder 已冻结")
        print(f"  ✅ 新增 token embedding (49408-49412) 可训练")
        
        # 统计参数
        total = sum(p.numel() for p in self.text_encoder.parameters())
        trainable = sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad)
        print(f"  总参数: {total/1e6:.2f}M")
        print(f"  可训练参数: {trainable/1e3:.2f}K")
        print(f"    (实际可更新: 新增5个token × 512维 = 2.56K)")
    
    def forward(self, sentence_ids):
        """
        编码句子，提取第8层的带上下文表示
        
        Args:
            sentence_ids: [B, 77] tokenizer 后的 id 序列
                         格式: [CLS, word1, word2, ..., SEP, PAD, PAD, ...]
        
        Returns:
            sentence_mask: [B, 77] 包括 CLS 和 SEP，PAD 位置为 0
            embedding: [B, 77, 512] 第8层的输出
            sentence_words_mask: [B, 77] 不包括 CLS 和 SEP，只有实际单词为 1
        
        示例：
            输入 sentence_ids: [49406, 320, 786, 476, 1870, 320, 1615, 49407, 0, 0, ...]
                              ([CLS]   a    man  is   driving  a   car   [SEP] [PAD][PAD]...)
            
            输出 sentence_mask:       [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, ...]
            输出 sentence_words_mask: [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, ...]
            输出 embedding:           [B, 77, 512]
        """
        B, L = sentence_ids.shape
        assert L == 77, f"输入长度必须是77，当前为{L}"
        
        # Step 1: 生成 sentence_mask（CLS 到 SEP 都是 1）
        # PAD 的 id 是 0
        sentence_mask = (sentence_ids != 0).long()  # [B, 77]
        
        # Step 2: 通过 CLIP Text Encoder 编码
        # 判断使用哪种版本
        is_hf = hasattr(self.text_encoder, 'text_model')
        
        if is_hf:
            # HuggingFace 版本
            outputs = self.text_encoder(
                input_ids=sentence_ids,
                output_hidden_states=True,
                return_dict=True
            )
            # outputs.hidden_states: tuple of 13个元素
            # [0]: embedding 层输出
            # [1-12]: transformer 层输出
            layer_output = outputs.hidden_states[self.layer_idx]  # [B, 77, 512]
        else:
            # OpenAI CLIP 版本
            layer_output = self._forward_to_layer(sentence_ids, self.layer_idx)
        
        embedding = layer_output  # [B, 77, 512]
        
        # Step 3: 生成 sentence_words_mask（不包括 CLS 和 SEP）
        sentence_words_mask = sentence_mask.clone()
        
        # CLS 通常是 49406，SEP 通常是 49407
        CLS_ID = 49406
        SEP_ID = 49407
        
        # 将 CLS 和 SEP 的位置设为 0
        sentence_words_mask[sentence_ids == CLS_ID] = 0
        sentence_words_mask[sentence_ids == SEP_ID] = 0
        
        return sentence_mask, embedding, sentence_words_mask
    
    def _forward_to_layer(self, input_ids, target_layer):
        """
        OpenAI CLIP 的前向传播（到指定层）
        
        OpenAI CLIP 结构：
        - model.token_embedding
        - model.positional_embedding  
        - model.transformer.resblocks (12层)
        - model.ln_final
        """
        # Embedding
        x = self.text_encoder.token_embedding(input_ids)  # [B, 77, 512]
        x = x + self.text_encoder.positional_embedding  # 位置编码
        x = x.permute(1, 0, 2)  # [77, B, 512] - OpenAI CLIP使用seq-first
        
        # 通过指定数量的 transformer blocks
        for i in range(target_layer):
            x = self.text_encoder.transformer.resblocks[i](x)
        
        x = x.permute(1, 0, 2)  # [B, 77, 512] - 转回batch-first
        
        return x


#-----------------------------------------------------------------


# if __name__ == "__main__":
    pass
    # test_positional_encodings()
    # test_sequence_mean_pooling()
    
    # 测试 MultiModalEmbedding
    def test_multi_modal_embedding():
        """测试多模态嵌入模块"""
        print("\n" + "=" * 80)
        print("测试 MultiModalEmbedding")
        print("=" * 80)
        
        # 创建模块
        multi_modal_emb = MultiModalEmbedding(
            vocab_size=49413,
            d_model=512,
            num_modalities=9,
            dropout=0.1,
            use_layernorm=True
        )
        
        # 测试序列（拼接：对象序列 + caption序列 + padding）
        # [CLS][OBJ_CLS] a man in black [OBJ_SEP] a red car [OBJ_END][SEP] a man is driving [SEP][PAD][PAD]
        input_ids = torch.tensor([
            [49406, 49409, 320, 786, 287, 1602, 49411, 320, 736, 1615, 49410, 49407,  # 对象序列
             320, 786, 476, 1870, 49407,  # caption序列
             0, 0, 0, 0, 0]  # padding
        ])  # [1, 22]
        
        modality_ids = torch.tensor([
            [1, 2, 3, 3, 3, 3, 4, 3, 3, 3, 5, 6,  # 对象序列模态
             7, 7, 7, 7, 6,  # caption序列模态
             8, 8, 8, 8, 8]  # padding模态
        ])  # [1, 22]
        
        print(f"\n输入:")
        print(f"  input_ids shape: {input_ids.shape}")
        print(f"  modality_ids shape: {modality_ids.shape}")
        print(f"  input_ids: {input_ids[0].tolist()}")
        print(f"  modality_ids: {modality_ids[0].tolist()}")
        
        # 前向传播
        embeddings = multi_modal_emb(input_ids, modality_ids)
        
        print(f"\n输出:")
        print(f"  embeddings shape: {embeddings.shape}")
        print(f"  embeddings dtype: {embeddings.dtype}")
        print(f"  embeddings mean: {embeddings.mean().item():.6f}")
        print(f"  embeddings std: {embeddings.std().item():.6f}")
        
        # 验证特殊位置（检查modality embedding是否真的被置零）
        print(f"\n验证模态embedding是否被正确置零:")
        
        # 直接检查 modality_emb（置零前后）
        with torch.no_grad():
            token_emb = multi_modal_emb.token_embedding(input_ids)
            modality_emb_raw = multi_modal_emb.modality_embedding(modality_ids)
            
            # 检查PAD位置的原始modality embedding
            pad_pos = 17
            pad_modality_before = modality_emb_raw[0, pad_pos].norm().item()
            print(f"  PAD[{pad_pos}] 置零前的模态embedding范数: {pad_modality_before:.6f}")
            
            # 通过forward获取置零后的
            # 需要手动计算置零后的modality_emb
            pad_mask = (input_ids == 0).unsqueeze(-1)
            mask_mask = (input_ids == 49408).unsqueeze(-1)
            special_mask = pad_mask | mask_mask
            
            modality_emb_zeroed = modality_emb_raw * (~special_mask).float()
            pad_modality_after = modality_emb_zeroed[0, pad_pos].norm().item()
            print(f"  PAD[{pad_pos}] 置零后的模态embedding范数: {pad_modality_after:.6f} (应=0)")
            
            # 检查正常位置
            normal_pos = 2
            normal_modality = modality_emb_zeroed[0, normal_pos].norm().item()
            print(f"  正常token[{normal_pos}] 模态embedding范数: {normal_modality:.6f} (应>0)")
            
            # 检查最终embedding（token + modality，置零后）
            final_check = (token_emb + modality_emb_zeroed)
            
            # PAD位置应该只有token embedding
            pad_should_be_token = (final_check[0, pad_pos] - token_emb[0, pad_pos]).norm().item()
            print(f"\n  验证PAD[{pad_pos}]最终只有token_emb: {pad_should_be_token:.6f} (应≈0)")
            
            if pad_should_be_token < 1e-5:
                print(f"  ✅ PAD的模态embedding已正确置零")
            else:
                print(f"  ❌ PAD的模态embedding未正确置零")
        
        # 测试 MASK
        print(f"\n测试 MASK token 置零:")
        mask_input = torch.tensor([[49406, 320, 786, 49408, 1615, 49407, 0, 0]])
        mask_modality = torch.tensor([[1, 7, 7, 7, 7, 6, 8, 8]])
        
        with torch.no_grad():
            mask_token_emb = multi_modal_emb.token_embedding(mask_input)
            mask_modality_raw = multi_modal_emb.modality_embedding(mask_modality)
            
            mask_pos = 3
            mask_modality_before = mask_modality_raw[0, mask_pos].norm().item()
            print(f"  MASK[{mask_pos}] 置零前的模态embedding范数: {mask_modality_before:.6f}")
            
            # 置零后
            mask_positions = (mask_input == 49408).unsqueeze(-1)
            mask_modality_zeroed = mask_modality_raw * (~mask_positions).float()
            mask_modality_after = mask_modality_zeroed[0, mask_pos].norm().item()
            print(f"  MASK[{mask_pos}] 置零后的模态embedding范数: {mask_modality_after:.6f} (应=0)")
            
            if mask_modality_after < 1e-5:
                print(f"  ✅ MASK的模态embedding已正确置零")
            else:
                print(f"  ❌ MASK的模态embedding未正确置零")
        
        # 测试不同输入模式
        print(f"\n" + "=" * 80)
        print("测试不同输入模式")
        print("=" * 80)
        
        # 模式1: 整体加模态（每个位置不同）[B, L]
        print(f"\n[模式1] 整体加模态 - modality_ids: [B, L]")
        input1 = torch.tensor([[320, 786, 287, 1602, 49407, 0, 0]])  # [1, 7]
        modality1 = torch.tensor([[3, 3, 3, 3, 6, 8, 8]])            # [1, 7]
        emb1 = multi_modal_emb(input1, modality1)
        print(f"  input shape: {input1.shape}")
        print(f"  modality shape: {modality1.shape}")
        print(f"  output shape: {emb1.shape}")
        
        # 模式2a: 单独加模态 - 标量（整个batch同一模态）
        print(f"\n[模式2a] 单独加模态 - modality_ids: int")
        input2 = torch.tensor([[320, 786, 287, 1602]])  # [1, 4] - 短语 "a man in black"
        modality2 = 3  # 标量 - 整个序列都是模态3（OBJECT）
        emb2 = multi_modal_emb(input2, modality2)
        print(f"  input shape: {input2.shape}")
        print(f"  modality: {modality2} (标量)")
        print(f"  output shape: {emb2.shape}")
        print(f"  说明: 整个序列 [320, 786, 287, 1602] 都加模态3")
        
        # 模式2b: 单独加模态 - batch中每个样本不同模态
        print(f"\n[模式2b] 单独加模态 - modality_ids: [B]")
        input3 = torch.tensor([
            [320, 786, 287, 1602],     # 样本1: a man in black
            [320, 736, 1615, 0]        # 样本2: a red car
        ])  # [2, 4]
        modality3 = torch.tensor([3, 3])  # [2] - 两个样本都是模态3
        emb3 = multi_modal_emb(input3, modality3)
        print(f"  input shape: {input3.shape}")
        print(f"  modality shape: {modality3.shape}")
        print(f"  output shape: {emb3.shape}")
        print(f"  说明: 每个样本整体加对应的模态")
        
        # 模式1变体: [L] 广播
        print(f"\n[模式1变体] 整体加模态 - modality_ids: [L] 广播")
        input4 = torch.tensor([
            [320, 786, 287, 1602],
            [320, 736, 1615, 0]
        ])  # [2, 4]
        modality4 = torch.tensor([3, 3, 3, 8])  # [4] - 每个位置的模态（会广播到batch）
        emb4 = multi_modal_emb(input4, modality4)
        print(f"  input shape: {input4.shape}")
        print(f"  modality shape: {modality4.shape}")
        print(f"  output shape: {emb4.shape}")
        print(f"  说明: 模态序列广播到所有样本")
        
        print("\n" + "=" * 80)
        print("✅ 所有模式测试完成！")
        print("=" * 80)
    
    # 测试 CLIPPhraseEncoder
    def test_clip_phrase_encoder():
        """测试 CLIP Phrase Encoder"""
        print("\n" + "=" * 80)
        print("测试 CLIPPhraseEncoder")
        print("=" * 80)
        
        # 加载 tokenizer
        try:
            from transformers import CLIPTokenizerFast as CLIPTokenizer
        except:
            from transformers import CLIPTokenizer
        
        tokenizer = CLIPTokenizer.from_pretrained(
            "./models/clip_tokenizer/models--openai--clip-vit-base-patch32/updated_with_obj_tokens",
            local_files_only=True
        )
        
        print(f"Tokenizer 词表大小: {len(tokenizer)}")
        
        # 修改 padding 为使用 id=0（而不是49407）
        print(f"原始 pad_token_id: {tokenizer.pad_token_id}")
        
        # 方法：在 tokenize 时手动替换
        # 或者创建一个包装函数
        
        # 创建 encoder
        encoder = CLIPPhraseEncoder(
            layer_idx=8,
            freeze_encoder=True
        )
        
        # 测试短语
        test_phrases = [
            "a big red car",
            "a man in black",
            "an athlete"
        ]
        
        print(f"\n测试短语:")
        for phrase in test_phrases:
            print(f"  - {phrase}")
        
        # Tokenize（添加 CLS, SEP, PAD）
        encoded = tokenizer(
            test_phrases,
            padding='max_length',
            max_length=77,
            truncation=True,
            return_tensors='pt'
        )
        
        phrase_ids = encoded['input_ids']  # [3, 77]
        
        # 将 padding id 从 49407 替换为 0
        # 保留第一个 49407（SEP），后续的都替换
        print(f"\n替换 padding id: 49407 → 0")
        for i in range(phrase_ids.shape[0]):
            # 找到第一个SEP的位置
            sep_positions = (phrase_ids[i] == 49407).nonzero(as_tuple=True)[0]
            if len(sep_positions) > 0:
                first_sep = sep_positions[0].item()
                # 只替换第一个SEP之后的49407
                phrase_ids[i, first_sep+1:][phrase_ids[i, first_sep+1:] == 49407] = 0
        
        print(f"输入 phrase_ids shape: {phrase_ids.shape}")
        
        # 显示第一个短语的 token 序列
        print(f"\n第一个短语的 token 序列:")
        tokens = tokenizer.convert_ids_to_tokens(phrase_ids[0].tolist())
        print(f"  Token IDs: {phrase_ids[0][:15].tolist()}...")
        print(f"  Tokens: {tokens[:15]}...")
        
        # 前向传播
        with torch.no_grad():
            all_mask, embedding, phrase_mask = encoder(phrase_ids)
        
        print(f"\n输出:")
        print(f"  all_mask shape: {all_mask.shape}")
        print(f"  embedding shape: {embedding.shape}")
        print(f"  phrase_mask shape: {phrase_mask.shape}")
        
        print(f"\n第一个短语的 mask:")
        print(f"  all_mask:    {all_mask[0][:15].tolist()}...")
        print(f"  phrase_mask: {phrase_mask[0][:15].tolist()}...")
        
        # 统计
        print(f"\n统计:")
        for i, phrase in enumerate(test_phrases):
            all_1s = all_mask[i].sum().item()
            phrase_1s = phrase_mask[i].sum().item()
            print(f"  '{phrase}':")
            print(f"    all_mask 中 1 的数量: {all_1s} (包括 CLS+SEP)")
            print(f"    phrase_mask 中 1 的数量: {phrase_1s} (纯短语)")
        
        print("\n" + "=" * 80)
        print("✅ 测试完成！")
        print("=" * 80)
    
    # 取消下面的注释来测试
    # test_multi_modal_embedding()  # 测试多模态嵌入
    # test_clip_phrase_encoder()      # 测试短语编码器


# =========================================================
# Cross-Attention Block & Module
# =========================================================
class CrossAttentionBlock(nn.Module):
    """
    单个交叉注意力块（Cross-Attention Block）
    
    功能：
    - Query (Q) 从可学习的查询向量得到
    - Key (K) 和 Value (V) 从视频帧特征得到
    - 使用 Multi-Head Attention 进行交叉注意力计算
    - 包含残差连接和 LayerNorm
    - 包含 Feed-Forward Network (FFN)
    
    架构（标准 Transformer Decoder Cross-Attention）：
        Input Q: [B, Q_nums, d_model]
        Input vid_feats: [B, T, d_model]
        
        1. LayerNorm(Q)
        2. Cross-Attention: Q 查询 vid_feats (K, V)
        3. Residual: Q = Q + Attention_output
        4. LayerNorm(Q)
        5. FFN(Q)
        6. Residual: Q = Q + FFN_output
        
        Output Q: [B, Q_nums, d_model]
    
    参数：
        d_model: 特征维度
        num_heads: 多头注意力的头数
        dim_feedforward: FFN 的隐藏层维度
        dropout: dropout 概率
        activation: FFN 的激活函数
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Multi-Head Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # [B, L, D] 格式
        )
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,           # [B, Q_nums, d_model]
        vid_feats: torch.Tensor,       # [B, T, d_model]
        vid_mask: Optional[torch.Tensor] = None  # [B, T]
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            query: [B, Q_nums, d_model] 查询向量
            vid_feats: [B, T, d_model] 视频帧特征（作为 Key 和 Value）
            vid_mask: [B, T] 视频帧的 mask（True 表示有效位置，False 表示 padding）
        
        Returns:
            output: [B, Q_nums, d_model] 精炼后的查询向量
        """
        
        # 处理 vid_mask
        # PyTorch MultiheadAttention 的 key_padding_mask:
        # True 表示需要被忽略的位置（padding），False 表示有效位置
        # 我们的 vid_mask: True 表示有效位置
        # 所以需要取反
        if vid_mask is not None:
            # 转换为 bool 类型（如果需要）
            if vid_mask.dtype != torch.bool:
                vid_mask = vid_mask.bool()
            # vid_mask: [B, T], True 表示有效
            # key_padding_mask: [B, T], True 表示 padding（需要忽略）
            key_padding_mask = ~vid_mask  # 取反
        else:
            key_padding_mask = None
        
        # 1. Cross-Attention with Residual Connection
        # Pre-LN: 先 LayerNorm，再 Attention
        query_norm = self.norm1(query)
        
        # MultiheadAttention: query 查询 vid_feats
        # query: [B, Q_nums, d_model]
        # key/value: [B, T, d_model]
        attn_output, attn_weights = self.cross_attn(
            query=query_norm,
            key=vid_feats,
            value=vid_feats,
            key_padding_mask=key_padding_mask,  # [B, T]
            need_weights=False  # 不需要返回注意力权重（加速）
        )
        
        # Residual connection
        query = query + self.dropout(attn_output)
        
        # 2. Feed-Forward Network with Residual Connection
        # Pre-LN: 先 LayerNorm，再 FFN
        query_norm = self.norm2(query)
        ffn_output = self.ffn(query_norm)
        
        # Residual connection
        query = query + ffn_output
        
        return query


class CrossAttentionModule(nn.Module):
    """
    交叉注意力模块（Cross-Attention Module）
    
    功能：
    - 包含多个堆叠的 CrossAttentionBlock（默认3个）
    - Query 在每一层都被精炼，并且每层都去查询原始视频特征
    - 最终输出精炼后的 Query
    
    设计理念：
    - Query 逐层精炼：每层的输出作为下一层的输入
    - 持续查询原图：每层都去查询原始 vid_feats（而不是上一层的输出）
    - 类似于 Deformable DETR 中的 Decoder 设计
    
    架构：
        Input:
            Q_0: [B, Q_nums, d_model]  初始查询
            vid_feats: [B, T, d_model]  视频特征
            vid_mask: [B, T]            视频 mask
        
        Layer 1: Q_1 = CrossAttentionBlock(Q_0, vid_feats, vid_mask)
        Layer 2: Q_2 = CrossAttentionBlock(Q_1, vid_feats, vid_mask)
        Layer 3: Q_3 = CrossAttentionBlock(Q_2, vid_feats, vid_mask)
        
        Output: Q_3: [B, Q_nums, d_model]
    
    参数：
        d_model: 特征维度
        num_heads: 多头注意力的头数
        dim_feedforward: FFN 的隐藏层维度
        num_layers: CrossAttentionBlock 的层数（默认3）
        dropout: dropout 概率
        activation: FFN 的激活函数
    
    使用示例：
        >>> cross_attn_module = CrossAttentionModule(
        ...     d_model=512,
        ...     num_heads=8,
        ...     num_layers=3
        ... )
        >>> 
        >>> # 输入
        >>> queries = torch.randn(2, 10, 512)       # [B, Q_nums, d_model]
        >>> vid_feats = torch.randn(2, 20, 512)     # [B, T, d_model]
        >>> vid_mask = torch.ones(2, 20).bool()     # [B, T]
        >>> 
        >>> # 前向传播
        >>> refined_queries = cross_attn_module(queries, vid_feats, vid_mask)
        >>> print(refined_queries.shape)  # [2, 10, 512]
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # 堆叠多个 CrossAttentionBlock
        self.layers = nn.ModuleList([
            CrossAttentionBlock(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation
            )
            for _ in range(num_layers)
        ])
        
        # 最后的 LayerNorm（可选，标准做法）
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        queries: torch.Tensor,         # [B, Q_nums, d_model]
        vid_feats: torch.Tensor,       # [B, T, d_model]
        vid_mask: Optional[torch.Tensor] = None  # [B, T]
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            queries: [B, Q_nums, d_model] 初始查询向量
            vid_feats: [B, T, d_model] 视频帧特征
            vid_mask: [B, T] 视频帧的 mask（True 表示有效位置）
        
        Returns:
            output: [B, Q_nums, d_model] 精炼后的查询向量
        """
        # 转换 mask 为 bool 类型（如果需要）
        if vid_mask is not None and vid_mask.dtype != torch.bool:
            vid_mask = vid_mask.bool()
        
        # 逐层精炼 Query，每层都去查询原始 vid_feats
        output = queries
        for i, layer in enumerate(self.layers):
            output = layer(
                query=output,         # 上一层的输出
                vid_feats=vid_feats,  # 原始视频特征（每层都查询）
                vid_mask=vid_mask
            )
        
        # 最后的 LayerNorm
        output = self.final_norm(output)
        
        return output


# =========================================================
# 测试代码（可选）
# =========================================================
if __name__ == "__main__":
    """
    测试 CrossAttentionBlock 和 CrossAttentionModule
    """
    
    def test_cross_attention_block():
        """测试单个 CrossAttentionBlock"""
        print("\n" + "=" * 80)
        print("测试 CrossAttentionBlock")
        print("=" * 80)
        
        # 参数
        B, Q_nums, T, d_model = 2, 10, 20, 512
        
        # 创建模块
        block = CrossAttentionBlock(
            d_model=d_model,
            num_heads=8,
            dim_feedforward=2048,
            dropout=0.1
        )
        
        # 输入数据
        queries = torch.randn(B, Q_nums, d_model)
        vid_feats = torch.randn(B, T, d_model)
        vid_mask = torch.ones(B, T).bool()
        # 模拟部分 padding
        vid_mask[0, 15:] = False
        vid_mask[1, 18:] = False
        
        print(f"\n输入:")
        print(f"  queries shape: {queries.shape}")
        print(f"  vid_feats shape: {vid_feats.shape}")
        print(f"  vid_mask shape: {vid_mask.shape}")
        print(f"  vid_mask[0]: {vid_mask[0].sum().item()}/{T} 有效帧")
        print(f"  vid_mask[1]: {vid_mask[1].sum().item()}/{T} 有效帧")
        
        # 前向传播
        with torch.no_grad():
            output = block(queries, vid_feats, vid_mask)
        
        print(f"\n输出:")
        print(f"  output shape: {output.shape}")
        print(f"  output mean: {output.mean().item():.6f}")
        print(f"  output std: {output.std().item():.6f}")
        
        print("\n✅ CrossAttentionBlock 测试通过！")
    
    def test_cross_attention_module():
        """测试 CrossAttentionModule"""
        print("\n" + "=" * 80)
        print("测试 CrossAttentionModule")
        print("=" * 80)
        
        # 参数
        B, Q_nums, T, d_model = 2, 10, 20, 512
        num_layers = 3
        
        # 创建模块
        module = CrossAttentionModule(
            d_model=d_model,
            num_heads=8,
            dim_feedforward=2048,
            num_layers=num_layers,
            dropout=0.1
        )
        
        # 输入数据
        queries = torch.randn(B, Q_nums, d_model)
        vid_feats = torch.randn(B, T, d_model)
        vid_mask = torch.ones(B, T).bool()
        # 模拟部分 padding
        vid_mask[0, 15:] = False
        vid_mask[1, 18:] = False
        
        print(f"\n配置:")
        print(f"  d_model: {d_model}")
        print(f"  num_heads: 8")
        print(f"  num_layers: {num_layers}")
        
        print(f"\n输入:")
        print(f"  queries shape: {queries.shape}")
        print(f"  vid_feats shape: {vid_feats.shape}")
        print(f"  vid_mask shape: {vid_mask.shape}")
        
        # 前向传播
        with torch.no_grad():
            output = module(queries, vid_feats, vid_mask)
        
        print(f"\n输出:")
        print(f"  output shape: {output.shape}")
        print(f"  output mean: {output.mean().item():.6f}")
        print(f"  output std: {output.std().item():.6f}")
        
        # 检查梯度流（训练模式）
        print(f"\n检查梯度流:")
        module.train()
        queries_grad = torch.randn(B, Q_nums, d_model, requires_grad=True)
        output_grad = module(queries_grad, vid_feats, vid_mask)
        loss = output_grad.sum()
        loss.backward()
        print(f"  queries 梯度范数: {queries_grad.grad.norm().item():.6f}")
        print(f"  ✅ 梯度流正常")
        
        print("\n✅ CrossAttentionModule 测试通过！")
    
    def test_different_shapes():
        """测试不同的输入形状"""
        print("\n" + "=" * 80)
        print("测试不同的输入形状")
        print("=" * 80)
        
        module = CrossAttentionModule(
            d_model=512,
            num_heads=8,
            num_layers=3
        )
        
        test_cases = [
            (1, 5, 10),   # B=1, Q_nums=5, T=10
            (4, 20, 30),  # B=4, Q_nums=20, T=30
            (8, 1, 50),   # B=8, Q_nums=1, T=50
        ]
        
        for i, (B, Q_nums, T) in enumerate(test_cases):
            print(f"\n测试案例 {i+1}: B={B}, Q_nums={Q_nums}, T={T}")
            
            queries = torch.randn(B, Q_nums, 512)
            vid_feats = torch.randn(B, T, 512)
            vid_mask = torch.ones(B, T).bool()
            
            with torch.no_grad():
                output = module(queries, vid_feats, vid_mask)
            
            print(f"  输入: queries {queries.shape}, vid_feats {vid_feats.shape}")
            print(f"  输出: {output.shape}")
            assert output.shape == (B, Q_nums, 512), f"形状不匹配！"
            print(f"  ✅ 通过")
        
        print("\n✅ 所有形状测试通过！")
    
    # 运行测试
    # 取消注释以运行测试
# test_cross_attention_block()
    # test_cross_attention_module()
    # test_different_shapes()


class CaptionModel_Base_SharedTemporal(nn.Module):
    """
    Baseline-compatible caption backbone with an explicit temporal encoder.

    This keeps the sentence-side interface close to ``CaptionModel_Base`` while
    exposing:
      1. pooled caption memory for sentence decoding
      2. temporal encoded video memory for Stage1 / Stage2 structured modules
    """

    def __init__(
        self,
        vocab_size: int = 49408,
        decoder_nhead: int = 8,
        d_model: int = 512,
        deocder_layer_nums: int = 3,
        temporal_layers: int = 2,
        temporal_dropout: float = 0.1,
        pretrained_clip_name: str = "ViT-B/32",
        init_from_clip: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 49406,
        eos_token_id: int = 49407,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.pad_token_id = int(pad_token_id)
        self.bos_token_id = int(bos_token_id)
        self.eos_token_id = int(eos_token_id)

        self.mean_pooling = SequenceMeanPooling(dim=1, keepdim=True, eps=1e-12)
        self.word_embeddings = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.pad_token_id)
        self.pos_embeddings = LearnedPositionalEmbedding(
            num_positions=200,
            embedding_dim=self.d_model,
            dropout=0.0,
            padding_idx=self.pad_token_id,
        )
        self.video_pos_embeddings = LearnedPositionalEmbedding(
            num_positions=64,
            embedding_dim=self.d_model,
            dropout=0.0,
            padding_idx=0,
        )
        self.norm_input = nn.LayerNorm(self.d_model)
        self.temporal_norm = nn.LayerNorm(self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=decoder_nhead,
            dim_feedforward=2048,
            dropout=float(temporal_dropout),
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=max(1, int(temporal_layers)),
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=decoder_nhead,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=deocder_layer_nums)
        self.norm_output = nn.LayerNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=True)

        if init_from_clip:
            pretrained_clip_model, _ = load_clip_model(pretrained_clip_name)
            pretrained_clip_model.eval()
            clip_token_emb = pretrained_clip_model.token_embedding.weight.detach()
            with torch.no_grad():
                if tuple(clip_token_emb.shape) == tuple(self.word_embeddings.weight.shape):
                    clip_token_emb = clip_token_emb.to(
                        device=self.word_embeddings.weight.device,
                        dtype=self.word_embeddings.weight.dtype,
                    )
                    self.word_embeddings.weight.copy_(clip_token_emb)
                    self.lm_head.weight.copy_(clip_token_emb)
            for param in pretrained_clip_model.parameters():
                param.requires_grad = False

    def encode_video(
        self,
        video_feats: torch.Tensor,
        vid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if video_feats.dim() != 3:
            raise ValueError(f"[encode_video] expected [B,T,D], got {tuple(video_feats.shape)}")
        if vid_mask.dtype != torch.bool:
            vid_mask = vid_mask.bool()
        encoded_video = self.video_pos_embeddings(video_feats, attention_mask=vid_mask.long())
        encoded_video = self.temporal_encoder(encoded_video, src_key_padding_mask=~vid_mask)
        encoded_video = self.temporal_norm(encoded_video)
        pooled_memory = self.mean_pooling(encoded_video, attention_mask=vid_mask)
        memory_key_padding_mask = torch.zeros(
            pooled_memory.size(0),
            pooled_memory.size(1),
            dtype=torch.bool,
            device=pooled_memory.device,
        )
        return encoded_video, vid_mask, pooled_memory, memory_key_padding_mask

    def embed_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = input_ids != self.pad_token_id
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand(input_ids.size(0), -1)
        hidden = self.word_embeddings(input_ids)
        hidden = self.pos_embeddings(hidden, position_ids=positions, attention_mask=attention_mask)
        return self.norm_input(hidden)

    def project_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm_output(hidden))

    def decode_from_memory(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
        captions: torch.Tensor,
        caption_mask: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        if captions.size(1) < 2:
            raise ValueError("captions length must be >= 2 for shifted decoding")
        input_ids = captions[:, :-1].contiguous()
        attn_mask = caption_mask[:, :-1].contiguous() if caption_mask is not None else None
        tgt_emb = self.embed_text(input_ids, attn_mask)
        length = input_ids.size(1)
        tgt_mask = torch.triu(
            torch.ones(length, length, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        tgt_key_padding_mask = (attn_mask == 0).bool() if attn_mask is not None else None
        hidden = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        logits = self.project_hidden(hidden)
        if return_hidden:
            return logits, hidden
        return logits

    def forward(
        self,
        video_feats: torch.Tensor,
        vid_mask: torch.Tensor,
        captions: torch.Tensor,
        caption_mask: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        _encoded_video, _clean_mask, pooled_memory, memory_key_padding_mask = self.encode_video(video_feats, vid_mask)
        return self.decode_from_memory(
            memory=pooled_memory,
            memory_key_padding_mask=memory_key_padding_mask,
            captions=captions,
            caption_mask=caption_mask,
            return_hidden=return_hidden,
        )
