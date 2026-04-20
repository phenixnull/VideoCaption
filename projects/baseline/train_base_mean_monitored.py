# -*- coding: utf-8 -*-
"""
train_base_mean_monitored.py

在不改变原有主要参数与训练流程风格的前提下：
1) 训练阶段仍使用 CaptionModel_Base 按 CE 训练；
2) 增加 TensorBoard 实时监控（step 粒度的 train/loss、train/acc、train/ppl 等），
3) **每个 epoch 结束后在 val split 上做推理评估**，计算并记录 BLEU-4、ROUGE_L、CIDEr、METEOR（不启用 SPICE），
   - 将每轮的指标写入 TensorBoard 与 epoch_metrics.csv/**val_metrics.jsonl**；
   - **并额外记录 val 的 epoch loss（Teacher-Forcing CE 均值）到 TensorBoard（val/epoch_loss）与 val_metrics.jsonl**；
   - 保存当轮 ckpt（每一轮均保存，便于 infer/eval 使用）。

日志与模型保存目录仍为：{out_dir}/{dataset_type}_{model_type}_YYYYmmdd_HHMMSS/，
其中 out_dir 的默认值可自定（示例使用 runs/base_mean_ks20），与原脚本一致的风格。

【TensorBoard 监控】
在服务器上启动（示例）：
  tensorboard --logdir ./runs --port 6006 --bind_all
从本地 Windows 通过 SSH 转发（示例）：
  ssh -N -L 16006:127.0.0.1:6006 <user>@<remote_host>
然后本地浏览 http://127.0.0.1:16006

【依赖说明】
- 必需：torch、tqdm、numpy、matplotlib（已有）
- 新增：tensorboard（SummaryWriter）
  pip install tensorboard
- 指标：pycocoevalcap（本工程已有 temp/pycocoevalcap-master 版本；本脚本有自动 fallback）
  pip install pycocoevalcap  # 若未安装将自动走本地 temp 目录
- METEOR 需要 Java 运行时（服务器建议安装 OpenJDK）：
  sudo apt-get update && sudo apt-get install -y openjdk-11-jre-headless

"""

import os
import math
import json
import time
import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# 项目内模块
from models import CaptionModel_Base
from dataloaders.dataset_msrvtt_feats import MSRVTT_FeaturesDataset
from dataloaders.dataset_msvd_feats import MSVD_FeaturesDataset
from load_tokenizers import CLIPTokenizer_Custom

# -----------------------
# 评测（不启用 SPICE）
# -----------------------
try:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
except Exception:
    # fallback 到工程内置路径
    import sys
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
    local_eval_pkg = os.path.join(project_root, 'temp', 'pycocoevalcap-master')
    if local_eval_pkg not in sys.path:
        sys.path.append(local_eval_pkg)
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider


# -----------------------
# 常用工具函数
# -----------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_gpus(gpus_str: str):
    if gpus_str is None or str(gpus_str).strip() == "":
        return []
    s = gpus_str.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    out = []
    for p in s.split(","):
        p = p.strip()
        if p == "":
            continue
        out.append(int(p))
    return out


def is_torchrun_env() -> bool:
    return ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ) and ("LOCAL_RANK" in os.environ)


def setup_ddp_env(local_rank: int, world_size: int, backend: str = "nccl"):
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=int(os.environ["RANK"]), world_size=world_size)
    return torch.device(f"cuda:{local_rank}")


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def moving_average(xs, window=100):
    n = len(xs)
    if n == 0:
        return []
    # 保证窗口不超过当前点数，且至少为 1
    window = max(1, min(int(window), n))
    cumsum = np.cumsum(np.insert(xs, 0, 0.0))
    ma = (cumsum[window:] - cumsum[:-window]) / float(window)
    # 长度与 xs 保持一致
    return [np.nan] * (window - 1) + list(ma)


def save_curves(log_steps, loss_steps, out_dir, ma_window=200):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 原始曲线
    plt.figure()
    plt.plot(log_steps, loss_steps, linewidth=1)
    plt.xlabel("Global Step")
    plt.ylabel("Train CE Loss")
    plt.title("Training Loss")
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=160)
    plt.close()

    # 平滑曲线：窗口不超过已有点数
    win = max(1, min(int(ma_window), len(loss_steps)))
    ma = moving_average(loss_steps, window=win)

    plt.figure()
    plt.plot(log_steps, loss_steps, alpha=0.35, linewidth=1)
    if len(ma) == len(log_steps):
        plt.plot(log_steps, ma, linewidth=2)
    plt.xlabel("Global Step")
    plt.ylabel(f"Loss (MA-{win})")
    plt.title("Training Loss (Moving Average)")
    plt.tight_layout()
    plt.savefig(out_dir / f"loss_curve_ma{win}.png", dpi=160)
    plt.close()


def save_metrics_csv(metrics: list, out_path: Path):
    import csv
    keys = sorted(list(metrics[0].keys())) if metrics else []
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for m in metrics:
            w.writerow(m)


def accuracy_from_logits(logits: torch.Tensor, target: torch.Tensor, pad_id: int = 0) -> float:
    with torch.no_grad():
        pred = logits.argmax(dim=-1)  # [B, L]
        mask = (target != pad_id).to(logits.dtype)
        correct = ((pred == target).to(logits.dtype) * mask).sum()
        denom = mask.sum().clamp_min(1.0)
        acc = (correct / denom).item()
    return float(acc)


def append_jsonl(path: Path, obj: dict):
    """可靠地追加一行 JSON（解决换行被意外拆断的问题）"""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# -----------------------
# 生成：
#   - greedy_generate_batch：批量贪心（训练监控更稳）
#   - beam_search_batch：批量集束搜索（更冲指标，默认 val 时启用）
# -----------------------
@torch.no_grad()
def greedy_generate_batch(
    model: torch.nn.Module,
    vid_feat: torch.Tensor,  # [B, T, D]
    vid_mask: torch.Tensor,  # [B, T] 1=有效
    tokenizer,
    max_new_tokens: int = 76,
    top_p: float = 0.9,
    temperature: float = 0.0,  # 0=贪心（默认）
) -> List[str]:
    device = vid_feat.device
    B = vid_feat.size(0)

    bos_id = getattr(model, "bos_token_id", 49406)
    eos_id = getattr(model, "eos_token_id", 49407)
    pad_id = getattr(model, "pad_token_id", 0)

    seqs = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for step in range(max_new_tokens):
        cur_len = seqs.size(1)
        captions = torch.full((B, cur_len + 1), pad_id, dtype=torch.long, device=device)
        captions[:, :cur_len] = seqs
        caption_mask = torch.zeros_like(captions, dtype=torch.bool)
        caption_mask[:, :cur_len] = True

        logits = model(vid_feat, vid_mask, captions, caption_mask)  # [B, cur_len, V]
        logits_last = logits[:, -1, :]

        if temperature <= 0:
            next_ids = torch.argmax(logits_last, dim=-1)
        else:
            logits_last = logits_last / max(1e-6, float(temperature))
            probs = torch.softmax(logits_last, dim=-1)
            # top-p 过滤
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cum = torch.cumsum(sorted_probs, dim=-1)
            mask = cum <= top_p
            mask[:, :1] = True
            filtered = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))
            filtered = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            next_sorted = torch.multinomial(filtered, num_samples=1).squeeze(-1)
            next_ids = sorted_idx.gather(dim=-1, index=next_sorted.unsqueeze(-1)).squeeze(-1)

        # 避免第一步直接 EOS
        early_eos = (cur_len == 1) & (next_ids == eos_id)
        if early_eos.any():
            logits_last2 = logits_last.clone()
            logits_last2[early_eos, eos_id] = -1e9
            next_ids2 = torch.argmax(logits_last2, dim=-1)
            next_ids = torch.where(early_eos, next_ids2, next_ids)

        seqs = torch.cat([seqs, next_ids.unsqueeze(-1)], dim=-1)
        finished = finished | (next_ids == eos_id)
        if bool(finished.all()):
            break

    texts = []
    for i in range(B):
        ids = seqs[i].tolist()
        if eos_id in ids:
            eos_pos = ids.index(eos_id)
            ids = ids[:eos_pos + 1]
        text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        texts.append(text)
    return texts


@torch.no_grad()
def beam_search_batch(
    model: torch.nn.Module,
    vid_feat: torch.Tensor,  # [B, T, D]
    vid_mask: torch.Tensor,  # [B, T]
    tokenizer,
    beam_size: int = 5,
    max_new_tokens: int = 76,
    alpha: float = 0.7,  # GNMT length penalty
) -> List[str]:
    """简洁高效的 beam search（按样本循环；每步对 beam 做一次前向）。
    说明：alpha=0.0 退化为无长度惩罚；beam_size<=1 近似贪心。
    """
    device = vid_feat.device
    B = vid_feat.size(0)

    bos_id = getattr(model, "bos_token_id", 49406)
    eos_id = getattr(model, "eos_token_id", 49407)
    pad_id = getattr(model, "pad_token_id", 0)

    def length_penalty(L: int, a: float) -> float:
        # GNMT：lp = ((5+L)^a)/((5+1)^a)
        return ((5.0 + L) ** a) / ((5.0 + 1.0) ** a)

    results: List[str] = []

    for i in range(B):
        vf = vid_feat[i:i+1]
        vm = vid_mask[i:i+1]

        # beams：list of (token_ids, cum_logp, ended)
        beams = [([bos_id], 0.0, False)]

        for step in range(max_new_tokens):
            # 以当前 beams 的 **最大** 序列长度对齐，避免不同 beam 长度不一致导致维度错误
            max_len = max(len(seq) for seq, _, _ in beams)

            # 组装批量 captions 与 mask
            m = len(beams)
            captions = torch.full((m, max_len + 1), pad_id, dtype=torch.long, device=device)
            caption_mask = torch.zeros((m, max_len + 1), dtype=torch.bool, device=device)
            for j, (seq, _, _) in enumerate(beams):
                sl = len(seq)
                captions[j, :sl] = torch.tensor(seq, dtype=torch.long, device=device)
                caption_mask[j, :sl] = True

            # 扩展视频特征到 m 条 beam 一起前向
            vf_rep = vf.expand(m, -1, -1)
            vm_rep = vm.expand(m, -1)

            logits = model(vf_rep, vm_rep, captions, caption_mask)
            logits_last = logits[:, -1, :]
            log_probs = torch.log_softmax(logits_last, dim=-1)

            candidates = []
            for j, (seq, score, ended) in enumerate(beams):
                if ended:
                    # 已结束的 beam 原样保留（不再扩展）
                    candidates.append((seq, score, True))
                    continue
                values, indices = torch.topk(log_probs[j], k=min(beam_size, log_probs.size(-1)))
                cur_len_seq = len(seq)
                for k in range(values.size(0)):
                    nid = int(indices[k].item())
                    nscore = float(score + values[k].item())
                    # 避免第一步直接 EOS（仅当序列还只有 BOS 时）
                    if cur_len_seq == 1 and nid == eos_id:
                        continue
                    nseq = seq + [nid]
                    nend = (nid == eos_id)
                    candidates.append((nseq, nscore, nend))

            # 选择新的前 beam_size 条
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_size]

            # 若全部结束则提前停止
            if all(e for _, _, e in beams):
                break

        # 选择最终输出（带长度惩罚）
        best_seq, best_score = None, -1e9
        for seq, score, ended in beams:
            L = max(1, len(seq) - 1)  # 不计 BOS
            lp = length_penalty(L, alpha)
            norm = score / lp
            if norm > best_score:
                best_score = norm
                best_seq = seq

        # 截到 EOS
        if eos_id in best_seq:
            eos_pos = best_seq.index(eos_id)
            best_seq = best_seq[:eos_pos + 1]
        text = tokenizer.decode(best_seq, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        results.append(text)

    return results


# -----------------------
# 指标计算（不启用 SPICE）
# -----------------------

def _normalize_hyp(d: Dict) -> Dict[str, List[str]]:
    out = {}
    for k, v in d.items():
        ks = str(k)
        if isinstance(v, list):
            out[ks] = [str(v[0] if v else "")]
        else:
            out[ks] = [str(v)]
    return out


def _normalize_ref(d: Dict) -> Dict[str, List[str]]:
    out = {}
    for k, v in d.items():
        ks = str(k)
        if isinstance(v, list):
            out[ks] = [str(x) for x in v]
        else:
            out[ks] = [str(v)]
    return out


def _intersect(gts: Dict[str, List[str]], res: Dict[str, List[str]]):
    ids = sorted(set(gts) & set(res), key=str)
    gts_f = {i: gts[i] for i in ids}
    res_f = {i: res[i] for i in ids}
    return gts_f, res_f, ids


def compute_metrics_no_spice(gts_raw: Dict, hyp_raw: Dict) -> Dict[str, float]:
    gts = _normalize_ref(gts_raw)
    res = _normalize_hyp(hyp_raw)
    gts_f, res_f, ids = _intersect(gts, res)

    bleu = Bleu(4)
    bleu_score, _ = bleu.compute_score(gts_f, res_f)
    meteor = Meteor()
    meteor_score, _ = meteor.compute_score(gts_f, res_f)
    rouge = Rouge()
    rouge_score, _ = rouge.compute_score(gts_f, res_f)
    cider = Cider()
    cider_score, _ = cider.compute_score(gts_f, res_f)

    return {
        "BLEU-1": float(bleu_score[0]) * 100,
        "BLEU-2": float(bleu_score[1]) * 100,
        "BLEU-3": float(bleu_score[2]) * 100,
        "BLEU-4": float(bleu_score[3]) * 100,
        "METEOR": float(meteor_score) * 100,
        "ROUGE_L": float(rouge_score) * 100,
        "CIDEr": float(cider_score) * 100,
    }


# -----------------------
# 训练与验证评估
# -----------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler,  # 新增：学习率调度器
    device: torch.device,
    epoch: int,
    args,
    ddp: bool,
    pad_id: int,
    global_step_start: int,
    writer: SummaryWriter = None,
):
    model.train()
    if ddp and isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)

    loss_meter = 0.0
    n_tok_meter = 0
    global_step = global_step_start

    pbar = tqdm(loader, desc=f"Epoch {epoch}", disable=ddp and dist.get_rank() != 0)

    accum = max(1, int(args.accum_steps))
    optimizer.zero_grad(set_to_none=True)

    for it, batch in enumerate(pbar, start=1):
        vid_feat, vid_mask, caption_ids, caption_mask, _, _, _ = batch
        vid_feat = vid_feat.to(device, non_blocking=True)
        vid_mask = vid_mask.to(device, non_blocking=True).bool()  # 用 bool 掩码以匹配 MHA 要求、避免 dtype 警告
        caption_ids = caption_ids.to(device, non_blocking=True)
        caption_mask = caption_mask.to(device, non_blocking=True)

        target = caption_ids[:, 1:].contiguous()

        with autocast(enabled=bool(args.amp)):
            logits = model(vid_feat, vid_mask, caption_ids, caption_mask)  # [B,76,V]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
                ignore_index=pad_id,
            )
            loss = loss / accum

        if bool(args.amp):
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if it % accum == 0:
            if bool(args.amp):
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            # 学习率调度器 step
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        # step 级指标
        with torch.no_grad():
            loss_raw = float(loss.item() * accum)
            ppl = float(math.exp(loss_raw))
            acc = accuracy_from_logits(logits.detach(), target, pad_id=pad_id)

        loss_meter += loss_raw
        n_tok_meter += int((target != pad_id).sum().item())
        global_step += 1

        if (not ddp) or dist.get_rank() == 0:
            pbar.set_postfix({"loss": f"{loss_meter / max(1, it):.4f}", "acc": f"{acc:.3f}", "ppl": f"{ppl:.2f}"})
            if writer is not None:
                writer.add_scalar("train/loss", loss_raw, global_step)
                writer.add_scalar("train/acc", acc, global_step)
                writer.add_scalar("train/ppl", ppl, global_step)
                # 记录当前学习率
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar("train/lr", current_lr, global_step)

    avg_loss = loss_meter / max(1, len(loader))
    avg_tok = n_tok_meter / max(1, len(loader))
    return avg_loss, avg_tok, global_step


@torch.no_grad()
def evaluate_on_val(
    model: nn.Module,
    device: torch.device,
    args,
    out_dir: Path,
    epoch: int,
) -> Dict[str, float]:
    """在 **val** split 上仅计算 CE loss（teacher-forcing），返回 `VAL_LOSS`。
    注意：仅在 rank0 上调用即可。
    """
    model.eval()

    # 数据（val）
    if args.dataset_type == 'msrvtt':
        val_dataset = MSRVTT_FeaturesDataset(
            features_path=args.clip_global_vision_feats_path,
            json_path=args.annotations_path,
            split='val',
        )
    elif args.dataset_type == 'msvd':
        val_dataset = MSVD_FeaturesDataset(
            features_path=args.clip_global_vision_feats_path,
            annotations_path=args.annotations_path,
            split='val',
        )
    else:
        raise ValueError(f"Unsupported dataset_type: {args.dataset_type}")

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # 统计 val loss（按 batch 平均）
    val_loss_sum = 0.0
    val_loss_n = 0

    pbar = tqdm(val_loader, desc=f"Val (epoch {epoch})", leave=False)
    for batch in pbar:
        vid_feat, vid_mask, caption_ids, caption_mask, _, vids, _ = batch
        vid_feat = vid_feat.to(device, non_blocking=True)
        vid_mask = vid_mask.to(device, non_blocking=True).bool()
        caption_ids = caption_ids.to(device, non_blocking=True)
        caption_mask = caption_mask.to(device, non_blocking=True)

        logits_tf = model(vid_feat, vid_mask, caption_ids, caption_mask)
        target = caption_ids[:, 1:].contiguous()
        loss_val = F.cross_entropy(
            logits_tf.reshape(-1, logits_tf.size(-1)),
            target.reshape(-1),
            ignore_index=0,
        )
        val_loss_sum += float(loss_val.item())
        val_loss_n += 1

    scores = {
        "VAL_LOSS": float(val_loss_sum / max(1, val_loss_n)),
    }
    return scores


@torch.no_grad()
def evaluate_on_test(
    model: nn.Module,
    device: torch.device,
    args,
    out_dir: Path,
    epoch: int,
) -> Dict[str, float]:
    """在 **test** split 上做完整推理与评测，返回指标字典（含 TEST_LOSS）。
    注意：仅在 rank0 上调用即可。
    """
    model.eval()

    # 数据（test）
    if args.dataset_type == 'msrvtt':
        test_dataset = MSRVTT_FeaturesDataset(
            features_path=args.clip_global_vision_feats_path,
            json_path=args.annotations_path,
            split='test',
        )
    elif args.dataset_type == 'msvd':
        test_dataset = MSVD_FeaturesDataset(
            features_path=args.clip_global_vision_feats_path,
            annotations_path=args.annotations_path,
            split='test',
        )
    else:
        raise ValueError(f"Unsupported dataset_type: {args.dataset_type}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    tokenizer = CLIPTokenizer_Custom()
    hyp: Dict[str, List[str]] = {}

    # 统计 test loss（按 batch 平均）
    test_loss_sum = 0.0
    test_loss_n = 0

    pbar = tqdm(test_loader, desc=f"Test (epoch {epoch})", leave=False)
    for batch in pbar:
        vid_feat, vid_mask, caption_ids, caption_mask, _, vids, _ = batch
        vid_feat = vid_feat.to(device, non_blocking=True)
        vid_mask = vid_mask.to(device, non_blocking=True).bool()
        caption_ids = caption_ids.to(device, non_blocking=True)
        caption_mask = caption_mask.to(device, non_blocking=True)

        # 1) CE loss（teacher-forcing）
        logits_tf = model(vid_feat, vid_mask, caption_ids, caption_mask)
        target = caption_ids[:, 1:].contiguous()
        loss_test = F.cross_entropy(
            logits_tf.reshape(-1, logits_tf.size(-1)),
            target.reshape(-1),
            ignore_index=0,
        )
        test_loss_sum += float(loss_test.item())
        test_loss_n += 1

        # 2) 生成文本用于四个指标
        texts = (
            beam_search_batch(model, vid_feat, vid_mask, tokenizer,
                              beam_size=max(2, int(getattr(args, 'beam_size', 5))),
                              max_new_tokens=76,
                              alpha=float(getattr(args, 'beam_alpha', 0.7)))
            if int(getattr(args, 'beam_size', 5)) > 1 else
            greedy_generate_batch(model, vid_feat, vid_mask, tokenizer,
                                  max_new_tokens=76, top_p=0.9, temperature=0.0)
        )
        for v, t in zip(vids, texts):
            hyp[str(v)] = [t]

    # 多参考：直接从 dataset 的 captions_data 取所有句子
    refs: Dict[str, List[str]] = {}
    for vid in test_dataset.video_ids:
        caps = [c for (c, _sid) in test_dataset.captions_data.get(vid, [])]
        if len(caps) == 0:
            caps = [""]
        refs[vid] = caps

    # 保存本轮 test 的 hyp/ref
    eval_dir = out_dir / "test_eval"
    eval_dir.mkdir(exist_ok=True, parents=True)
    hyp_path = eval_dir / f"test_epoch_{epoch:03d}_hyp.json"
    ref_path = eval_dir / f"test_epoch_{epoch:03d}_ref.json"
    with open(hyp_path, "w", encoding="utf-8") as f:
        json.dump(hyp, f, ensure_ascii=False, indent=2)
    with open(ref_path, "w", encoding="utf-8") as f:
        json.dump(refs, f, ensure_ascii=False, indent=2)

    # 计算指标（不启用 SPICE）
    scores = compute_metrics_no_spice(refs, hyp)
    scores["TEST_LOSS"] = float(test_loss_sum / max(1, test_loss_n))
    return scores


# -----------------------
# 主流程
# -----------------------

def main_worker(local_rank: int, args):
    ddp = bool(args.ddp)

    # 可见 GPU 设置
    gpu_ids = parse_gpus(args.gpus)
    if len(gpu_ids) > 0:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", ",".join(map(str, gpu_ids)))

    # 设备/进程
    if ddp and is_torchrun_env():
        device = setup_ddp_env(local_rank=int(os.environ["LOCAL_RANK"]), world_size=int(os.environ["WORLD_SIZE"]))
        rank = dist.get_rank()
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        rank = 0

    # 仅 rank0 建目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"{args.dataset_type}_{args.model_type}_{timestamp}"
    if (not ddp) or rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "checkpoints").mkdir(exist_ok=True)
        with open(out_dir / "args.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # 随机种子
    set_seed(args.seed + rank)

    # 数据（train）
    if args.dataset_type == 'msrvtt':
        train_dataset = MSRVTT_FeaturesDataset(
            features_path=args.clip_global_vision_feats_path,
            json_path=args.annotations_path,
            split='train',
        )
    elif args.dataset_type == 'msvd':
        train_dataset = MSVD_FeaturesDataset(
            features_path=args.clip_global_vision_feats_path,
            annotations_path=args.annotations_path,
            split='train',
        )
    else:
        raise ValueError(f"Unsupported dataset_type: {args.dataset_type}")

    if ddp and is_torchrun_env():
        sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # 模型
    model = CaptionModel_Base(
        vocab_size=49408,
        decoder_nhead=args.decoder_nhead,
        d_model=args.d_model,
        deocder_layer_nums=args.num_layers,
        init_we=args.init_we,
        init_lmhead=args.init_lmhead,
        pad_token_id=0,
        bos_token_id=49406,
        eos_token_id=49407,
        frozen_we=args.frozen_we,
        frozen_lmhead=args.frozen_lmhead,
    ).to(device)

    if ddp and is_torchrun_env():
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])], output_device=int(os.environ["LOCAL_RANK"]), find_unused_parameters=False)

    # 优化器 / AMP
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    scaler = GradScaler(enabled=bool(args.amp))
    
    # 学习率调度器
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    
    def lr_lambda(current_step: int) -> float:
        """Warmup + Cosine/Linear decay 学习率调度"""
        if current_step < warmup_steps:
            # 线性 warmup
            return float(current_step) / float(max(1, warmup_steps))
        
        if args.scheduler == "none":
            return 1.0
        
        # decay 阶段
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        
        if args.scheduler == "cosine":
            # cosine decay 到 min_lr
            min_lr_ratio = args.min_lr / args.lr
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        elif args.scheduler == "linear":
            # 线性 decay 到 min_lr
            min_lr_ratio = args.min_lr / args.lr
            return max(min_lr_ratio, 1.0 - progress)
        else:
            return 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    if (not ddp) or rank == 0:
        print(f"[Scheduler] {args.scheduler}, warmup_epochs={args.warmup_epochs}, warmup_steps={warmup_steps}, total_steps={total_steps}")

    # TensorBoard（rank0 负责写入）
    writer = None
    if (not ddp) or rank == 0:
        writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    # 训练循环
    global_step = 0
    epoch_metrics = []

    # 记录 step loss 以画图（这里仍使用 epoch 聚合点防止早期窗口过大）
    loss_steps: List[float] = []
    step_idx: List[int] = []

    try:
        for epoch in range(1, args.epochs + 1):
            avg_loss, avg_tok, global_step = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,  # 传入学习率调度器
                device=device,
                epoch=epoch,
                args=args,
                ddp=ddp and is_torchrun_env(),
                pad_id=0,
                global_step_start=global_step,
                writer=writer,
            )

            if (not ddp) or dist.get_rank() == 0:
                ppl_epoch = float(math.exp(avg_loss))
                epoch_metrics.append({
                    "epoch": epoch,
                    "avg_loss": float(avg_loss),
                    "ppl": ppl_epoch,
                    "avg_tokens_per_step": float(avg_tok),
                    "time": time.time(),
                })

                # 追加 epoch 点以画快速对比曲线
                step_idx.append(global_step)
                loss_steps.append(avg_loss)
                save_curves(step_idx, loss_steps, out_dir, ma_window=args.ma_window)

                # TB 记 epoch 粒度
                if writer is not None:
                    writer.add_scalar("train_epoch/avg_loss", avg_loss, epoch)
                    writer.add_scalar("train_epoch/ppl", ppl_epoch, epoch)

                # ========== 验证评估（val：只看 loss） ==========
                val_scores = evaluate_on_val(model if not isinstance(model, DDP) else model.module,
                                             device=device, args=args, out_dir=out_dir, epoch=epoch)
                # 打印 val 上的 loss
                print(f"[Val] epoch {epoch}: VAL_LOSS={val_scores['VAL_LOSS']:.4f}")
                # 写入 TB（val/epoch_loss）
                writer.add_scalar("val/epoch_loss", val_scores["VAL_LOSS"], epoch)

                # 记录当前学习率
                current_lr = optimizer.param_groups[0]['lr']

                # 记录 val loss 到 jsonl
                append_jsonl(out_dir / "val_metrics.jsonl",
                             {"epoch": epoch, "VAL_LOSS": float(val_scores["VAL_LOSS"]), "time": time.time()})

                # ========== 测试评估（test：四个指标 + loss） ==========
                test_scores = evaluate_on_test(model if not isinstance(model, DDP) else model.module,
                                               device=device, args=args, out_dir=out_dir, epoch=epoch)
                print(
                    f"[Test] epoch {epoch}: "
                    f"BLEU-4={test_scores['BLEU-4']:.2f}, "
                    f"CIDEr={test_scores['CIDEr']:.2f}, "
                    f"ROUGE_L={test_scores['ROUGE_L']:.2f}, "
                    f"METEOR={test_scores['METEOR']:.2f}, "
                    f"TEST_LOSS={test_scores['TEST_LOSS']:.4f}"
                )

                # 写入 TB（test/*）
                writer.add_scalar("test/BLEU-4",  test_scores["BLEU-4"],  epoch)
                writer.add_scalar("test/ROUGE_L", test_scores["ROUGE_L"], epoch)
                writer.add_scalar("test/CIDEr",   test_scores["CIDEr"],   epoch)
                writer.add_scalar("test/METEOR",  test_scores["METEOR"],  epoch)
                writer.add_scalar("test/epoch_loss", test_scores["TEST_LOSS"], epoch)

                # 逐轮保存到 jsonl 与 csv
                append_jsonl(out_dir / "test_metrics.jsonl",
                             {"epoch": epoch, **{k: float(v) for k, v in test_scores.items()}, "time": time.time()})
                append_jsonl(out_dir / "metrics.jsonl",
                             {"epoch": epoch, "avg_loss": float(avg_loss), "ppl": ppl_epoch, "lr": current_lr, "time": time.time()})

                save_metrics_csv(epoch_metrics, out_dir / "epoch_metrics.csv")

                # 保存 ckpt（每轮一份）
                ckpt = {
                    "epoch": epoch,
                    "model": (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "scaler": scaler.state_dict(),
                    "args": vars(args),
                    "global_step": global_step,
                    "current_lr": current_lr,
                }
                torch.save(ckpt, out_dir / "checkpoints" / f"epoch_{epoch:03d}.pt")

    finally:
        if writer is not None:
            writer.flush()
            writer.close()
        if ddp and is_torchrun_env():
            cleanup_ddp()


# -----------------------
# CLI
# -----------------------

def build_parser():
    ap = argparse.ArgumentParser()
    # 数据集与路径
    ap.add_argument("--dataset_type", default="msrvtt", choices=["msrvtt", "msvd"])
    ap.add_argument("--model_type", default="base")
    # 特征与标注路径留空，稍后根据 dataset_type 自动填充默认值
    ap.add_argument("--clip_global_vision_feats_path", default=None)
    ap.add_argument("--annotations_path", default=None)
    ap.add_argument("--out_dir", default="./runs/base_mean_ks20")

    # 模型结构
    ap.add_argument("--decoder_nhead", type=int, default=8)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--init_we", type=int, default=1)
    ap.add_argument("--init_lmhead", type=int, default=1)
    ap.add_argument("--frozen_we", type=int, default=1)
    ap.add_argument("--frozen_lmhead", type=int, default=0)

    # 训练超参
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--min_lr", type=float, default=1e-6, help="cosine decay 最小学习率")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--accum_steps", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--ma_window", type=int, default=200)
    
    # 学习率调度
    ap.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine", "linear"], help="学习率调度策略")
    ap.add_argument("--warmup_epochs", type=int, default=5, help="warmup 轮数")

    # 推理/搜索
    ap.add_argument("--beam_size", type=int, default=5, help="beam size >1 启用 beam search；=1 等价贪心")
    ap.add_argument("--beam_alpha", type=float, default=0.7, help="GNMT length penalty alpha")

    # 环境
    ap.add_argument("--ddp", type=int, default=0)
    ap.add_argument("--gpus", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    return ap


def main():
    parser = build_parser()
    args = parser.parse_args()

    # 根据 dataset_type 自动填充默认的数据路径
    if args.dataset_type == "msrvtt":
        if args.clip_global_vision_feats_path is None:
            args.clip_global_vision_feats_path = "../datasets/MSRVTT/feats/ViT-B-32_k_split_ks12_features.pickle"
        if args.annotations_path is None:
            args.annotations_path = "../datasets/MSRVTT/MSRVTT_data.json"
        # out_dir 默认保持原样
    elif args.dataset_type == "msvd":
        if args.clip_global_vision_feats_path is None:
            args.clip_global_vision_feats_path = "../datasets/MSVD/feats/ViT-B-32_k_split_ks12_features.pickle"
        if args.annotations_path is None:
            # 使用全集预处理标注，由 Dataset 内部结合 train/val/test.txt 做划分
            args.annotations_path = "../datasets/MSVD/annotations_preprocessed.txt"
        # 如果仍是默认 MSRVTT 输出目录，则为 MSVD 换一个更合理的默认
        if args.out_dir == "./runs/base_mean_ks20":
            args.out_dir = "./runs/msvd_base_ks12"
    else:
        raise ValueError(f"Unsupported dataset_type: {args.dataset_type}")

    if bool(args.ddp) and is_torchrun_env():
        main_worker(local_rank=int(os.environ.get("LOCAL_RANK", 0)), args=args)
    else:
        main_worker(local_rank=0, args=args)


if __name__ == "__main__":
    main()

"""
# ===================== MSRVTT =====================
# 参考命令（单卡）：
# python train_base_mean_monitored.py \
#   --dataset_type msrvtt \
#   --clip_global_vision_feats_path ../datasets/MSRVTT/feats/ViT-B-32_k_split_ks12_features.pickle \
#   --annotations_path ../datasets/MSRVTT/MSRVTT_data.json \
#   --out_dir ./runs/base_mean \
#   --batch_size 128
#
# 参考命令（多卡-DDP，4卡）：
# torchrun --nproc_per_node=4 train_base_mean_monitored.py \
#   --ddp 1 --gpus 0,1,2,3 \
#   --model_type base_ks20 \
#   --dataset_type msrvtt \
#   --clip_global_vision_feats_path ../datasets/MSRVTT/feats/ViT-B-32_k_split_ks20_features.pickle \
#   --annotations_path ../datasets/MSRVTT/MSRVTT_data.json \
#   --out_dir ./runs/base_mean_ks20 \
#   --batch_size 256

# ===================== MSVD =====================
# 参考命令（单卡）：
# python train_base_mean_monitored.py \
#   --dataset_type msvd \
#   --clip_global_vision_feats_path ../datasets/MSVD/feats/ViT-B-32_k_split_ks12_features.pickle \
#   --annotations_path ../datasets/MSVD/annotations_preprocessed.txt \
#   --out_dir ./runs/msvd_base_ks12 \
#   --batch_size 128
#
# 参考命令（多卡-DDP，2卡，使用卡2和卡3）：
# CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 train_base_mean_monitored.py \
#   --ddp 1 \
#   --model_type base_ks12 \
#   --dataset_type msvd \
#   --clip_global_vision_feats_path ../datasets/MSVD/feats/ViT-B-32_k_split_ks12_features.pickle \
#   --annotations_path ../datasets/MSVD/annotations_preprocessed.txt \
#   --out_dir ./runs/msvd_base_ks12 \
#   --batch_size 256
"""
