# -*- coding: utf-8 -*-
"""Structured decoder training with entity/action/phrase supervision.

This script keeps baseline training/eval flow and adds auxiliary losses:
1) video -> entity prior BCE
2) video -> action prior BCE
3) video -> attribute prior BCE (optional/known-mask)
4) video -> scene prior BCE (optional/known-mask)
5) decoder hidden <-> phrase-unit embedding cosine alignment
"""

import argparse
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import train_base_mean_monitored as base
from dataloaders.dataset_msvd_feats import MSVD_FeaturesDataset
from dataloaders.dataset_msrvtt_feats import MSRVTT_FeaturesDataset
from dataloaders.dataset_structured_caption import StructuredCaptionDataset
from models_structured import StructuredCaptionModel
from phrase_lexical_anchors import build_phrase_lexical_anchor_kwargs


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = bool(requires_grad)


def _set_optional_parameter_requires_grad(model: nn.Module, attr_name: str, requires_grad: bool) -> None:
    value = getattr(model, attr_name, None)
    if isinstance(value, nn.Parameter):
        value.requires_grad = bool(requires_grad)


def _set_optional_module_requires_grad(model: nn.Module, attr_names: Tuple[str, ...], requires_grad: bool) -> None:
    for attr_name in attr_names:
        _set_requires_grad(getattr(model, attr_name, None), requires_grad)


_STAGE1_PRIOR_MODULE_ATTRS = (
    "entity_prior_head",
    "action_prior_head",
    "attribute_prior_head",
    "scene_prior_head",
)

_STAGE2_PHRASE_MODULE_ATTRS = (
    "phrase_decoder",
    "phrase_output_norm",
    "slot_embeddings",
    "slot_role_embeddings",
    "entity_label_embeddings",
    "action_label_embeddings",
    "attribute_label_embeddings",
    "scene_label_embeddings",
    "slot_planner_query",
    "slot_planner_key",
    "slot_planner_value",
    "slot_planner_norm",
    "slot_planner_flow_query",
    "slot_planner_flow_key",
    "slot_planner_flow_value",
    "slot_planner_flow_gate",
    "slot_planner_flow_update",
    "slot_planner_flow_norm",
    "slot_presence_head",
    "slot_presence_support_proj",
    "slot_presence_evidence_proj",
    "slot_guidance_target_proj",
    "slot_guidance_support_proj",
    "slot_guidance_evidence_proj",
    "slot_guidance_norm",
    "slot_role_anchor_target_proj",
    "slot_role_anchor_role_proj",
    "slot_role_anchor_norm",
    "entity_to_context",
    "action_to_context",
    "attribute_to_context",
    "scene_to_context",
    "condition_gate",
    "condition_norm",
    "struct_condition_query_bridge_source_embeddings",
    "struct_condition_query_bridge_source_key",
    "struct_condition_query_bridge_source_value",
    "struct_condition_query_bridge_token_norm",
)

_STAGE2_PHRASE_PARAMETER_ATTRS = (
    "attribute_context_scale",
    "scene_context_scale",
    "struct_condition_query_bridge_queries",
    "struct_condition_query_bridge_scale",
    "struct_condition_query_bridge_memory_scale",
)

_STAGE3_SENTENCE_MODULE_ATTRS = (
    "condition_dropout",
    "condition_post_norm",
    "struct_condition_query_bridge_hidden_query",
    "struct_condition_query_bridge_hidden_key",
    "struct_condition_query_bridge_hidden_value",
    "struct_condition_query_bridge_hidden_gate",
    "struct_condition_query_bridge_hidden_norm",
    "phrase_to_context",
    "phrase_condition_gate",
    "phrase_condition_norm",
    "phrase_slot_condition_proj",
    "phrase_slot_condition_presence_proj",
    "phrase_slot_condition_norm",
    "phrase_condition_core_proj",
    "phrase_condition_aux_query",
    "phrase_condition_aux_key",
    "phrase_condition_aux_value",
    "phrase_condition_aux_gate",
    "phrase_condition_aux_norm",
    "phrase_condition_slot_residual_query",
    "phrase_condition_slot_residual_key",
    "phrase_condition_slot_residual_value",
    "phrase_condition_slot_residual_gate",
    "phrase_condition_slot_residual_norm",
    "phrase_condition_slot_residual_order_embeddings",
    "phrase_condition_slot_residual_type_embeddings",
    "phrase_condition_family_embeddings",
    "phrase_condition_family_query",
    "phrase_condition_family_key",
    "phrase_condition_family_value",
    "phrase_condition_family_gate",
    "phrase_condition_family_norm",
    "phrase_condition_candidate_query",
    "phrase_condition_candidate_key",
    "phrase_condition_candidate_value",
    "phrase_condition_candidate_gate",
    "phrase_condition_query_bridge_summary_key",
    "phrase_condition_query_bridge_summary_value",
    "phrase_condition_query_bridge_summary_norm",
    "phrase_condition_query_bridge_hidden_query",
    "phrase_condition_query_bridge_hidden_key",
    "phrase_condition_query_bridge_hidden_value",
    "phrase_condition_query_bridge_gate",
    "phrase_condition_query_bridge_norm",
)

_STAGE3_SENTENCE_PARAMETER_ATTRS = (
    "struct_scale",
    "struct_condition_query_bridge_hidden_scale",
    "phrase_scale",
    "phrase_condition_aux_scale",
    "phrase_condition_slot_residual_scale",
    "phrase_condition_family_scale",
    "phrase_condition_candidate_scale",
    "phrase_condition_query_bridge_queries",
    "phrase_condition_query_bridge_scale",
)

_PHRASE_LOSS_ATTRS = (
    "lambda_phrase",
    "lambda_phrase_gen",
    "lambda_phrase_pred_gen",
    "lambda_phrase_slot_presence",
    "lambda_phrase_slot_div",
    "lambda_phrase_ref_slot_align",
    "lambda_phrase_ref_bridge",
    "lambda_phrase_bridge",
    "lambda_phrase_slot_source_align",
    "lambda_phrase_slot_source_comp",
)

_PRIOR_LOSS_ATTRS = (
    "lambda_entity",
    "lambda_action",
    "lambda_attr",
    "lambda_scene",
)


def _apply_training_stage_loss_profile(args: argparse.Namespace) -> str:
    stage_name = str(getattr(args, "training_stage", "joint")).strip().lower() or "joint"
    if stage_name in {"joint", "full", "none"}:
        return "joint"

    if stage_name == "stage1_word":
        setattr(args, "lambda_ce", 0.0)
        for attr_name in _PHRASE_LOSS_ATTRS:
            setattr(args, attr_name, 0.0)
    elif stage_name == "stage2_phrase":
        setattr(args, "lambda_ce", 0.0)
        for attr_name in _PRIOR_LOSS_ATTRS:
            setattr(args, attr_name, 0.0)
    elif stage_name == "stage3_sentence":
        for attr_name in _PRIOR_LOSS_ATTRS + _PHRASE_LOSS_ATTRS:
            setattr(args, attr_name, 0.0)
    else:
        raise ValueError(f"Unsupported training_stage: {stage_name}")

    return stage_name


def _apply_training_stage_profile(model: StructuredCaptionModel, args: argparse.Namespace) -> str:
    stage_name = str(getattr(args, "training_stage", "joint")).strip().lower() or "joint"
    if stage_name in {"joint", "full", "none"}:
        return "joint"

    for param in model.parameters():
        param.requires_grad = False

    if stage_name == "stage1_word":
        _set_optional_module_requires_grad(model, _STAGE1_PRIOR_MODULE_ATTRS, True)
        return stage_name

    if stage_name == "stage2_phrase":
        _set_optional_module_requires_grad(model, _STAGE2_PHRASE_MODULE_ATTRS, True)
        for attr_name in _STAGE2_PHRASE_PARAMETER_ATTRS:
            _set_optional_parameter_requires_grad(model, attr_name, True)
        return stage_name

    if stage_name == "stage3_sentence":
        _set_requires_grad(model.caption_model, True)
        _set_optional_module_requires_grad(model, _STAGE3_SENTENCE_MODULE_ATTRS, True)
        for attr_name in _STAGE3_SENTENCE_PARAMETER_ATTRS:
            _set_optional_parameter_requires_grad(model, attr_name, True)
        if bool(getattr(args, "frozen_we", False)):
            _set_requires_grad(getattr(model.caption_model, "word_embeddings", None), False)
        if bool(getattr(args, "frozen_lmhead", False)):
            _set_requires_grad(getattr(model.caption_model, "lm_head", None), False)
        return stage_name

    raise ValueError(f"Unsupported training_stage: {stage_name}")


_STRUCTURED_VOCAB_INIT_PREFIXES = (
    "entity_prior_head.",
    "action_prior_head.",
    "attribute_prior_head.",
    "scene_prior_head.",
    "entity_to_context.",
    "action_to_context.",
    "attribute_to_context.",
    "scene_to_context.",
    "entity_label_embeddings.",
    "action_label_embeddings.",
    "attribute_label_embeddings.",
    "scene_label_embeddings.",
)


def _preview_key_list(keys: List[str], limit: int = 8) -> str:
    if not keys:
        return "none"
    shown = keys[:limit]
    suffix = "" if len(keys) <= limit else f" ... (+{len(keys) - limit} more)"
    return ", ".join(shown) + suffix


def _filter_compatible_init_state(
    *,
    target_state: Dict[str, torch.Tensor],
    init_state: Dict[str, torch.Tensor],
    skip_prefixes: Tuple[str, ...],
) -> Tuple[Dict[str, torch.Tensor], List[str], List[str], List[str], List[str]]:
    compatible_state: Dict[str, torch.Tensor] = {}
    skipped_vocab: List[str] = []
    skipped_shape: List[str] = []
    unexpected: List[str] = []
    loaded: List[str] = []

    for key, value in init_state.items():
        if any(key.startswith(prefix) for prefix in skip_prefixes):
            skipped_vocab.append(key)
            continue

        target_value = target_state.get(key)
        if target_value is None:
            unexpected.append(key)
            continue

        if not torch.is_tensor(value) or not torch.is_tensor(target_value):
            unexpected.append(key)
            continue

        if tuple(value.shape) != tuple(target_value.shape):
            skipped_shape.append(f"{key}:{tuple(value.shape)}->{tuple(target_value.shape)}")
            continue

        compatible_state[key] = value
        loaded.append(key)

    missing = sorted(set(target_state.keys()) - set(compatible_state.keys()))
    return compatible_state, loaded, missing, unexpected, skipped_vocab + skipped_shape


def load_initial_caption_weights(
    model: StructuredCaptionModel,
    ckpt_path: str,
    *,
    skip_structured_vocab_modules: bool = False,
) -> None:
    ckpt_file = Path(ckpt_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"init_caption_ckpt not found: {ckpt_file}")

    payload = torch.load(str(ckpt_file), map_location="cpu")
    state = payload.get("model", payload)
    if not isinstance(state, dict):
        raise ValueError(f"Invalid checkpoint format: {ckpt_file}")

    # If checkpoint already matches wrapper keys, load directly.
    is_wrapper_state = any(k.startswith("caption_model.") for k in state.keys())
    target_module = model if is_wrapper_state else model.caption_model
    target_state = target_module.state_dict()
    skip_prefixes = ()
    if skip_structured_vocab_modules:
        base_prefixes = tuple(_STRUCTURED_VOCAB_INIT_PREFIXES)
        skip_prefixes = tuple(
            f"caption_model.{prefix}" for prefix in base_prefixes
        ) if is_wrapper_state else base_prefixes

    compatible_state, loaded, missing, unexpected, skipped = _filter_compatible_init_state(
        target_state=target_state,
        init_state=state,
        skip_prefixes=skip_prefixes,
    )
    if is_wrapper_state:
        missing_after_load, unexpected_after_load = model.load_state_dict(compatible_state, strict=False)
    else:
        missing_after_load, unexpected_after_load = model.caption_model.load_state_dict(
            compatible_state,
            strict=False,
        )
    missing = sorted(set(missing) | set(missing_after_load))
    unexpected = sorted(set(unexpected) | set(unexpected_after_load))
    skipped_count = len(skipped)

    print(
        f"[Init] loaded compatible caption weights from {ckpt_file} "
        f"(loaded={len(loaded)}, missing={len(missing)}, unexpected={len(unexpected)}, skipped={skipped_count}, "
        f"skip_structured_vocab_modules={int(skip_structured_vocab_modules)})"
    )
    if skipped:
        print(f"[Init] skipped init keys: {_preview_key_list(skipped)}")
    if unexpected:
        print(f"[Init] unexpected init keys: {_preview_key_list(unexpected)}")
    if missing:
        print(f"[Init] missing target keys after warm start: {_preview_key_list(missing)}")


def _safe_zero_like(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.sum() * 0.0


def _reshape_known_mask_like(target: torch.Tensor, known_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if known_mask is None:
        return None
    if known_mask.shape == target.shape:
        return known_mask
    if target.dim() >= 2 and known_mask.dim() == target.dim() - 1 and tuple(known_mask.shape) == tuple(target.shape[:-1]):
        return known_mask.unsqueeze(-1).expand_as(target)
    if target.dim() == 2 and known_mask.dim() == 1 and known_mask.size(0) == target.size(0):
        return known_mask.unsqueeze(-1).expand_as(target)
    return known_mask


def _masked_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    known_mask: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if logits is None or target is None or known_mask is None:
        return target.sum() * 0.0 if target is not None else logits.sum() * 0.0
    if target.numel() == 0:
        return target.sum() * 0.0
    known_mask = _reshape_known_mask_like(target, known_mask)
    known = known_mask.reshape(-1) > 0.5
    if not bool(known.any()):
        return target.sum() * 0.0
    masked_weight = weight[known] if weight is not None else None
    return F.binary_cross_entropy_with_logits(logits[known], target[known], weight=masked_weight)


def _asymmetric_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    known_mask: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    gamma_neg: float = 4.0,
    gamma_pos: float = 1.0,
    clip: float = 0.05,
    eps: float = 1e-8,
) -> torch.Tensor:
    if logits is None or target is None:
        return target.sum() * 0.0 if target is not None else logits.sum() * 0.0
    if target.numel() == 0:
        return target.sum() * 0.0

    flat_logits = logits.reshape(-1)
    flat_target = target.reshape(-1)
    flat_weight = weight.reshape(-1) if weight is not None else None
    if known_mask is not None:
        known_mask = _reshape_known_mask_like(target, known_mask)
        known = known_mask.reshape(-1) > 0.5
        if not bool(known.any()):
            return target.sum() * 0.0
        flat_logits = flat_logits[known]
        flat_target = flat_target[known]
        if flat_weight is not None:
            flat_weight = flat_weight[known]

    prob_pos = torch.sigmoid(flat_logits)
    prob_neg = 1.0 - prob_pos
    if clip > 0.0:
        prob_neg = (prob_neg + float(clip)).clamp(max=1.0)

    loss = flat_target * torch.log(prob_pos.clamp_min(eps))
    loss = loss + (1.0 - flat_target) * torch.log(prob_neg.clamp_min(eps))

    if gamma_neg > 0.0 or gamma_pos > 0.0:
        pt = prob_pos * flat_target + prob_neg * (1.0 - flat_target)
        gamma = gamma_pos * flat_target + gamma_neg * (1.0 - flat_target)
        loss = loss * torch.pow((1.0 - pt).clamp_min(0.0), gamma)

    loss = -loss
    if flat_weight is not None:
        loss = loss * flat_weight
    return loss.mean()


def _compute_prior_loss(
    *,
    logits: Optional[torch.Tensor],
    target: torch.Tensor,
    args,
    fallback: torch.Tensor,
    known_mask: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if logits is None or target is None or target.numel() == 0:
        return _safe_zero_like(fallback)
    prior_loss_type = str(getattr(args, "prior_loss_type", "bce")).strip().lower()
    if prior_loss_type == "asl":
        return _asymmetric_loss_with_logits(
            logits=logits,
            target=target,
            known_mask=known_mask,
            weight=weight,
            gamma_neg=float(getattr(args, "prior_asl_gamma_neg", 4.0)),
            gamma_pos=float(getattr(args, "prior_asl_gamma_pos", 1.0)),
            clip=float(getattr(args, "prior_asl_clip", 0.05)),
            eps=float(getattr(args, "prior_asl_eps", 1e-8)),
        )
    if known_mask is not None:
        return _masked_bce_with_logits(
            logits=logits,
            target=target,
            known_mask=known_mask,
            weight=weight,
        )
    return F.binary_cross_entropy_with_logits(logits, target, weight=weight)


def _build_caption_aware_positive_weight(
    target: torch.Tensor,
    caption_target: Optional[torch.Tensor],
    *,
    caption_pos_weight: float,
    video_only_pos_weight: float,
) -> Optional[torch.Tensor]:
    if target is None or target.numel() == 0:
        return None
    if abs(float(caption_pos_weight) - 1.0) < 1e-8 and abs(float(video_only_pos_weight) - 1.0) < 1e-8:
        return None

    weights = torch.ones_like(target)
    positive_mask = target > 0.5
    if not bool(positive_mask.any()):
        return weights

    if caption_target is None or caption_target.numel() == 0:
        weights = torch.where(
            positive_mask,
            torch.full_like(weights, float(video_only_pos_weight)),
            weights,
        )
        return weights

    caption_positive_mask = positive_mask & (caption_target > 0.5)
    video_only_positive_mask = positive_mask & (~caption_positive_mask)
    if bool(caption_positive_mask.any()):
        weights = torch.where(
            caption_positive_mask,
            torch.full_like(weights, float(caption_pos_weight)),
            weights,
        )
    if bool(video_only_positive_mask.any()):
        weights = torch.where(
            video_only_positive_mask,
            torch.full_like(weights, float(video_only_pos_weight)),
            weights,
        )
    return weights


def _resolve_prior_weight_pair(args, head_name: str) -> Tuple[float, float]:
    caption_key = f"{head_name}_prior_caption_pos_weight"
    video_only_key = f"{head_name}_prior_video_only_pos_weight"
    caption_weight = getattr(args, caption_key, None)
    video_only_weight = getattr(args, video_only_key, None)
    if caption_weight is None:
        caption_weight = getattr(args, "prior_caption_pos_weight", 1.0)
    if video_only_weight is None:
        video_only_weight = getattr(args, "prior_video_only_pos_weight", 1.0)
    return float(caption_weight), float(video_only_weight)


def phrase_alignment_loss(
    hidden_states: torch.Tensor,
    caption_ids: torch.Tensor,
    caption_mask: torch.Tensor,
    phrase_ids: torch.Tensor,
    phrase_mask: torch.Tensor,
    embedding_layer: nn.Embedding,
    pad_id: int = 0,
    bos_id: int = 49406,
    eos_id: int = 49407,
) -> torch.Tensor:
    """Align decoder hidden semantics with phrase-unit text embeddings."""
    hidden_token_ids = caption_ids[:, :-1].contiguous()
    hidden_valid = caption_mask[:, :-1].bool()
    hidden_valid = hidden_valid & hidden_token_ids.ne(pad_id)
    hidden_valid = hidden_valid & hidden_token_ids.ne(bos_id) & hidden_token_ids.ne(eos_id)
    hidden_mask = hidden_valid.to(hidden_states.dtype).unsqueeze(-1)
    hidden_den = hidden_mask.sum(dim=1).clamp_min(1.0)
    hidden_pooled = (hidden_states * hidden_mask).sum(dim=1) / hidden_den

    phrase_emb = embedding_layer(phrase_ids)
    phrase_valid = phrase_mask.bool()
    phrase_valid = phrase_valid & phrase_ids.ne(pad_id)
    phrase_valid = phrase_valid & phrase_ids.ne(bos_id) & phrase_ids.ne(eos_id)
    phrase_mask_f = phrase_valid.to(phrase_emb.dtype).unsqueeze(-1)
    phrase_den = phrase_mask_f.sum(dim=1).clamp_min(1.0)
    phrase_pooled = (phrase_emb * phrase_mask_f).sum(dim=1) / phrase_den

    valid_rows = hidden_valid.any(dim=1) & phrase_valid.any(dim=1)
    if not bool(valid_rows.any()):
        return _safe_zero_like(hidden_states)

    cos_sim = F.cosine_similarity(hidden_pooled[valid_rows], phrase_pooled[valid_rows], dim=-1)
    return (1.0 - cos_sim).mean()


def _resolve_phrase_slot_multiref_reduce_settings(
    *,
    args,
    predicted: bool = False,
) -> Tuple[str, float]:
    reduce_mode = str(getattr(args, "phrase_slot_multiref_reduce", "mean")).strip().lower() or "mean"
    softmin_temp = 1.0
    if predicted:
        pred_reduce_mode = (
            str(getattr(args, "phrase_slot_pred_multiref_reduce", "inherit")).strip().lower() or "inherit"
        )
        if pred_reduce_mode != "inherit":
            reduce_mode = pred_reduce_mode
        softmin_temp = max(1e-4, float(getattr(args, "phrase_slot_pred_multiref_softmin_temp", 1.0)))
    return reduce_mode, softmin_temp


def _reduce_phrase_slot_multiref_ref_loss(
    *,
    ref_loss: torch.Tensor,
    ref_valid: torch.Tensor,
    reduce_mode: str,
    softmin_temp: float = 1.0,
) -> torch.Tensor:
    slot_valid = ref_valid.any(dim=-1)
    if not bool(slot_valid.any()):
        return _safe_zero_like(ref_loss)

    if reduce_mode == "sum":
        slot_loss = ref_loss.sum(dim=-1)
    elif reduce_mode == "mean":
        slot_loss = ref_loss.sum(dim=-1) / ref_valid.to(ref_loss.dtype).sum(dim=-1).clamp_min(1.0)
    elif reduce_mode == "min":
        masked_ref_loss = torch.where(ref_valid, ref_loss, torch.full_like(ref_loss, 1e6))
        slot_loss = masked_ref_loss.min(dim=-1).values
    elif reduce_mode == "softmin":
        slot_loss = torch.zeros(ref_loss.shape[:2], dtype=torch.float32, device=ref_loss.device)
        valid_ref_loss = ref_loss[slot_valid].float()
        valid_ref_mask = ref_valid[slot_valid]
        if valid_ref_loss.numel() > 0:
            logits = (-valid_ref_loss / max(1e-4, float(softmin_temp))).masked_fill(
                ~valid_ref_mask,
                -1e4,
            )
            weights = torch.softmax(logits, dim=-1)
            weights = weights * valid_ref_mask.to(weights.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            slot_loss[slot_valid] = (valid_ref_loss * weights).sum(dim=-1)
    else:
        raise ValueError(f"Unsupported phrase slot multiref reduce mode: {reduce_mode}")

    slot_loss = slot_loss * slot_valid.to(slot_loss.dtype)
    return slot_loss.sum() / slot_valid.to(slot_loss.dtype).sum().clamp_min(1.0)


def phrase_slot_multiref_loss(
    *,
    model_ref: StructuredCaptionModel,
    aux: Dict[str, torch.Tensor],
    phrase_slot_ref_ids: Optional[torch.Tensor],
    phrase_slot_ref_mask: Optional[torch.Tensor],
    phrase_slot_ref_valid: Optional[torch.Tensor],
    slot_token_weights: Optional[torch.Tensor],
    args,
    fallback: torch.Tensor,
) -> torch.Tensor:
    if (
        phrase_slot_ref_ids is None
        or phrase_slot_ref_mask is None
        or phrase_slot_ref_valid is None
        or phrase_slot_ref_ids.numel() == 0
        or not bool(phrase_slot_ref_valid.any())
    ):
        return _safe_zero_like(fallback)

    phrase_teacher_memory = aux.get("phrase_teacher_memory")
    phrase_teacher_memory_key_padding_mask = aux.get("phrase_teacher_memory_key_padding_mask")
    if phrase_teacher_memory is None or phrase_teacher_memory_key_padding_mask is None:
        return _safe_zero_like(fallback)

    ref_valid = phrase_slot_ref_valid.bool()
    ref_loss = torch.zeros_like(phrase_slot_ref_valid, dtype=fallback.dtype)
    ref_count = int(phrase_slot_ref_ids.size(2))
    chunk_size = max(1, int(getattr(args, "phrase_slot_multiref_chunk_size", 4)))
    ce_batch_chunk = 2
    pad_id = int(model_ref.pad_token_id)
    slot_contexts = aux.get("phrase_teacher_slot_contexts")
    for ref_start in range(0, ref_count, chunk_size):
        ref_end = min(ref_count, ref_start + chunk_size)
        chunk_ref_count = ref_end - ref_start
        ref_slice_valid = ref_valid[:, :, ref_start:ref_end]
        if not bool(ref_slice_valid.any()):
            continue
        chunk_slot_ids = phrase_slot_ref_ids[:, :, ref_start:ref_end, :]
        chunk_slot_mask = phrase_slot_ref_mask[:, :, ref_start:ref_end, :]
        flat_slot_ids = chunk_slot_ids.permute(0, 2, 1, 3).reshape(
            phrase_slot_ref_ids.size(0) * chunk_ref_count,
            phrase_slot_ref_ids.size(1),
            phrase_slot_ref_ids.size(3),
        )
        flat_slot_mask = chunk_slot_mask.permute(0, 2, 1, 3).reshape(
            phrase_slot_ref_mask.size(0) * chunk_ref_count,
            phrase_slot_ref_mask.size(1),
            phrase_slot_ref_mask.size(3),
        )
        flat_slot_contexts = None
        if slot_contexts is not None:
            flat_slot_contexts = slot_contexts.repeat_interleave(chunk_ref_count, dim=0)
        flat_teacher_memory = phrase_teacher_memory.repeat_interleave(chunk_ref_count, dim=0)
        flat_teacher_memory_key_padding_mask = (
            phrase_teacher_memory_key_padding_mask.repeat_interleave(chunk_ref_count, dim=0)
        )
        flat_ref_target = chunk_slot_ids[:, :, :, 1:].permute(0, 2, 1, 3).reshape(
            phrase_slot_ref_ids.size(0) * chunk_ref_count,
            phrase_slot_ref_ids.size(1),
            phrase_slot_ref_ids.size(3) - 1,
        )
        flat_ref_valid = ref_slice_valid.permute(0, 2, 1).reshape(
            phrase_slot_ref_ids.size(0) * chunk_ref_count,
            phrase_slot_ref_ids.size(1),
        )
        if not bool((flat_ref_target.ne(pad_id) & flat_ref_valid.unsqueeze(-1)).any()):
            continue

        slot_scale = None
        if slot_token_weights is not None and slot_token_weights.numel() >= flat_ref_target.size(1):
            slot_scale = slot_token_weights[: flat_ref_target.size(1)].view(1, -1, 1)

        flat_ref_loss = torch.zeros(
            (flat_ref_target.size(0), flat_ref_target.size(1)),
            dtype=fallback.dtype,
            device=fallback.device,
        )
        for flat_start in range(0, flat_ref_target.size(0), ce_batch_chunk):
            flat_end = min(flat_ref_target.size(0), flat_start + ce_batch_chunk)
            decode_outputs = model_ref._decode_phrase_slots(
                memory=flat_teacher_memory[flat_start:flat_end],
                memory_key_padding_mask=flat_teacher_memory_key_padding_mask[flat_start:flat_end],
                phrase_slot_ids=flat_slot_ids[flat_start:flat_end],
                phrase_slot_mask=flat_slot_mask[flat_start:flat_end],
                slot_contexts=(
                    flat_slot_contexts[flat_start:flat_end]
                    if flat_slot_contexts is not None
                    else None
                ),
            )
            logits_slice = decode_outputs["phrase_slot_logits"]
            target_slice = flat_ref_target[flat_start:flat_end]
            valid_slice = flat_ref_valid[flat_start:flat_end]
            token_loss = F.cross_entropy(
                logits_slice.reshape(-1, logits_slice.size(-1)),
                target_slice.reshape(-1),
                ignore_index=pad_id,
                reduction="none",
            ).view_as(target_slice)
            token_mask = target_slice.ne(pad_id) & valid_slice.unsqueeze(-1)
            if not bool(token_mask.any()):
                continue
            weight_mask = token_mask.to(token_loss.dtype)
            if slot_scale is not None:
                weight_mask = weight_mask * slot_scale.to(dtype=token_loss.dtype, device=token_loss.device)
            flat_ref_loss[flat_start:flat_end] = (
                (token_loss * weight_mask).sum(dim=-1)
                / weight_mask.sum(dim=-1).clamp_min(1.0)
            )
        ref_loss[:, :, ref_start:ref_end] = flat_ref_loss.reshape(
            phrase_slot_ref_ids.size(0),
            chunk_ref_count,
            phrase_slot_ref_ids.size(1),
        ).permute(0, 2, 1)
    ref_loss = ref_loss * ref_valid.to(ref_loss.dtype)

    reduce_mode, softmin_temp = _resolve_phrase_slot_multiref_reduce_settings(args=args, predicted=False)
    return float(getattr(args, "phrase_slot_multiref_gain", 1.0)) * _reduce_phrase_slot_multiref_ref_loss(
        ref_loss=ref_loss,
        ref_valid=ref_valid,
        reduce_mode=reduce_mode,
        softmin_temp=softmin_temp,
    )


def phrase_slot_multiref_loss_from_logits(
    *,
    model_ref: StructuredCaptionModel,
    slot_logits: Optional[torch.Tensor],
    phrase_slot_ref_ids: Optional[torch.Tensor],
    phrase_slot_ref_valid: Optional[torch.Tensor],
    slot_token_weights: Optional[torch.Tensor],
    args,
    fallback: torch.Tensor,
) -> torch.Tensor:
    if (
        slot_logits is None
        or phrase_slot_ref_ids is None
        or phrase_slot_ref_valid is None
        or slot_logits.numel() == 0
        or phrase_slot_ref_ids.numel() == 0
        or not bool(phrase_slot_ref_valid.any())
    ):
        return _safe_zero_like(fallback)

    if slot_logits.dim() != 4 or phrase_slot_ref_ids.dim() != 4 or phrase_slot_ref_valid.dim() != 3:
        raise ValueError(
            "slot_logits must be [batch, slots, seq_len, vocab], "
            "phrase_slot_ref_ids [batch, slots, refs, seq_len], "
            "phrase_slot_ref_valid [batch, slots, refs]."
        )

    ref_valid = phrase_slot_ref_valid.bool()
    slot_valid = ref_valid.any(dim=-1)
    if not bool(slot_valid.any()):
        return _safe_zero_like(fallback)

    batch_size, slot_count, pred_len, vocab_size = slot_logits.shape
    _ref_batch, _ref_slot_count, ref_count, ref_seq_len = phrase_slot_ref_ids.shape
    if _ref_batch != batch_size or _ref_slot_count != slot_count:
        raise ValueError("slot_logits and phrase_slot_ref_ids batch/slot dimensions must match.")

    ref_target = phrase_slot_ref_ids[:, :, :, 1:].contiguous()
    ref_loss = torch.zeros(
        (batch_size, slot_count, ref_count),
        dtype=slot_logits.dtype,
        device=slot_logits.device,
    )
    chunk_size = max(1, int(getattr(args, "phrase_slot_multiref_chunk_size", 4)))

    for ref_start in range(0, ref_count, chunk_size):
        ref_end = min(ref_count, ref_start + chunk_size)
        chunk_valid = ref_valid[:, :, ref_start:ref_end]
        if not bool(chunk_valid.any()):
            continue

        chunk_ref_count = ref_end - ref_start
        aligned_target = torch.full(
            (batch_size, slot_count, chunk_ref_count, pred_len),
            int(model_ref.pad_token_id),
            dtype=torch.long,
            device=slot_logits.device,
        )
        copy_len = min(pred_len, max(0, ref_seq_len - 1))
        if copy_len > 0:
            aligned_target[:, :, :, :copy_len] = ref_target[:, :, ref_start:ref_end, :copy_len].to(
                device=slot_logits.device
            )

        expanded_logits = slot_logits.unsqueeze(2).expand(-1, -1, chunk_ref_count, -1, -1)
        token_loss = F.cross_entropy(
            expanded_logits.reshape(-1, vocab_size),
            aligned_target.reshape(-1),
            ignore_index=int(model_ref.pad_token_id),
            reduction="none",
        ).view(batch_size, slot_count, chunk_ref_count, pred_len)

        token_mask = aligned_target.ne(int(model_ref.pad_token_id)) & chunk_valid.unsqueeze(-1)
        if not bool(token_mask.any()):
            continue

        weight_mask = token_mask.to(token_loss.dtype)
        if slot_token_weights is not None and slot_token_weights.numel() >= slot_count:
            slot_scale = slot_token_weights[:slot_count].view(1, slot_count, 1, 1)
            weight_mask = weight_mask * slot_scale.to(dtype=token_loss.dtype, device=token_loss.device)

        chunk_ref_loss = (token_loss * weight_mask).sum(dim=-1) / weight_mask.sum(dim=-1).clamp_min(1.0)
        ref_loss[:, :, ref_start:ref_end] = chunk_ref_loss

    ref_loss = ref_loss * ref_valid.to(ref_loss.dtype)

    reduce_mode, softmin_temp = _resolve_phrase_slot_multiref_reduce_settings(args=args, predicted=True)
    return float(getattr(args, "phrase_slot_multiref_gain", 1.0)) * _reduce_phrase_slot_multiref_ref_loss(
        ref_loss=ref_loss,
        ref_valid=ref_valid,
        reduce_mode=reduce_mode,
        softmin_temp=softmin_temp,
    )


def phrase_slot_diversity_loss(
    slot_summary: torch.Tensor,
    slot_valid: torch.Tensor,
) -> torch.Tensor:
    """Penalize slot-level hidden collapse without using any oracle text at inference."""
    if slot_summary is None or slot_valid is None:
        return slot_summary.sum() * 0.0 if slot_summary is not None else slot_valid.sum() * 0.0
    if slot_summary.dim() != 3 or slot_valid.dim() != 2:
        raise ValueError("slot_summary must be [batch, slots, hidden_dim] and slot_valid must be [batch, slots].")

    pair_counts = slot_valid.sum(dim=1)
    if not bool((pair_counts > 1).any()):
        return _safe_zero_like(slot_summary)

    slot_norm = F.normalize(slot_summary, dim=-1)
    sim = torch.matmul(slot_norm, slot_norm.transpose(1, 2))
    slot_mask = slot_valid.bool()
    off_diag = ~torch.eye(slot_mask.size(1), dtype=torch.bool, device=slot_mask.device).unsqueeze(0)
    pair_mask = slot_mask.unsqueeze(1) & slot_mask.unsqueeze(2) & off_diag
    if not bool(pair_mask.any()):
        return _safe_zero_like(slot_summary)

    return F.relu(sim[pair_mask]).mean()


def phrase_bridge_consistency_loss(
    pred_summary: torch.Tensor,
    pred_valid: torch.Tensor,
    teacher_summary: torch.Tensor,
    teacher_valid: torch.Tensor,
) -> torch.Tensor:
    """Keep the predicted phrase bridge close to the teacher phrase bridge used by supervision."""
    if pred_summary is None or teacher_summary is None:
        if pred_summary is not None:
            return pred_summary.sum() * 0.0
        if teacher_summary is not None:
            return teacher_summary.sum() * 0.0
        raise ValueError("phrase_bridge_consistency_loss requires at least one summary tensor.")
    if pred_summary.dim() != 2 or teacher_summary.dim() != 2:
        raise ValueError("phrase bridge summaries must be [batch, hidden_dim].")

    if pred_valid is None:
        pred_valid = torch.ones(pred_summary.size(0), dtype=torch.bool, device=pred_summary.device)
    if teacher_valid is None:
        teacher_valid = torch.ones(
            teacher_summary.size(0),
            dtype=torch.bool,
            device=teacher_summary.device,
        )

    valid_rows = pred_valid.bool() & teacher_valid.bool()
    if not bool(valid_rows.any()):
        return _safe_zero_like(pred_summary)

    teacher_detached = teacher_summary.detach()
    cos_sim = F.cosine_similarity(pred_summary[valid_rows], teacher_detached[valid_rows], dim=-1)
    return (1.0 - cos_sim).mean()


def summary_alignment_loss(
    pred_summary: Optional[torch.Tensor],
    pred_valid: Optional[torch.Tensor],
    target_summary: Optional[torch.Tensor],
    target_valid: Optional[torch.Tensor],
) -> torch.Tensor:
    """Align predicted summaries to reference-bank semantic centers with cosine distance."""
    if pred_summary is None or target_summary is None:
        if pred_summary is not None:
            return pred_summary.sum() * 0.0
        if target_summary is not None:
            return target_summary.sum() * 0.0
        raise ValueError("summary_alignment_loss requires at least one summary tensor.")
    if pred_summary.dim() != target_summary.dim():
        raise ValueError("pred_summary and target_summary must have the same rank.")

    target_detached = target_summary.detach()
    if pred_summary.dim() == 2:
        if pred_valid is None:
            pred_valid = torch.ones(pred_summary.size(0), dtype=torch.bool, device=pred_summary.device)
        if target_valid is None:
            target_valid = torch.ones(target_summary.size(0), dtype=torch.bool, device=target_summary.device)
        valid_rows = pred_valid.bool() & target_valid.bool()
        if not bool(valid_rows.any()):
            return _safe_zero_like(pred_summary)
        cos_sim = F.cosine_similarity(pred_summary[valid_rows], target_detached[valid_rows], dim=-1)
        return (1.0 - cos_sim).mean()

    if pred_summary.dim() != 3:
        raise ValueError("summary_alignment_loss supports only [batch, hidden] or [batch, slots, hidden].")

    if pred_valid is None:
        pred_valid = torch.ones(
            pred_summary.size(0),
            pred_summary.size(1),
            dtype=torch.bool,
            device=pred_summary.device,
        )
    if target_valid is None:
        target_valid = torch.ones(
            target_summary.size(0),
            target_summary.size(1),
            dtype=torch.bool,
            device=target_summary.device,
        )
    valid_rows = pred_valid.bool() & target_valid.bool()
    if not bool(valid_rows.any()):
        return _safe_zero_like(pred_summary)
    cos_sim = F.cosine_similarity(pred_summary[valid_rows], target_detached[valid_rows], dim=-1)
    return (1.0 - cos_sim).mean()


def _build_slot_source_targets(
    *,
    slot_count: int,
    source_names: List[str],
    phrase_slot_schema: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    prior_map = StructuredCaptionModel.get_slot_source_prior_map(phrase_slot_schema)
    if prior_map is None:
        raise ValueError(f"Unsupported phrase_slot_schema for slot-source targets: {phrase_slot_schema}")
    slot_specs = StructuredCaptionDataset.get_phrase_slot_type_specs(
        max_phrase_slots=slot_count,
        phrase_slot_schema=phrase_slot_schema,
    )
    rows = []
    for spec in slot_specs:
        slot_type = str(spec.get("slot_type", "extra"))
        source_prior = prior_map.get(slot_type, prior_map.get("extra", prior_map.get("generic", {})))
        row = [max(0.0, float(source_prior.get(str(source_name), 0.0))) for source_name in source_names]
        row_sum = float(sum(row))
        if row_sum <= 0.0:
            row = [1.0 / max(1, len(source_names)) for _ in source_names]
        else:
            row = [value / row_sum for value in row]
        rows.append(row)
    return torch.tensor(rows, device=device, dtype=dtype)


def _build_slot_source_pairwise_controls(
    *,
    slot_count: int,
    phrase_slot_schema: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    slot_specs = StructuredCaptionDataset.get_phrase_slot_type_specs(
        max_phrase_slots=slot_count,
        phrase_slot_schema=phrase_slot_schema,
    )
    slot_families = [
        str(spec.get("slot_type_family", spec.get("slot_type", "generic"))).strip().lower() or "generic"
        for spec in slot_specs
    ]

    pair_weights = torch.ones((slot_count, slot_count), device=device, dtype=dtype)
    hard_caps = torch.full((slot_count, slot_count), float("inf"), device=device, dtype=dtype)
    pair_enabled = torch.ones((slot_count, slot_count), device=device, dtype=torch.bool)

    entity_slot_families = {"subject_entity", "object_entity", "subject_modifier", "object_modifier"}
    skip_pairs = {
        frozenset({"subject_entity", "subject_action"}),
        frozenset({"object_entity", "object_passive"}),
    }
    strict_pair_rules = {
        frozenset({"subject_entity", "subject_modifier"}): (1.75, 0.30),
        frozenset({"object_entity", "object_modifier"}): (1.75, 0.30),
        frozenset({"relation_detail", "scene_context"}): (1.35, 0.18),
        frozenset({"instrument_detail", "scene_context"}): (1.20, 0.20),
    }

    for idx in range(slot_count):
        pair_enabled[idx, idx] = False

    for left_idx in range(slot_count):
        left_family = slot_families[left_idx]
        for right_idx in range(left_idx + 1, slot_count):
            right_family = slot_families[right_idx]
            family_pair = frozenset({left_family, right_family})
            if family_pair in skip_pairs:
                pair_enabled[left_idx, right_idx] = False
                pair_enabled[right_idx, left_idx] = False
                continue

            if "scene_context" in family_pair and family_pair & entity_slot_families:
                weight = 1.45
                cap = 0.18
            else:
                weight, cap = strict_pair_rules.get(family_pair, (1.0, None))

            pair_weights[left_idx, right_idx] = weight
            pair_weights[right_idx, left_idx] = weight
            if cap is not None:
                hard_caps[left_idx, right_idx] = cap
                hard_caps[right_idx, left_idx] = cap

    return pair_weights, hard_caps, pair_enabled


def phrase_slot_source_alignment_loss(
    slot_source_weights: torch.Tensor,
    slot_valid: torch.Tensor,
    source_names: List[str],
    phrase_slot_schema: str,
) -> torch.Tensor:
    """Align schema-aware slot planner sources with their intended semantic roles."""
    if slot_source_weights is None or slot_valid is None:
        if slot_source_weights is not None:
            return slot_source_weights.sum() * 0.0
        if slot_valid is not None:
            return slot_valid.sum() * 0.0
        raise ValueError("phrase_slot_source_alignment_loss requires at least one tensor input.")
    if slot_source_weights.dim() != 3 or slot_valid.dim() != 2:
        raise ValueError(
            "slot_source_weights must be [batch, slots, sources] and slot_valid must be [batch, slots]."
        )
    if not source_names:
        return _safe_zero_like(slot_source_weights)
    if StructuredCaptionModel.get_slot_source_prior_map(phrase_slot_schema) is None:
        return _safe_zero_like(slot_source_weights)

    valid_rows = slot_valid.bool()
    if not bool(valid_rows.any()):
        return _safe_zero_like(slot_source_weights)

    target = _build_slot_source_targets(
        slot_count=slot_source_weights.size(1),
        source_names=list(source_names),
        phrase_slot_schema=phrase_slot_schema,
        device=slot_source_weights.device,
        dtype=slot_source_weights.dtype,
    )
    target = target.unsqueeze(0).expand(slot_source_weights.size(0), -1, -1)

    pred = slot_source_weights.clamp_min(1e-8)
    loss = F.kl_div(pred.log(), target, reduction="none").sum(dim=-1)
    return loss[valid_rows].mean()


def phrase_slot_source_competition_loss(
    slot_source_weights: torch.Tensor,
    slot_valid: torch.Tensor,
    source_names: List[str],
    phrase_slot_schema: str,
    margin: float = 0.05,
) -> torch.Tensor:
    """Discourage different phrase slots from collapsing onto the same source mixture."""
    if slot_source_weights is None or slot_valid is None:
        if slot_source_weights is not None:
            return slot_source_weights.sum() * 0.0
        if slot_valid is not None:
            return slot_valid.sum() * 0.0
        raise ValueError("phrase_slot_source_competition_loss requires at least one tensor input.")
    if slot_source_weights.dim() != 3 or slot_valid.dim() != 2:
        raise ValueError(
            "slot_source_weights must be [batch, slots, sources] and slot_valid must be [batch, slots]."
        )
    if not source_names:
        return _safe_zero_like(slot_source_weights)
    if StructuredCaptionModel.get_slot_source_prior_map(phrase_slot_schema) is None:
        return _safe_zero_like(slot_source_weights)

    slot_mask = slot_valid.bool()
    pair_counts = slot_mask.sum(dim=1)
    if not bool((pair_counts > 1).any()):
        return _safe_zero_like(slot_source_weights)

    pred = slot_source_weights / slot_source_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    target = _build_slot_source_targets(
        slot_count=slot_source_weights.size(1),
        source_names=list(source_names),
        phrase_slot_schema=phrase_slot_schema,
        device=slot_source_weights.device,
        dtype=slot_source_weights.dtype,
    )
    target = target.unsqueeze(0).expand(slot_source_weights.size(0), -1, -1)

    pred_overlap = torch.matmul(pred, pred.transpose(1, 2))
    target_overlap = torch.matmul(target, target.transpose(1, 2))
    pair_weights, pair_hard_caps, pair_enabled = _build_slot_source_pairwise_controls(
        slot_count=slot_source_weights.size(1),
        phrase_slot_schema=phrase_slot_schema,
        device=slot_source_weights.device,
        dtype=slot_source_weights.dtype,
    )
    default_cap = target_overlap + float(margin)
    if bool(torch.isfinite(pair_hard_caps).any()):
        pair_cap = torch.minimum(default_cap, pair_hard_caps.unsqueeze(0))
    else:
        pair_cap = default_cap

    off_diag = ~torch.eye(slot_mask.size(1), dtype=torch.bool, device=slot_mask.device).unsqueeze(0)
    pair_mask = slot_mask.unsqueeze(1) & slot_mask.unsqueeze(2) & off_diag & pair_enabled.unsqueeze(0)
    if not bool(pair_mask.any()):
        return _safe_zero_like(slot_source_weights)

    pair_weights = pair_weights.unsqueeze(0).expand_as(pred_overlap)
    excess_overlap = pred_overlap - pair_cap
    penalties = F.relu(excess_overlap[pair_mask]) * pair_weights[pair_mask]
    return penalties.sum() / pair_weights[pair_mask].sum().clamp_min(1e-8)


def build_phrase_slot_presence_thresholds(
    *,
    base_threshold: float,
    pos_weights: List[float],
    min_threshold: float,
    max_threshold: float,
) -> List[float]:
    thresholds: List[float] = []
    base = float(base_threshold)
    lower = float(min_threshold)
    upper = float(max_threshold)
    if lower > upper:
        raise ValueError(
            "phrase_slot_presence_threshold_min must be <= phrase_slot_presence_threshold_max, "
            f"got {min_threshold} > {max_threshold}"
        )
    for pos_weight in pos_weights:
        weight = max(1e-8, float(pos_weight))
        denom = base + (1.0 - base) * weight
        if denom <= 0.0:
            threshold = base
        else:
            threshold = base / denom
        threshold = min(upper, max(lower, float(threshold)))
        thresholds.append(float(threshold))
    return thresholds


def _parse_slot_type_list(slot_types: str) -> List[str]:
    parsed: List[str] = []
    seen = set()
    for raw_slot_type in str(slot_types or "").split(","):
        slot_type = str(raw_slot_type).strip().lower()
        if not slot_type or slot_type in seen:
            continue
        parsed.append(slot_type)
        seen.add(slot_type)
    return parsed


def configure_phrase_slot_reweight(
    args,
    train_dataset: StructuredCaptionDataset,
    out_dir: Path,
    rank: int,
) -> None:
    args._phrase_slot_token_weights = None
    args._phrase_slot_presence_pos_weights = None
    args.phrase_slot_presence_thresholds = None

    calibration_mode = str(getattr(args, "phrase_slot_presence_calibration_mode", "none")).strip().lower()
    need_reweight = bool(getattr(args, "phrase_slot_reweight_enable", 0))
    need_calibration = calibration_mode != "none"
    if str(args.phrase_target_mode).strip().lower() != "slot":
        return
    if not need_reweight and not need_calibration:
        return
    dataset_for_stats = train_dataset
    if not isinstance(dataset_for_stats, StructuredCaptionDataset):
        dataset_for_stats = base.unwrap_dataset(train_dataset)
    if not isinstance(dataset_for_stats, StructuredCaptionDataset):
        raise TypeError("configure_phrase_slot_reweight requires StructuredCaptionDataset after unwrap.")

    reweight_stats = dataset_for_stats.get_phrase_slot_reweight_stats(
        power=float(args.phrase_slot_reweight_power),
        min_weight=float(args.phrase_slot_reweight_min),
        max_weight=float(args.phrase_slot_reweight_max),
    )
    slot_token_weights = list(reweight_stats.get("slot_token_weights", []))
    slot_presence_pos_weights = list(reweight_stats.get("slot_presence_pos_weights", []))
    slot_specs = dataset_for_stats.get_phrase_slot_type_specs(
        max_phrase_slots=int(args.max_phrase_slots),
        phrase_slot_schema=str(args.phrase_slot_schema),
    )

    if need_reweight:
        args._phrase_slot_token_weights = torch.tensor(slot_token_weights, dtype=torch.float32)
        args._phrase_slot_presence_pos_weights = torch.tensor(slot_presence_pos_weights, dtype=torch.float32)

    presence_calibration = {
        "mode": calibration_mode,
        "base_threshold": float(args.phrase_slot_presence_threshold),
        "threshold_min": float(args.phrase_slot_presence_threshold_min),
        "threshold_max": float(args.phrase_slot_presence_threshold_max),
        "selected_slot_types": [],
        "per_slot_thresholds": [],
    }
    if need_calibration:
        if calibration_mode != "pos_weight":
            raise ValueError(f"Unsupported phrase_slot_presence_calibration_mode: {calibration_mode}")
        thresholds = build_phrase_slot_presence_thresholds(
            base_threshold=float(args.phrase_slot_presence_threshold),
            pos_weights=slot_presence_pos_weights,
            min_threshold=float(args.phrase_slot_presence_threshold_min),
            max_threshold=float(args.phrase_slot_presence_threshold_max),
        )
        calibration_slot_types = _parse_slot_type_list(
            getattr(args, "phrase_slot_presence_calibration_slot_types", "")
        )
        if calibration_slot_types:
            selected_slot_types = set(calibration_slot_types)
            thresholds = [
                float(calibrated if str(spec.get("slot_type", "")).strip().lower() in selected_slot_types else args.phrase_slot_presence_threshold)
                for spec, calibrated in zip(slot_specs[: len(thresholds)], thresholds)
            ]
            if len(slot_presence_pos_weights) > len(thresholds):
                thresholds.extend(
                    [float(args.phrase_slot_presence_threshold)] * (len(slot_presence_pos_weights) - len(thresholds))
                )
            presence_calibration["selected_slot_types"] = list(calibration_slot_types)
        args.phrase_slot_presence_thresholds = [float(x) for x in thresholds]
        presence_calibration["per_slot_thresholds"] = list(args.phrase_slot_presence_thresholds)

    reweight_stats["presence_calibration"] = presence_calibration

    if rank == 0:
        with (out_dir / "phrase_slot_reweight_stats.json").open("w", encoding="utf-8") as f:
            json.dump(reweight_stats, f, indent=2, ensure_ascii=False)
        if need_reweight:
            print(
                "[PhraseSlotReweight] "
                f"counts={reweight_stats.get('slot_valid_counts')} "
                f"token_weights={reweight_stats.get('slot_token_weights')} "
                f"presence_pos_weights={reweight_stats.get('slot_presence_pos_weights')}"
            )
        if need_calibration:
            threshold_pairs = ", ".join(
                f"{str(spec.get('slot_type', idx))}={float(args.phrase_slot_presence_thresholds[idx]):.3f}"
                for idx, spec in enumerate(slot_specs[: len(args.phrase_slot_presence_thresholds)])
            )
            print(
                "[PhraseSlotPresenceCalibration] "
                f"mode={calibration_mode} "
                f"base_threshold={float(args.phrase_slot_presence_threshold):.3f} "
                f"clip=[{float(args.phrase_slot_presence_threshold_min):.3f}, "
                f"{float(args.phrase_slot_presence_threshold_max):.3f}] "
                f"slot_types={','.join(presence_calibration['selected_slot_types'])} "
                f"per_slot={threshold_pairs}"
            )


def train_one_epoch_structured(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler,
    device: torch.device,
    epoch: int,
    args,
    ddp: bool,
    pad_id: int,
    global_step_start: int,
    writer: SummaryWriter = None,
) -> Tuple[float, float, float, int, Dict[str, float]]:
    model.train()
    if ddp and isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)

    total_meter = 0.0
    ce_meter = 0.0
    tok_meter = 0
    global_step = global_step_start

    accum = max(1, int(args.accum_steps))
    optimizer.zero_grad(set_to_none=True)
    slot_token_weights = None
    slot_presence_pos_weights = None
    if bool(getattr(args, "phrase_slot_reweight_enable", 0)) and str(args.phrase_target_mode).lower() == "slot":
        token_weights = getattr(args, "_phrase_slot_token_weights", None)
        presence_weights = getattr(args, "_phrase_slot_presence_pos_weights", None)
        if token_weights is not None and len(token_weights) > 0:
            slot_token_weights = token_weights.to(device=device, dtype=torch.float32)
        if presence_weights is not None and len(presence_weights) > 0:
            slot_presence_pos_weights = presence_weights.to(device=device, dtype=torch.float32)

    pbar = tqdm(loader, desc=f"Epoch {epoch}", disable=ddp and dist.get_rank() != 0)
    model_ref = model.module if isinstance(model, DDP) else model
    max_train_steps = max(0, int(getattr(args, "max_train_steps_per_epoch", 0)))
    processed_steps = 0
    relation_detail_slot_idx = None
    relation_detail_stats = {
        "supervised_samples": 0.0,
        "supervised_positive": 0.0,
        "predicted_positive": 0.0,
        "true_positive": 0.0,
        "prob_sum": 0.0,
        "planner_support_sum": 0.0,
        "planner_support_majority": 0.0,
        "planner_support_samples": 0.0,
    }
    if str(args.phrase_target_mode).lower() == "slot":
        slot_specs = StructuredCaptionDataset.get_phrase_slot_type_specs(
            max_phrase_slots=int(args.max_phrase_slots),
            phrase_slot_schema=str(args.phrase_slot_schema),
        )
        for slot_idx, spec in enumerate(slot_specs):
            if str(spec.get("slot_type", "")).strip().lower() == "relation_detail":
                relation_detail_slot_idx = int(slot_idx)
                break
    lambda_phrase_pred_gen_active = float(
        getattr(args, "_lambda_phrase_pred_gen_epoch", getattr(args, "lambda_phrase_pred_gen", 0.0))
    )

    for it, batch in enumerate(pbar, start=1):
        if max_train_steps > 0 and it > max_train_steps:
            break
        batch, aux_inputs = base.split_batch_and_aux(batch)
        (
            vid_feat,
            vid_mask,
            caption_ids,
            caption_mask,
            _,
            _,
            _,
            entity_target,
            action_target,
            attr_target,
            scene_target,
            caption_entity_target,
            caption_action_target,
            caption_attr_target,
            caption_scene_target,
            attr_known_mask,
            scene_known_mask,
            phrase_ids,
            phrase_mask,
            phrase_slot_ids,
            phrase_slot_mask,
            phrase_slot_valid,
        ) = batch

        vid_feat = vid_feat.to(device, non_blocking=True)
        vid_mask = vid_mask.to(device, non_blocking=True).bool()
        caption_ids = caption_ids.to(device, non_blocking=True)
        caption_mask = caption_mask.to(device, non_blocking=True)
        entity_target = entity_target.to(device, non_blocking=True)
        action_target = action_target.to(device, non_blocking=True)
        attr_target = attr_target.to(device, non_blocking=True)
        scene_target = scene_target.to(device, non_blocking=True)
        caption_entity_target = caption_entity_target.to(device, non_blocking=True)
        caption_action_target = caption_action_target.to(device, non_blocking=True)
        caption_attr_target = caption_attr_target.to(device, non_blocking=True)
        caption_scene_target = caption_scene_target.to(device, non_blocking=True)
        attr_known_mask = attr_known_mask.to(device, non_blocking=True)
        scene_known_mask = scene_known_mask.to(device, non_blocking=True)
        phrase_ids = phrase_ids.to(device, non_blocking=True)
        phrase_mask = phrase_mask.to(device, non_blocking=True)
        phrase_slot_ids = phrase_slot_ids.to(device, non_blocking=True)
        phrase_slot_mask = phrase_slot_mask.to(device, non_blocking=True)
        phrase_slot_valid = phrase_slot_valid.to(device, non_blocking=True)
        aux_inputs = base.move_aux_tensors_to_device(aux_inputs, device)
        phrase_slot_ref_ids = aux_inputs.get("phrase_slot_ref_ids")
        phrase_slot_ref_mask = aux_inputs.get("phrase_slot_ref_mask")
        phrase_slot_ref_valid = aux_inputs.get("phrase_slot_ref_valid")
        model_aux_inputs = dict(aux_inputs)

        target = caption_ids[:, 1:].contiguous()
        entity_caption_pos_weight, entity_video_only_pos_weight = _resolve_prior_weight_pair(args, "entity")
        action_caption_pos_weight, action_video_only_pos_weight = _resolve_prior_weight_pair(args, "action")
        attr_caption_pos_weight, attr_video_only_pos_weight = _resolve_prior_weight_pair(args, "attr")
        scene_caption_pos_weight, scene_video_only_pos_weight = _resolve_prior_weight_pair(args, "scene")
        entity_prior_weight = _build_caption_aware_positive_weight(
            entity_target,
            caption_entity_target,
            caption_pos_weight=entity_caption_pos_weight,
            video_only_pos_weight=entity_video_only_pos_weight,
        )
        action_prior_weight = _build_caption_aware_positive_weight(
            action_target,
            caption_action_target,
            caption_pos_weight=action_caption_pos_weight,
            video_only_pos_weight=action_video_only_pos_weight,
        )
        attr_prior_weight = _build_caption_aware_positive_weight(
            attr_target,
            caption_attr_target,
            caption_pos_weight=attr_caption_pos_weight,
            video_only_pos_weight=attr_video_only_pos_weight,
        )
        scene_prior_weight = _build_caption_aware_positive_weight(
            scene_target,
            caption_scene_target,
            caption_pos_weight=scene_caption_pos_weight,
            video_only_pos_weight=scene_video_only_pos_weight,
        )

        with autocast(enabled=bool(args.amp)):
            logits, hidden, aux = model(
                vid_feat,
                vid_mask,
                caption_ids,
                caption_mask,
                phrase_ids=phrase_ids,
                phrase_mask=phrase_mask,
                phrase_slot_ids=phrase_slot_ids,
                phrase_slot_mask=phrase_slot_mask,
                return_hidden=True,
                return_aux=True,
                **model_aux_inputs,
            )

            ce_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
                ignore_index=pad_id,
            )

            if args.lambda_entity > 0:
                entity_loss = _compute_prior_loss(
                    logits=aux.get("entity_prior_logits"),
                    target=entity_target,
                    args=args,
                    fallback=ce_loss,
                    weight=entity_prior_weight,
                )
            else:
                entity_loss = _safe_zero_like(ce_loss)

            if args.lambda_action > 0:
                action_loss = _compute_prior_loss(
                    logits=aux.get("action_prior_logits"),
                    target=action_target,
                    args=args,
                    fallback=ce_loss,
                    weight=action_prior_weight,
                )
            else:
                action_loss = _safe_zero_like(ce_loss)

            if args.lambda_attr > 0:
                attr_loss = _compute_prior_loss(
                    logits=aux.get("attribute_prior_logits"),
                    target=attr_target,
                    args=args,
                    fallback=ce_loss,
                    known_mask=attr_known_mask,
                    weight=attr_prior_weight,
                )
            else:
                attr_loss = _safe_zero_like(ce_loss)

            if args.lambda_scene > 0:
                scene_loss = _compute_prior_loss(
                    logits=aux.get("scene_prior_logits"),
                    target=scene_target,
                    args=args,
                    fallback=ce_loss,
                    known_mask=scene_known_mask,
                    weight=scene_prior_weight,
                )
            else:
                scene_loss = _safe_zero_like(ce_loss)

            if args.lambda_phrase > 0:
                phrase_loss = phrase_alignment_loss(
                    hidden_states=hidden,
                    caption_ids=caption_ids,
                    caption_mask=caption_mask,
                    phrase_ids=phrase_ids,
                    phrase_mask=phrase_mask,
                    embedding_layer=model_ref.caption_model.word_embeddings,
                    pad_id=int(model_ref.pad_token_id),
                    bos_id=int(model_ref.bos_token_id),
                    eos_id=int(model_ref.eos_token_id),
                )
            else:
                phrase_loss = _safe_zero_like(ce_loss)

            if args.lambda_phrase_gen > 0 and aux.get("phrase_decoder_logits") is not None:
                if str(args.phrase_target_mode).lower() == "slot" and aux.get("phrase_slot_logits") is not None:
                    use_multiref = bool(
                        getattr(args, "phrase_slot_multiref_enable", 0)
                        and phrase_slot_ref_ids is not None
                        and phrase_slot_ref_mask is not None
                        and phrase_slot_ref_valid is not None
                    )
                    if use_multiref:
                        phrase_gen_loss = phrase_slot_multiref_loss(
                            model_ref=model_ref,
                            aux=aux,
                            phrase_slot_ref_ids=phrase_slot_ref_ids,
                            phrase_slot_ref_mask=phrase_slot_ref_mask,
                            phrase_slot_ref_valid=phrase_slot_ref_valid,
                            slot_token_weights=slot_token_weights,
                            args=args,
                            fallback=ce_loss,
                        )
                    else:
                        phrase_target = phrase_slot_ids[:, :, 1:].contiguous()
                        slot_token_loss = F.cross_entropy(
                            aux["phrase_slot_logits"].reshape(-1, aux["phrase_slot_logits"].size(-1)),
                            phrase_target.reshape(-1),
                            ignore_index=int(model_ref.pad_token_id),
                            reduction="none",
                        )
                        slot_token_loss = slot_token_loss.view_as(phrase_target)
                        slot_token_mask = phrase_target.ne(int(model_ref.pad_token_id))
                        slot_token_mask = slot_token_mask & phrase_slot_valid.unsqueeze(-1).bool()
                        if bool(slot_token_mask.any()):
                            slot_token_weight_mask = slot_token_mask.to(slot_token_loss.dtype)
                            if slot_token_weights is not None and slot_token_weights.numel() >= phrase_target.size(1):
                                slot_scale = slot_token_weights[: phrase_target.size(1)].view(1, -1, 1)
                                slot_token_weight_mask = slot_token_weight_mask * slot_scale.to(slot_token_loss.dtype)
                            denom = slot_token_weight_mask.sum().clamp_min(1.0)
                            phrase_gen_loss = (slot_token_loss * slot_token_weight_mask).sum() / denom
                        else:
                            phrase_gen_loss = _safe_zero_like(ce_loss)
                else:
                    phrase_target = phrase_ids[:, 1:].contiguous()
                    phrase_gen_loss = F.cross_entropy(
                        aux["phrase_decoder_logits"].reshape(-1, aux["phrase_decoder_logits"].size(-1)),
                        phrase_target.reshape(-1),
                        ignore_index=int(model_ref.pad_token_id),
                    )
            else:
                phrase_gen_loss = _safe_zero_like(ce_loss)

            if (
                lambda_phrase_pred_gen_active > 0
                and str(args.phrase_target_mode).lower() == "slot"
                and bool(args.phrase_condition_train_use_predicted)
                and aux.get("pred_phrase_slot_logits") is not None
            ):
                if (
                    getattr(args, "phrase_slot_multiref_enable", 0)
                    and phrase_slot_ref_ids is not None
                    and phrase_slot_ref_valid is not None
                ):
                    pred_phrase_gen_loss = phrase_slot_multiref_loss_from_logits(
                        model_ref=model_ref,
                        slot_logits=aux.get("pred_phrase_slot_logits"),
                        phrase_slot_ref_ids=phrase_slot_ref_ids,
                        phrase_slot_ref_valid=phrase_slot_ref_valid,
                        slot_token_weights=slot_token_weights,
                        args=args,
                        fallback=ce_loss,
                    )
                else:
                    pred_ref_ids = phrase_slot_ids.unsqueeze(2) if phrase_slot_ids is not None else None
                    pred_ref_valid = phrase_slot_valid.unsqueeze(-1) if phrase_slot_valid is not None else None
                    pred_phrase_gen_loss = phrase_slot_multiref_loss_from_logits(
                        model_ref=model_ref,
                        slot_logits=aux.get("pred_phrase_slot_logits"),
                        phrase_slot_ref_ids=pred_ref_ids,
                        phrase_slot_ref_valid=pred_ref_valid,
                        slot_token_weights=slot_token_weights,
                        args=args,
                        fallback=ce_loss,
                    )
            else:
                pred_phrase_gen_loss = _safe_zero_like(ce_loss)

            if (
                args.lambda_phrase_slot_presence > 0
                and str(args.phrase_target_mode).lower() == "slot"
                and aux.get("phrase_slot_presence_logits") is not None
            ):
                presence_kwargs = {}
                if (
                    slot_presence_pos_weights is not None
                    and slot_presence_pos_weights.numel() >= aux["phrase_slot_presence_logits"].size(1)
                ):
                    presence_kwargs["pos_weight"] = slot_presence_pos_weights[
                        : aux["phrase_slot_presence_logits"].size(1)
                    ].to(dtype=aux["phrase_slot_presence_logits"].dtype)
                phrase_slot_presence_loss = F.binary_cross_entropy_with_logits(
                    aux["phrase_slot_presence_logits"],
                    phrase_slot_valid.float(),
                    **presence_kwargs,
                )
            else:
                phrase_slot_presence_loss = _safe_zero_like(ce_loss)

            if (
                args.lambda_phrase_slot_div > 0
                and str(args.phrase_target_mode).lower() == "slot"
                and aux.get("phrase_slot_summary") is not None
            ):
                phrase_slot_div_loss = phrase_slot_diversity_loss(
                    slot_summary=aux["phrase_slot_summary"],
                    slot_valid=phrase_slot_valid,
                )
            else:
                phrase_slot_div_loss = _safe_zero_like(ce_loss)

            align_slot_summary = aux.get("phrase_slot_summary")
            align_slot_valid = aux.get("phrase_slot_valid")
            align_phrase_summary = aux.get("phrase_decoder_summary")
            align_phrase_valid = aux.get("phrase_decoder_valid")
            if bool(args.phrase_condition_train_use_predicted):
                if aux.get("pred_phrase_slot_summary") is not None:
                    align_slot_summary = aux.get("pred_phrase_slot_summary")
                if aux.get("pred_phrase_slot_valid") is not None:
                    align_slot_valid = aux.get("pred_phrase_slot_valid")
                if aux.get("pred_phrase_decoder_summary") is not None:
                    align_phrase_summary = aux.get("pred_phrase_decoder_summary")
                if aux.get("pred_phrase_decoder_valid") is not None:
                    align_phrase_valid = aux.get("pred_phrase_decoder_valid")

            if (
                args.lambda_phrase_ref_slot_align > 0
                and str(args.phrase_target_mode).lower() == "slot"
                and align_slot_summary is not None
                and aux.get("phrase_ref_bank_slot_summary") is not None
            ):
                phrase_ref_slot_align_loss = summary_alignment_loss(
                    pred_summary=align_slot_summary,
                    pred_valid=align_slot_valid,
                    target_summary=aux.get("phrase_ref_bank_slot_summary"),
                    target_valid=aux.get("phrase_ref_bank_slot_valid"),
                )
            else:
                phrase_ref_slot_align_loss = _safe_zero_like(ce_loss)

            if (
                args.lambda_phrase_ref_bridge > 0
                and align_phrase_summary is not None
                and aux.get("phrase_ref_bank_phrase_summary") is not None
            ):
                phrase_ref_bridge_loss = summary_alignment_loss(
                    pred_summary=align_phrase_summary,
                    pred_valid=align_phrase_valid,
                    target_summary=aux.get("phrase_ref_bank_phrase_summary"),
                    target_valid=aux.get("phrase_ref_bank_phrase_valid"),
                )
            else:
                phrase_ref_bridge_loss = _safe_zero_like(ce_loss)

            if (
                args.lambda_phrase_bridge > 0
                and bool(args.phrase_condition_enable)
                and bool(args.phrase_condition_train_use_predicted)
                and aux.get("pred_phrase_decoder_summary") is not None
                and aux.get("phrase_decoder_summary") is not None
            ):
                phrase_bridge_loss = phrase_bridge_consistency_loss(
                    pred_summary=aux.get("pred_phrase_decoder_summary"),
                    pred_valid=aux.get("pred_phrase_decoder_valid"),
                    teacher_summary=aux.get("phrase_decoder_summary"),
                    teacher_valid=aux.get("phrase_decoder_valid"),
                )
            else:
                phrase_bridge_loss = _safe_zero_like(ce_loss)

            if (
                args.lambda_phrase_slot_source_align > 0
                and str(args.phrase_target_mode).lower() == "slot"
                and aux.get("phrase_slot_source_weights") is not None
                and aux.get("phrase_slot_source_names")
            ):
                phrase_slot_source_align_loss = phrase_slot_source_alignment_loss(
                    slot_source_weights=aux.get("phrase_slot_source_weights"),
                    slot_valid=phrase_slot_valid,
                    source_names=list(aux.get("phrase_slot_source_names") or []),
                    phrase_slot_schema=args.phrase_slot_schema,
                )
            else:
                phrase_slot_source_align_loss = _safe_zero_like(ce_loss)

            if (
                args.lambda_phrase_slot_source_comp > 0
                and str(args.phrase_target_mode).lower() == "slot"
                and aux.get("phrase_slot_source_weights") is not None
                and aux.get("phrase_slot_source_names")
            ):
                phrase_slot_source_comp_loss = phrase_slot_source_competition_loss(
                    slot_source_weights=aux.get("phrase_slot_source_weights"),
                    slot_valid=phrase_slot_valid,
                    source_names=list(aux.get("phrase_slot_source_names") or []),
                    phrase_slot_schema=args.phrase_slot_schema,
                    margin=float(args.phrase_slot_source_comp_margin),
                )
            else:
                phrase_slot_source_comp_loss = _safe_zero_like(ce_loss)

            total_loss = (
                float(args.lambda_ce) * ce_loss
                + float(args.lambda_entity) * entity_loss
                + float(args.lambda_action) * action_loss
                + float(args.lambda_attr) * attr_loss
                + float(args.lambda_scene) * scene_loss
                + float(args.lambda_phrase) * phrase_loss
                + float(args.lambda_phrase_gen) * phrase_gen_loss
                + lambda_phrase_pred_gen_active * pred_phrase_gen_loss
                + float(args.lambda_phrase_slot_presence) * phrase_slot_presence_loss
                + float(args.lambda_phrase_slot_div) * phrase_slot_div_loss
                + float(args.lambda_phrase_ref_slot_align) * phrase_ref_slot_align_loss
                + float(args.lambda_phrase_ref_bridge) * phrase_ref_bridge_loss
                + float(args.lambda_phrase_bridge) * phrase_bridge_loss
                + float(args.lambda_phrase_slot_source_align) * phrase_slot_source_align_loss
                + float(args.lambda_phrase_slot_source_comp) * phrase_slot_source_comp_loss
            )
            loss = total_loss / accum

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
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            total_raw = float(total_loss.item())
            ce_raw = float(ce_loss.item())
            ppl = float(math.exp(ce_raw))
            acc = base.accuracy_from_logits(logits.detach(), target, pad_id=pad_id)
            ent_raw = float(entity_loss.item())
            act_raw = float(action_loss.item())
            attr_raw = float(attr_loss.item())
            scene_raw = float(scene_loss.item())
            phr_raw = float(phrase_loss.item())
            phr_gen_raw = float(phrase_gen_loss.item())
            phr_slot_presence_raw = float(phrase_slot_presence_loss.item())
            phr_slot_div_raw = float(phrase_slot_div_loss.item())
            phr_ref_slot_align_raw = float(phrase_ref_slot_align_loss.item())
            phr_ref_bridge_raw = float(phrase_ref_bridge_loss.item())
            phr_bridge_raw = float(phrase_bridge_loss.item())
            phr_pred_gen_raw = float(pred_phrase_gen_loss.item())
            phr_slot_source_align_raw = float(phrase_slot_source_align_loss.item())
            phr_slot_source_comp_raw = float(phrase_slot_source_comp_loss.item())
            if relation_detail_slot_idx is not None and relation_detail_slot_idx < phrase_slot_valid.size(1):
                detail_target = phrase_slot_valid[:, relation_detail_slot_idx].bool()
                relation_detail_stats["supervised_samples"] += float(detail_target.numel())
                relation_detail_stats["supervised_positive"] += float(detail_target.sum().item())

                detail_probs = aux.get("phrase_slot_presence_probs")
                if (
                    detail_probs is not None
                    and torch.is_tensor(detail_probs)
                    and detail_probs.dim() >= 2
                    and detail_probs.size(1) > relation_detail_slot_idx
                ):
                    relation_detail_stats["prob_sum"] += float(
                        detail_probs[:, relation_detail_slot_idx].detach().float().sum().item()
                    )

                detail_pred = aux.get("phrase_slot_presence_pred")
                if (
                    detail_pred is not None
                    and torch.is_tensor(detail_pred)
                    and detail_pred.dim() >= 2
                    and detail_pred.size(1) > relation_detail_slot_idx
                ):
                    detail_pred = detail_pred[:, relation_detail_slot_idx].detach().bool()
                    relation_detail_stats["predicted_positive"] += float(detail_pred.sum().item())
                    relation_detail_stats["true_positive"] += float((detail_pred & detail_target).sum().item())

                source_weights = aux.get("phrase_slot_source_weights")
                source_names = list(aux.get("phrase_slot_source_names") or [])
                if (
                    source_weights is not None
                    and torch.is_tensor(source_weights)
                    and source_weights.dim() >= 3
                    and source_weights.size(1) > relation_detail_slot_idx
                    and source_names
                ):
                    support_indices = [
                        source_idx
                        for source_idx, source_name in enumerate(source_names)
                        if str(source_name).strip().lower() in {"action", "attribute", "struct"}
                    ]
                    if support_indices:
                        detail_support = source_weights[:, relation_detail_slot_idx, support_indices].detach().float().sum(dim=-1)
                        relation_detail_stats["planner_support_sum"] += float(detail_support.sum().item())
                        relation_detail_stats["planner_support_majority"] += float((detail_support >= 0.5).sum().item())
                        relation_detail_stats["planner_support_samples"] += float(detail_support.numel())

        total_meter += total_raw
        ce_meter += ce_raw
        tok_meter += int((target != pad_id).sum().item())
        global_step += 1
        processed_steps += 1

        if (not ddp) or dist.get_rank() == 0:
            pbar.set_postfix(
                {
                    "total": f"{total_meter / max(1, it):.4f}",
                    "ce": f"{ce_meter / max(1, it):.4f}",
                    "acc": f"{acc:.3f}",
                    "ppl": f"{ppl:.2f}",
                }
            )
            if writer is not None:
                writer.add_scalar("train/total_loss", total_raw, global_step)
                writer.add_scalar("train/ce_loss", ce_raw, global_step)
                writer.add_scalar("train/entity_loss", ent_raw, global_step)
                writer.add_scalar("train/action_loss", act_raw, global_step)
                writer.add_scalar("train/attr_loss", attr_raw, global_step)
                writer.add_scalar("train/scene_loss", scene_raw, global_step)
                writer.add_scalar("train/phrase_loss", phr_raw, global_step)
                writer.add_scalar("train/phrase_gen_loss", phr_gen_raw, global_step)
                writer.add_scalar("train/pred_phrase_gen_loss", phr_pred_gen_raw, global_step)
                writer.add_scalar("train/phrase_slot_presence_loss", phr_slot_presence_raw, global_step)
                writer.add_scalar("train/phrase_slot_div_loss", phr_slot_div_raw, global_step)
                writer.add_scalar("train/phrase_ref_slot_align_loss", phr_ref_slot_align_raw, global_step)
                writer.add_scalar("train/phrase_ref_bridge_loss", phr_ref_bridge_raw, global_step)
                writer.add_scalar("train/phrase_bridge_loss", phr_bridge_raw, global_step)
                writer.add_scalar("train/phrase_slot_source_align_loss", phr_slot_source_align_raw, global_step)
                writer.add_scalar("train/phrase_slot_source_comp_loss", phr_slot_source_comp_raw, global_step)
                writer.add_scalar("train/acc", acc, global_step)
                writer.add_scalar("train/ppl", ppl, global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

    if max_train_steps > 0 and processed_steps < len(loader):
        print(f"[SmokeLimit] epoch={epoch}, processed_steps={processed_steps}, max_train_steps={max_train_steps}")
    avg_total = total_meter / max(1, processed_steps)
    avg_ce = ce_meter / max(1, processed_steps)
    avg_tok = tok_meter / max(1, processed_steps)
    diagnostics: Dict[str, float] = {}
    if relation_detail_slot_idx is not None:
        threshold_list = list(getattr(args, "phrase_slot_presence_thresholds", None) or [])
        relation_detail_threshold = float(args.phrase_slot_presence_threshold)
        if relation_detail_slot_idx < len(threshold_list):
            relation_detail_threshold = float(threshold_list[relation_detail_slot_idx])
        supervised_samples = max(1.0, relation_detail_stats["supervised_samples"])
        predicted_positive = relation_detail_stats["predicted_positive"]
        planner_support_samples = relation_detail_stats["planner_support_samples"]
        diagnostics = {
            "relation_detail_slot_index": float(relation_detail_slot_idx),
            "relation_detail_presence_threshold": float(relation_detail_threshold),
            "relation_detail_supervised_rate": float(relation_detail_stats["supervised_positive"] / supervised_samples),
            "relation_detail_prob_mean": float(relation_detail_stats["prob_sum"] / supervised_samples),
            "relation_detail_pred_active_rate": float(predicted_positive / supervised_samples),
            "relation_detail_precision": float(
                relation_detail_stats["true_positive"] / max(1.0, predicted_positive)
            ),
            "relation_detail_planner_support_mass": float(
                relation_detail_stats["planner_support_sum"] / max(1.0, planner_support_samples)
            ),
            "relation_detail_planner_support_majority_rate": float(
                relation_detail_stats["planner_support_majority"] / max(1.0, planner_support_samples)
            ),
        }
    return avg_total, avg_ce, avg_tok, global_step, diagnostics


def build_train_dataset(args):
    if args.dataset_type == "msrvtt":
        base_train = MSRVTT_FeaturesDataset(
            features_path=args.clip_global_vision_feats_path,
            json_path=args.annotations_path,
            split="train",
        )
    elif args.dataset_type == "msvd":
        base_train = MSVD_FeaturesDataset(
            features_path=args.clip_global_vision_feats_path,
            annotations_path=args.annotations_path,
            split="train",
        )
    else:
        raise ValueError(f"Unsupported dataset_type: {args.dataset_type}")

    train_dataset = StructuredCaptionDataset(
        base_dataset=base_train,
        structured_gt_path=args.structured_gt_path,
        phrase_max_len=args.phrase_max_len,
        phrase_fallback_to_caption=bool(args.phrase_fallback_to_caption),
        phrase_target_mode=args.phrase_target_mode,
        max_phrase_slots=args.max_phrase_slots,
        phrase_slot_max_len=args.phrase_slot_max_len,
        phrase_slot_schema=args.phrase_slot_schema,
        phrase_include_attr_units=bool(args.phrase_include_attr_units),
        phrase_include_scene_units=bool(args.phrase_include_scene_units),
        phrase_include_video_phrase_units=bool(args.phrase_include_video_phrase_units),
        phrase_include_video_attr_units=bool(args.phrase_include_video_attr_units),
        phrase_include_video_scene_units=bool(args.phrase_include_video_scene_units),
        phrase_video_phrase_min_support=args.phrase_video_phrase_min_support,
        phrase_video_phrase_max_units=args.phrase_video_phrase_max_units,
        phrase_slot_active_slot_types=str(getattr(args, "phrase_slot_active_slot_types", "")),
        phrase_slot_multiref_enable=bool(getattr(args, "phrase_slot_multiref_enable", 0)),
        phrase_slot_multiref_max_refs=int(getattr(args, "phrase_slot_multiref_max_refs", 0)),
        phrase_slot_family_sample_mode=str(getattr(args, "phrase_slot_family_sample_mode", "first")),
        phrase_slot_family_sample_seed=(
            int(getattr(args, "phrase_slot_family_sample_seed", -1))
            if int(getattr(args, "phrase_slot_family_sample_seed", -1)) >= 0
            else int(args.seed)
        ),
        phrase_slot_family_expand_mode=str(getattr(args, "phrase_slot_family_expand_mode", "none")),
    )
    train_dataset = base.maybe_wrap_with_visual_evidence(train_dataset, args, split="train")
    return train_dataset


def _resolve_phrase_unit_schedule_window(args) -> Tuple[int, int]:
    start_epoch = int(getattr(args, "phrase_attr_scene_units_start_epoch", 0))
    end_epoch = int(getattr(args, "phrase_attr_scene_units_end_epoch", 0))
    return start_epoch, end_epoch


def _resolve_epoch_phrase_unit_flags(args, epoch: int) -> Tuple[bool, bool]:
    include_attr_units = bool(getattr(args, "phrase_include_attr_units", 0))
    include_scene_units = bool(getattr(args, "phrase_include_scene_units", 0))
    start_epoch, end_epoch = _resolve_phrase_unit_schedule_window(args)
    if (start_epoch <= 0 and end_epoch <= 0) or (not include_attr_units and not include_scene_units):
        return include_attr_units, include_scene_units

    active = True
    if start_epoch > 0 and epoch < start_epoch:
        active = False
    if end_epoch > 0 and epoch > end_epoch:
        active = False
    if not active:
        return False, False
    return include_attr_units, include_scene_units


def _resolve_epoch_predicted_phrase_settings(args, epoch: int) -> Tuple[float, bool]:
    pred_gen_start_epoch = max(1, int(getattr(args, "phrase_pred_gen_start_epoch", 1)))
    pred_gen_lambda = float(getattr(args, "lambda_phrase_pred_gen", 0.0))
    current_pred_gen_lambda = pred_gen_lambda if epoch >= pred_gen_start_epoch else 0.0

    pred_detach = bool(getattr(args, "phrase_condition_pred_detach", 1))
    pred_detach_until_epoch = int(getattr(args, "phrase_condition_pred_detach_until_epoch", 0))
    if pred_detach_until_epoch > 0:
        pred_detach = epoch <= pred_detach_until_epoch

    return current_pred_gen_lambda, pred_detach


def _apply_epoch_phrase_unit_schedule(train_dataset, args, epoch: int, rank: int) -> Dict[str, float]:
    dataset = base.unwrap_dataset(train_dataset)
    if not isinstance(dataset, StructuredCaptionDataset):
        return {}

    dataset.set_epoch(epoch)
    include_attr_units, include_scene_units = _resolve_epoch_phrase_unit_flags(args, epoch)
    dataset.phrase_include_attr_units = include_attr_units
    dataset.phrase_include_scene_units = include_scene_units

    start_epoch, end_epoch = _resolve_phrase_unit_schedule_window(args)
    if rank == 0:
        if start_epoch > 0 or end_epoch > 0:
            start_text = str(start_epoch) if start_epoch > 0 else "-inf"
            end_text = str(end_epoch) if end_epoch > 0 else "+inf"
            print(
                f"[PhraseSchedule] epoch={epoch}, include_attr_units={int(include_attr_units)}, "
                f"include_scene_units={int(include_scene_units)}, window=[{start_text}, {end_text}]"
            )
        else:
            print(
                f"[PhraseSchedule] epoch={epoch}, include_attr_units={int(include_attr_units)}, "
                f"include_scene_units={int(include_scene_units)}, window=always"
            )

    return {
        "phrase_include_attr_units_effective": float(int(include_attr_units)),
        "phrase_include_scene_units_effective": float(int(include_scene_units)),
    }


def _apply_epoch_predicted_phrase_schedule(model, args, epoch: int, rank: int) -> Dict[str, float]:
    current_pred_gen_lambda, pred_detach = _resolve_epoch_predicted_phrase_settings(args, epoch)
    args._lambda_phrase_pred_gen_epoch = current_pred_gen_lambda

    model_ref = model.module if isinstance(model, DDP) else model
    if hasattr(model_ref, "phrase_condition_pred_detach"):
        model_ref.phrase_condition_pred_detach = bool(pred_detach)

    pred_gen_start_epoch = max(1, int(getattr(args, "phrase_pred_gen_start_epoch", 1)))
    pred_detach_until_epoch = int(getattr(args, "phrase_condition_pred_detach_until_epoch", 0))
    if rank == 0:
        print(
            f"[PhrasePredSchedule] epoch={epoch}, "
            f"lambda_phrase_pred_gen_active={current_pred_gen_lambda:.4f}, "
            f"pred_gen_start_epoch={pred_gen_start_epoch}, "
            f"condition_pred_detach_active={int(bool(pred_detach))}, "
            f"condition_pred_detach_until_epoch={pred_detach_until_epoch}"
        )

    return {
        "lambda_phrase_pred_gen_active": float(current_pred_gen_lambda),
        "phrase_condition_pred_detach_active": float(int(bool(pred_detach))),
    }


def main_worker(local_rank: int, args):
    ddp = bool(args.ddp)
    if ddp and base.is_torchrun_env():
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        device = base.setup_ddp_env(local_rank, world_size, backend="nccl")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        rank = 0

    out_dir = base.resolve_run_dir(args)
    if (not ddp) or rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "checkpoints").mkdir(exist_ok=True)
        with (out_dir / "args.json").open("w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)

    base.set_seed(args.seed + rank)

    train_dataset = build_train_dataset(args)
    configure_phrase_slot_reweight(args=args, train_dataset=train_dataset, out_dir=out_dir, rank=rank)

    if ddp and base.is_torchrun_env():
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

    lexical_anchor_kwargs = {}
    anchor_dataset = base.unwrap_dataset(train_dataset)
    if (
        bool(getattr(args, "phrase_slot_decode_anchor_enable", 0))
        and isinstance(anchor_dataset, StructuredCaptionDataset)
        and str(args.phrase_target_mode).strip().lower() == "slot"
    ):
        lexical_anchor_kwargs = build_phrase_lexical_anchor_kwargs(
            tokenizer=anchor_dataset.tokenizer,
            structured_payload_or_videos=anchor_dataset.structured_videos,
            entity_vocab=anchor_dataset.entity_vocab,
            action_vocab=anchor_dataset.action_vocab,
            attribute_vocab=anchor_dataset.attribute_vocab,
            scene_vocab=anchor_dataset.scene_vocab,
            phrase_slot_schema=str(args.phrase_slot_schema),
            max_phrase_slots=int(args.max_phrase_slots),
            family_topk_tokens=int(getattr(args, "phrase_slot_decode_anchor_family_topk", 64)),
            family_min_count=int(getattr(args, "phrase_slot_decode_anchor_family_min_count", 2)),
        )
        if rank == 0:
            family_anchor_map = lexical_anchor_kwargs.get("slot_family_anchor_token_ids", {})
            family_anchor_sizes = {
                str(slot_type): len(token_ids)
                for slot_type, token_ids in family_anchor_map.items()
            }
            print(
                "[PhraseDecodeAnchor] "
                f"enabled=1 topk={int(getattr(args, 'phrase_slot_decode_anchor_topk', 8))} "
                f"family_sizes={family_anchor_sizes}"
            )

    model = StructuredCaptionModel(
        entity_dim=train_dataset.entity_dim,
        action_dim=train_dataset.action_dim,
        attribute_dim=train_dataset.attribute_dim,
        scene_dim=train_dataset.scene_dim,
        prior_dropout=args.prior_dropout,
        struct_condition=bool(args.struct_condition),
        struct_condition_scale=args.struct_condition_scale,
        struct_condition_query_bridge_enable=bool(
            getattr(args, "struct_condition_query_bridge_enable", 0)
        ),
        struct_condition_query_bridge_num_queries=int(
            getattr(args, "struct_condition_query_bridge_num_queries", 4)
        ),
        struct_condition_query_bridge_scale=float(
            getattr(args, "struct_condition_query_bridge_scale", 0.15)
        ),
        struct_condition_query_bridge_memory_enable=bool(
            getattr(args, "struct_condition_query_bridge_memory_enable", 0)
        ),
        struct_condition_query_bridge_memory_scale=float(
            getattr(args, "struct_condition_query_bridge_memory_scale", 0.15)
        ),
        struct_condition_query_bridge_hidden_enable=bool(
            getattr(args, "struct_condition_query_bridge_hidden_enable", 0)
        ),
        struct_condition_query_bridge_hidden_scale=float(
            getattr(args, "struct_condition_query_bridge_hidden_scale", 0.15)
        ),
        phrase_decoder_enable=bool(args.phrase_decoder_enable),
        phrase_condition_enable=bool(args.phrase_condition_enable),
        phrase_condition_slot_aware_enable=bool(getattr(args, "phrase_condition_slot_aware_enable", 0)),
        phrase_condition_slot_selective_enable=bool(
            getattr(args, "phrase_condition_slot_selective_enable", 0)
        ),
        phrase_condition_slot_residual_enable=bool(
            getattr(args, "phrase_condition_slot_residual_enable", 0)
        ),
        phrase_condition_pred_detach=bool(getattr(args, "phrase_condition_pred_detach", 1)),
        phrase_condition_teacher_source=str(getattr(args, "phrase_condition_teacher_source", "single_ref")),
        phrase_decoder_layers=args.phrase_decoder_layers,
        phrase_condition_scale=args.phrase_condition_scale,
        phrase_condition_aux_scale=float(getattr(args, "phrase_condition_aux_scale", 0.15)),
        phrase_condition_slot_residual_scale=float(
            getattr(args, "phrase_condition_slot_residual_scale", 0.15)
        ),
        phrase_condition_core_slot_types=str(getattr(args, "phrase_condition_core_slot_types", "")),
        phrase_condition_aux_slot_types=str(getattr(args, "phrase_condition_aux_slot_types", "")),
        phrase_condition_slot_residual_slot_types=str(
            getattr(args, "phrase_condition_slot_residual_slot_types", "")
        ),
        phrase_condition_family_bridge_enable=bool(
            getattr(args, "phrase_condition_family_bridge_enable", 0)
        ),
        phrase_condition_family_bridge_scale=float(
            getattr(args, "phrase_condition_family_bridge_scale", 0.20)
        ),
        phrase_condition_candidate_bias_enable=bool(
            getattr(args, "phrase_condition_candidate_bias_enable", 0)
        ),
        phrase_condition_candidate_bias_scale=float(
            getattr(args, "phrase_condition_candidate_bias_scale", 0.10)
        ),
        phrase_condition_candidate_topk=int(getattr(args, "phrase_condition_candidate_topk", 12)),
        phrase_condition_candidate_slot_types=str(
            getattr(args, "phrase_condition_candidate_slot_types", "")
        ),
        phrase_condition_query_bridge_enable=bool(
            getattr(args, "phrase_condition_query_bridge_enable", 0)
        ),
        phrase_condition_query_bridge_num_queries=int(
            getattr(args, "phrase_condition_query_bridge_num_queries", 4)
        ),
        phrase_condition_query_bridge_scale=float(
            getattr(args, "phrase_condition_query_bridge_scale", 0.15)
        ),
        phrase_condition_train_use_predicted=bool(args.phrase_condition_train_use_predicted),
        phrase_gen_max_len=args.phrase_gen_max_len,
        phrase_memory_mode=args.phrase_memory_mode,
        phrase_target_mode=args.phrase_target_mode,
        phrase_slot_schema=args.phrase_slot_schema,
        max_phrase_slots=args.max_phrase_slots,
        phrase_slot_max_len=args.phrase_slot_max_len,
        phrase_slot_planner_enable=bool(args.phrase_slot_planner_enable),
        phrase_slot_planner_flow_enable=bool(getattr(args, "phrase_slot_planner_flow_enable", 0)),
        phrase_slot_planner_flow_scale=float(getattr(args, "phrase_slot_planner_flow_scale", 0.20)),
        phrase_slot_planner_flow_slot_types=str(
            getattr(args, "phrase_slot_planner_flow_slot_types", "")
        ),
        phrase_slot_guidance_enable=bool(getattr(args, "phrase_slot_guidance_enable", 0)),
        phrase_slot_role_anchor_enable=bool(getattr(args, "phrase_slot_role_anchor_enable", 0)),
        phrase_slot_role_anchor_topk=int(getattr(args, "phrase_slot_role_anchor_topk", 4)),
        phrase_slot_role_anchor_scale=float(getattr(args, "phrase_slot_role_anchor_scale", 1.0)),
        phrase_slot_role_anchor_slot_types=str(getattr(args, "phrase_slot_role_anchor_slot_types", "")),
        phrase_slot_decode_anchor_enable=bool(getattr(args, "phrase_slot_decode_anchor_enable", 0)),
        phrase_slot_decode_anchor_topk=int(getattr(args, "phrase_slot_decode_anchor_topk", 8)),
        phrase_slot_decode_anchor_scale=float(getattr(args, "phrase_slot_decode_anchor_scale", 1.0)),
        phrase_slot_decode_anchor_early_scale=float(
            getattr(args, "phrase_slot_decode_anchor_early_scale", 1.25)
        ),
        phrase_slot_decode_anchor_family_scale=float(
            getattr(args, "phrase_slot_decode_anchor_family_scale", 0.75)
        ),
        phrase_slot_decode_anchor_stopword_penalty=float(
            getattr(args, "phrase_slot_decode_anchor_stopword_penalty", 0.75)
        ),
        phrase_slot_decode_anchor_stopword_steps=int(
            getattr(args, "phrase_slot_decode_anchor_stopword_steps", 2)
        ),
        phrase_slot_decode_anchor_debug_topk=int(
            getattr(args, "phrase_slot_decode_anchor_debug_topk", 8)
        ),
        phrase_slot_presence_enable=bool(args.phrase_slot_presence_enable),
        phrase_slot_presence_support_enable=bool(getattr(args, "phrase_slot_presence_support_enable", 0)),
        phrase_slot_presence_evidence_enable=bool(getattr(args, "phrase_slot_presence_evidence_enable", 0)),
        phrase_slot_presence_context_slot_types=str(
            getattr(args, "phrase_slot_presence_context_slot_types", "")
        ),
        phrase_slot_presence_threshold=args.phrase_slot_presence_threshold,
        phrase_slot_presence_thresholds=getattr(args, "phrase_slot_presence_thresholds", None),
        phrase_slot_active_slot_types=str(getattr(args, "phrase_slot_active_slot_types", "")),
        prior_head_type=str(getattr(args, "prior_head_type", "simple")),
        prior_head_num_heads=int(getattr(args, "prior_head_num_heads", 8)),
        prior_head_hidden_dim=int(getattr(args, "prior_head_hidden_dim", 2048)),
        prior_head_num_blocks=int(getattr(args, "prior_head_num_blocks", 4)),
        prior_head_num_clusters=int(getattr(args, "prior_head_num_clusters", 16)),
        prior_head_expansion=int(getattr(args, "prior_head_expansion", 2)),
        prior_head_groups=int(getattr(args, "prior_head_groups", 8)),
        aux_visual_enable=bool(getattr(args, "aux_visual_enable", 0)),
        aux_raw_global_enable=bool(getattr(args, "aux_raw_global_enable", 0)),
        aux_patch_enable=bool(getattr(args, "aux_patch_enable", 0)),
        aux_visual_raw_global_dim=int(getattr(args, "aux_visual_raw_global_dim", 512)),
        aux_visual_patch_dim=int(getattr(args, "aux_visual_patch_dim", 768)),
        aux_visual_prior_scale=float(getattr(args, "aux_visual_prior_scale", 0.15)),
        aux_visual_struct_scale=float(getattr(args, "aux_visual_struct_scale", 0.10)),
        aux_visual_memory_scale=float(getattr(args, "aux_visual_memory_scale", 0.10)),
        phrase_progress_enable=bool(getattr(args, "phrase_progress_enable", 0)),
        phrase_progress_memory_scale=float(getattr(args, "phrase_progress_memory_scale", 0.10)),
        phrase_progress_source_scale=float(getattr(args, "phrase_progress_source_scale", 0.10)),
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
        **lexical_anchor_kwargs,
    ).to(device)

    if args.init_model_ckpt:
        load_initial_caption_weights(
            model,
            args.init_model_ckpt,
            skip_structured_vocab_modules=False,
        )

    if args.init_caption_ckpt:
        load_initial_caption_weights(
            model,
            args.init_caption_ckpt,
            skip_structured_vocab_modules=bool(getattr(args, "init_skip_structured_vocab_modules", 0)),
        )

    training_stage_name = str(getattr(args, "training_stage", "joint")).strip().lower() or "joint"
    if training_stage_name in {"joint", "full", "none"}:
        # Optional low-risk finetune mode: train only selected heads while keeping decoder fixed.
        if bool(args.freeze_caption_model):
            _set_requires_grad(model.caption_model, False)

        if bool(args.freeze_entity_action_heads):
            _set_requires_grad(model.entity_prior_head, False)
            _set_requires_grad(model.action_prior_head, False)
            _set_requires_grad(model.entity_to_context, False)
            _set_requires_grad(model.action_to_context, False)

        if bool(args.freeze_struct_condition):
            _set_requires_grad(model.entity_to_context, False)
            _set_requires_grad(model.action_to_context, False)
            _set_requires_grad(model.attribute_to_context, False)
            _set_requires_grad(model.scene_to_context, False)
            _set_requires_grad(model.condition_gate, False)
            _set_requires_grad(model.condition_norm, False)
            _set_requires_grad(model.condition_post_norm, False)
            if getattr(model, "struct_scale", None) is not None:
                model.struct_scale.requires_grad = False
            if getattr(model, "attribute_context_scale", None) is not None:
                model.attribute_context_scale.requires_grad = False
            if getattr(model, "scene_context_scale", None) is not None:
                model.scene_context_scale.requires_grad = False
    else:
        if bool(args.freeze_caption_model or args.freeze_entity_action_heads or args.freeze_struct_condition):
            print(
                f"[Stage] training_stage={training_stage_name} overrides "
                "freeze_caption_model / freeze_entity_action_heads / freeze_struct_condition."
            )
        applied_stage = _apply_training_stage_profile(model, args)
        print(f"[Stage] applied training profile: {applied_stage}")

    if ddp and base.is_torchrun_env():
        model = DDP(
            model,
            device_ids=[int(os.environ["LOCAL_RANK"])],
            output_device=int(os.environ["LOCAL_RANK"]),
            find_unused_parameters=False,
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found after freeze_* options.")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    scaler = GradScaler(enabled=bool(args.amp))

    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        if args.scheduler == "none":
            return 1.0
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        min_lr_ratio = args.min_lr / args.lr
        if args.scheduler == "cosine":
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        if args.scheduler == "linear":
            return max(min_lr_ratio, 1.0 - progress)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if (not ddp) or rank == 0:
        entity_caption_pos_weight, entity_video_only_pos_weight = _resolve_prior_weight_pair(args, "entity")
        action_caption_pos_weight, action_video_only_pos_weight = _resolve_prior_weight_pair(args, "action")
        attr_caption_pos_weight, attr_video_only_pos_weight = _resolve_prior_weight_pair(args, "attr")
        scene_caption_pos_weight, scene_video_only_pos_weight = _resolve_prior_weight_pair(args, "scene")
        print(
            f"[Structured] entity_dim={train_dataset.entity_dim}, action_dim={train_dataset.action_dim}, "
            f"attribute_dim={train_dataset.attribute_dim}, scene_dim={train_dataset.scene_dim}, "
            f"lambda_ce={args.lambda_ce}, lambda_entity={args.lambda_entity}, lambda_action={args.lambda_action}, "
            f"lambda_attr={args.lambda_attr}, lambda_scene={args.lambda_scene}, "
            f"prior_caption_pos_weight={float(getattr(args, 'prior_caption_pos_weight', 1.0)):.3f}, "
            f"prior_video_only_pos_weight={float(getattr(args, 'prior_video_only_pos_weight', 1.0)):.3f}, "
            f"prior_loss_type={str(getattr(args, 'prior_loss_type', 'bce'))}, "
            f"prior_asl=(gn={float(getattr(args, 'prior_asl_gamma_neg', 4.0)):.3f}, "
            f"gp={float(getattr(args, 'prior_asl_gamma_pos', 1.0)):.3f}, "
            f"clip={float(getattr(args, 'prior_asl_clip', 0.05)):.3f}), "
            f"entity_prior_pair=({entity_caption_pos_weight:.3f},{entity_video_only_pos_weight:.3f}), "
            f"action_prior_pair=({action_caption_pos_weight:.3f},{action_video_only_pos_weight:.3f}), "
            f"attr_prior_pair=({attr_caption_pos_weight:.3f},{attr_video_only_pos_weight:.3f}), "
            f"scene_prior_pair=({scene_caption_pos_weight:.3f},{scene_video_only_pos_weight:.3f}), "
            f"lambda_phrase={args.lambda_phrase}, lambda_phrase_gen={args.lambda_phrase_gen}, "
            f"lambda_phrase_pred_gen={args.lambda_phrase_pred_gen}, "
            f"lambda_phrase_slot_presence={args.lambda_phrase_slot_presence}, "
            f"lambda_phrase_slot_div={args.lambda_phrase_slot_div}, "
            f"lambda_phrase_ref_slot_align={args.lambda_phrase_ref_slot_align}, "
            f"lambda_phrase_ref_bridge={args.lambda_phrase_ref_bridge}, "
            f"lambda_phrase_bridge={args.lambda_phrase_bridge}, "
            f"lambda_phrase_slot_source_align={args.lambda_phrase_slot_source_align}, "
            f"lambda_phrase_slot_source_comp={args.lambda_phrase_slot_source_comp}, "
            f"phrase_slot_source_comp_margin={args.phrase_slot_source_comp_margin}"
        )
        print(
            f"[Phrase] decoder_enable={int(bool(args.phrase_decoder_enable))}, "
            f"condition_enable={int(bool(args.phrase_condition_enable))}, "
            f"condition_slot_aware_enable={int(bool(getattr(args, 'phrase_condition_slot_aware_enable', 0)))}, "
            f"condition_slot_selective_enable={int(bool(getattr(args, 'phrase_condition_slot_selective_enable', 0)))}, "
            f"condition_slot_residual_enable={int(bool(getattr(args, 'phrase_condition_slot_residual_enable', 0)))}, "
            f"condition_family_bridge_enable={int(bool(getattr(args, 'phrase_condition_family_bridge_enable', 0)))}, "
            f"condition_candidate_bias_enable={int(bool(getattr(args, 'phrase_condition_candidate_bias_enable', 0)))}, "
            f"condition_train_source={'predicted_detach' if bool(args.phrase_condition_train_use_predicted) else 'teacher'}, "
            f"condition_teacher_source={str(getattr(args, 'phrase_condition_teacher_source', 'single_ref'))}, "
            f"decoder_layers={int(args.phrase_decoder_layers)}, "
            f"condition_scale={float(args.phrase_condition_scale):.3f}, "
            f"condition_aux_scale={float(getattr(args, 'phrase_condition_aux_scale', 0.15)):.3f}, "
            f"condition_slot_residual_scale={float(getattr(args, 'phrase_condition_slot_residual_scale', 0.15)):.3f}, "
            f"condition_family_bridge_scale={float(getattr(args, 'phrase_condition_family_bridge_scale', 0.20)):.3f}, "
            f"condition_candidate_bias_scale={float(getattr(args, 'phrase_condition_candidate_bias_scale', 0.10)):.3f}, "
            f"condition_candidate_topk={int(getattr(args, 'phrase_condition_candidate_topk', 12))}, "
            f"condition_core_slot_types={str(getattr(args, 'phrase_condition_core_slot_types', ''))}, "
            f"condition_aux_slot_types={str(getattr(args, 'phrase_condition_aux_slot_types', ''))}, "
            f"condition_slot_residual_slot_types={str(getattr(args, 'phrase_condition_slot_residual_slot_types', ''))}, "
            f"condition_candidate_slot_types={str(getattr(args, 'phrase_condition_candidate_slot_types', ''))}, "
            f"gen_max_len={int(args.phrase_gen_max_len)}, "
            f"memory_mode={str(args.phrase_memory_mode)}, "
            f"target_mode={str(args.phrase_target_mode)}, "
            f"slot_schema={str(args.phrase_slot_schema)}, "
            f"include_attr_units={int(bool(args.phrase_include_attr_units))}, "
            f"include_scene_units={int(bool(args.phrase_include_scene_units))}, "
            f"include_video_phrase_units={int(bool(getattr(args, 'phrase_include_video_phrase_units', 0)))}, "
            f"include_video_attr_units={int(bool(getattr(args, 'phrase_include_video_attr_units', 0)))}, "
            f"include_video_scene_units={int(bool(getattr(args, 'phrase_include_video_scene_units', 0)))}, "
            f"video_phrase_min_support={int(getattr(args, 'phrase_video_phrase_min_support', 2))}, "
            f"video_phrase_max_units={int(getattr(args, 'phrase_video_phrase_max_units', 4))}, "
            f"max_slots={int(args.max_phrase_slots)}, "
            f"slot_max_len={int(args.phrase_slot_max_len)}, "
            f"slot_planner_enable={int(bool(args.phrase_slot_planner_enable))}, "
            f"slot_planner_flow_enable={int(bool(getattr(args, 'phrase_slot_planner_flow_enable', 0)))}, "
            f"slot_planner_flow_scale={float(getattr(args, 'phrase_slot_planner_flow_scale', 0.20)):.3f}, "
            f"slot_planner_flow_slot_types={str(getattr(args, 'phrase_slot_planner_flow_slot_types', ''))}, "
            f"slot_guidance_enable={int(bool(getattr(args, 'phrase_slot_guidance_enable', 0)))}, "
            f"slot_role_anchor_enable={int(bool(getattr(args, 'phrase_slot_role_anchor_enable', 0)))}, "
            f"slot_role_anchor_topk={int(getattr(args, 'phrase_slot_role_anchor_topk', 4))}, "
            f"slot_role_anchor_scale={float(getattr(args, 'phrase_slot_role_anchor_scale', 1.0)):.3f}, "
            f"slot_role_anchor_slot_types={str(getattr(args, 'phrase_slot_role_anchor_slot_types', ''))}, "
            f"slot_presence_enable={int(bool(args.phrase_slot_presence_enable))}, "
            f"slot_presence_support_enable={int(bool(getattr(args, 'phrase_slot_presence_support_enable', 0)))}, "
            f"slot_presence_evidence_enable={int(bool(getattr(args, 'phrase_slot_presence_evidence_enable', 0)))}, "
            f"slot_presence_context_slot_types={str(getattr(args, 'phrase_slot_presence_context_slot_types', ''))}, "
            f"slot_active_slot_types={str(getattr(args, 'phrase_slot_active_slot_types', ''))}, "
            f"slot_presence_threshold={float(args.phrase_slot_presence_threshold):.3f}, "
            f"slot_presence_calibration_slot_types={str(getattr(args, 'phrase_slot_presence_calibration_slot_types', ''))}, "
            f"slot_reweight_enable={int(bool(args.phrase_slot_reweight_enable))}, "
            f"slot_reweight_power={float(args.phrase_slot_reweight_power):.3f}, "
            f"slot_reweight_clip=[{float(args.phrase_slot_reweight_min):.3f}, {float(args.phrase_slot_reweight_max):.3f}], "
            f"slot_multiref_enable={int(bool(getattr(args, 'phrase_slot_multiref_enable', 0)))}, "
            f"slot_multiref_max_refs={int(getattr(args, 'phrase_slot_multiref_max_refs', 0))}, "
            f"slot_multiref_reduce={str(getattr(args, 'phrase_slot_multiref_reduce', 'mean'))}, "
            f"slot_multiref_gain={float(getattr(args, 'phrase_slot_multiref_gain', 1.0)):.3f}, "
            f"slot_multiref_chunk_size={int(getattr(args, 'phrase_slot_multiref_chunk_size', 4))}, "
            f"slot_pred_multiref_reduce={str(getattr(args, 'phrase_slot_pred_multiref_reduce', 'inherit'))}, "
            f"slot_pred_multiref_softmin_temp={float(getattr(args, 'phrase_slot_pred_multiref_softmin_temp', 1.0)):.3f}, "
            f"slot_family_expand_mode={str(getattr(args, 'phrase_slot_family_expand_mode', 'none'))}, "
            f"slot_family_sample_mode={str(getattr(args, 'phrase_slot_family_sample_mode', 'first'))}, "
            f"slot_family_sample_seed="
            f"{(int(getattr(args, 'phrase_slot_family_sample_seed', -1)) if int(getattr(args, 'phrase_slot_family_sample_seed', -1)) >= 0 else int(args.seed))}, "
            f"train_samples={len(train_dataset)}, "
            f"base_train_samples={int(getattr(train_dataset, 'base_sample_count', len(train_dataset)))}, "
            f"expanded_train_extra={int(getattr(train_dataset, 'expanded_sample_extra', 0))}, "
            f"max_train_steps_per_epoch={int(getattr(args, 'max_train_steps_per_epoch', 0))}, "
            f"prior_head_type={str(getattr(args, 'prior_head_type', 'simple'))}, "
            f"prior_head_num_heads={int(getattr(args, 'prior_head_num_heads', 8))}, "
            f"prior_head_hidden_dim={int(getattr(args, 'prior_head_hidden_dim', 2048))}, "
            f"prior_head_num_blocks={int(getattr(args, 'prior_head_num_blocks', 4))}, "
            f"prior_head_num_clusters={int(getattr(args, 'prior_head_num_clusters', 16))}, "
            f"prior_head_expansion={int(getattr(args, 'prior_head_expansion', 2))}, "
            f"prior_head_groups={int(getattr(args, 'prior_head_groups', 8))}, "
            f"attr_scene_schedule=[{int(getattr(args, 'phrase_attr_scene_units_start_epoch', 0))}, "
            f"{int(getattr(args, 'phrase_attr_scene_units_end_epoch', 0))}], "
            f"fallback_to_caption={int(bool(args.phrase_fallback_to_caption))}"
        )
        print(
            f"[AuxVisual] enable={int(bool(getattr(args, 'aux_visual_enable', 0)))}, "
            f"raw_global_enable={int(bool(getattr(args, 'aux_raw_global_enable', 0)))}, "
            f"patch_enable={int(bool(getattr(args, 'aux_patch_enable', 0)))}, "
            f"raw_global_feats_path={str(getattr(args, 'aux_raw_global_feats_path', ''))}, "
            f"patch_root={str(getattr(args, 'aux_patch_root', ''))}, "
            f"patch_block={int(getattr(args, 'aux_patch_block', 6))}, "
            f"prior_scale={float(getattr(args, 'aux_visual_prior_scale', 0.15)):.3f}, "
            f"struct_scale={float(getattr(args, 'aux_visual_struct_scale', 0.10)):.3f}, "
            f"memory_scale={float(getattr(args, 'aux_visual_memory_scale', 0.10)):.3f}"
        )
        print(
            f"[PhraseProgress] enable={int(bool(getattr(args, 'phrase_progress_enable', 0)))}, "
            f"memory_scale={float(getattr(args, 'phrase_progress_memory_scale', 0.10)):.3f}, "
            f"source_scale={float(getattr(args, 'phrase_progress_source_scale', 0.10)):.3f}"
        )
        if getattr(args, "phrase_slot_presence_thresholds", None):
            threshold_pairs = ", ".join(
                f"{str(spec.get('slot_type', idx))}={float(args.phrase_slot_presence_thresholds[idx]):.3f}"
                for idx, spec in enumerate(
                    train_dataset.get_phrase_slot_type_specs(
                        max_phrase_slots=int(args.max_phrase_slots),
                        phrase_slot_schema=str(args.phrase_slot_schema),
                    )[: len(args.phrase_slot_presence_thresholds)]
                )
            )
            print(
                f"[PhrasePresenceThresholds] mode={str(args.phrase_slot_presence_calibration_mode)} "
                f"per_slot={threshold_pairs}"
            )
        print(
            f"[Freeze] caption_model={int(bool(args.freeze_caption_model))}, "
            f"entity_action_heads={int(bool(args.freeze_entity_action_heads))}, "
            f"struct_condition={int(bool(args.freeze_struct_condition))}, "
            f"trainable_params={sum(p.numel() for p in trainable_params)}"
        )
        print(
            f"[Scheduler] {args.scheduler}, warmup_epochs={args.warmup_epochs}, "
            f"warmup_steps={warmup_steps}, total_steps={total_steps}"
        )

    writer = None
    if (not ddp) or rank == 0:
        writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    global_step = 0
    epoch_metrics: List[dict] = []
    step_idx: List[int] = []
    loss_steps: List[float] = []

    try:
        for epoch in range(1, args.epochs + 1):
            epoch_phrase_schedule = _apply_epoch_phrase_unit_schedule(
                train_dataset=train_dataset,
                args=args,
                epoch=epoch,
                rank=rank,
            )
            epoch_pred_schedule = _apply_epoch_predicted_phrase_schedule(
                model=model,
                args=args,
                epoch=epoch,
                rank=rank,
            )
            avg_total, avg_ce, avg_tok, global_step, relation_detail_diag = train_one_epoch_structured(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                device=device,
                epoch=epoch,
                args=args,
                ddp=ddp and base.is_torchrun_env(),
                pad_id=0,
                global_step_start=global_step,
                writer=writer,
            )

            if (not ddp) or dist.get_rank() == 0:
                ppl_epoch = float(math.exp(avg_ce))
                epoch_record = {
                    "epoch": epoch,
                    "avg_total_loss": float(avg_total),
                    "avg_ce_loss": float(avg_ce),
                    "ppl": ppl_epoch,
                    "avg_tokens_per_step": float(avg_tok),
                    "time": time.time(),
                }
                epoch_record.update(epoch_phrase_schedule)
                epoch_record.update(epoch_pred_schedule)
                epoch_record.update(relation_detail_diag)
                epoch_metrics.append(epoch_record)

                step_idx.append(global_step)
                loss_steps.append(avg_ce)
                base.save_curves(step_idx, loss_steps, out_dir, ma_window=args.ma_window)

                if writer is not None:
                    writer.add_scalar("train_epoch/avg_total_loss", avg_total, epoch)
                    writer.add_scalar("train_epoch/avg_ce_loss", avg_ce, epoch)
                    writer.add_scalar("train_epoch/ppl", ppl_epoch, epoch)
                    if relation_detail_diag:
                        writer.add_scalar(
                            "train_epoch/relation_detail_supervised_rate",
                            relation_detail_diag["relation_detail_supervised_rate"],
                            epoch,
                        )
                        writer.add_scalar(
                            "train_epoch/relation_detail_pred_active_rate",
                            relation_detail_diag["relation_detail_pred_active_rate"],
                            epoch,
                        )
                        writer.add_scalar(
                            "train_epoch/relation_detail_precision",
                            relation_detail_diag["relation_detail_precision"],
                            epoch,
                        )
                        writer.add_scalar(
                            "train_epoch/relation_detail_prob_mean",
                            relation_detail_diag["relation_detail_prob_mean"],
                            epoch,
                        )
                        writer.add_scalar(
                            "train_epoch/relation_detail_planner_support_mass",
                            relation_detail_diag["relation_detail_planner_support_mass"],
                            epoch,
                        )

                model_for_eval = model if not isinstance(model, DDP) else model.module
                if relation_detail_diag:
                    print(
                        f"[RelationDetail] epoch {epoch}: "
                        f"threshold={relation_detail_diag['relation_detail_presence_threshold']:.3f}, "
                        f"supervised_rate={relation_detail_diag['relation_detail_supervised_rate']:.4f}, "
                        f"prob_mean={relation_detail_diag['relation_detail_prob_mean']:.4f}, "
                        f"pred_active_rate={relation_detail_diag['relation_detail_pred_active_rate']:.4f}, "
                        f"precision={relation_detail_diag['relation_detail_precision']:.4f}, "
                        f"planner_support_mass={relation_detail_diag['relation_detail_planner_support_mass']:.4f}, "
                        f"planner_support_majority_rate="
                        f"{relation_detail_diag['relation_detail_planner_support_majority_rate']:.4f}"
                    )
                val_scores = base.evaluate_on_val(model_for_eval, device=device, args=args, out_dir=out_dir, epoch=epoch)
                print(f"[Val] epoch {epoch}: VAL_LOSS={val_scores['VAL_LOSS']:.4f}")
                if writer is not None:
                    writer.add_scalar("val/epoch_loss", val_scores["VAL_LOSS"], epoch)

                current_lr = optimizer.param_groups[0]["lr"]
                base.append_jsonl(
                    out_dir / "val_metrics.jsonl",
                    {"epoch": epoch, "VAL_LOSS": float(val_scores["VAL_LOSS"]), "time": time.time()},
                )

                test_scores = base.evaluate_on_test(model_for_eval, device=device, args=args, out_dir=out_dir, epoch=epoch)
                print(
                    f"[Test] epoch {epoch}: "
                    f"BLEU-4={test_scores['BLEU-4']:.2f}, "
                    f"CIDEr={test_scores['CIDEr']:.2f}, "
                    f"ROUGE_L={test_scores['ROUGE_L']:.2f}, "
                    f"METEOR={test_scores['METEOR']:.2f}, "
                    f"TEST_LOSS={test_scores['TEST_LOSS']:.4f}"
                )

                if writer is not None:
                    writer.add_scalar("test/BLEU-4", test_scores["BLEU-4"], epoch)
                    writer.add_scalar("test/ROUGE_L", test_scores["ROUGE_L"], epoch)
                    writer.add_scalar("test/CIDEr", test_scores["CIDEr"], epoch)
                    writer.add_scalar("test/METEOR", test_scores["METEOR"], epoch)
                    writer.add_scalar("test/epoch_loss", test_scores["TEST_LOSS"], epoch)

                base.append_jsonl(
                    out_dir / "test_metrics.jsonl",
                    {"epoch": epoch, **{k: float(v) for k, v in test_scores.items()}, "time": time.time()},
                )
                base.append_jsonl(
                    out_dir / "metrics.jsonl",
                    {
                        "epoch": epoch,
                        "avg_total_loss": float(avg_total),
                        "avg_ce_loss": float(avg_ce),
                        "ppl": ppl_epoch,
                        "lr": current_lr,
                        "time": time.time(),
                    },
                )

                base.save_metrics_csv(epoch_metrics, out_dir / "epoch_metrics.csv")

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
        if ddp and base.is_torchrun_env():
            base.cleanup_ddp()


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.add_argument("--structured_gt_path", default=None)
    parser.add_argument("--lambda_ce", type=float, default=1.0)
    parser.add_argument("--lambda_entity", type=float, default=0.20)
    parser.add_argument("--lambda_action", type=float, default=0.20)
    parser.add_argument("--lambda_attr", type=float, default=0.08)
    parser.add_argument("--lambda_scene", type=float, default=0.06)
    parser.add_argument("--prior_caption_pos_weight", type=float, default=1.0)
    parser.add_argument("--prior_video_only_pos_weight", type=float, default=1.0)
    parser.add_argument("--entity_prior_caption_pos_weight", type=float, default=None)
    parser.add_argument("--entity_prior_video_only_pos_weight", type=float, default=None)
    parser.add_argument("--action_prior_caption_pos_weight", type=float, default=None)
    parser.add_argument("--action_prior_video_only_pos_weight", type=float, default=None)
    parser.add_argument("--attr_prior_caption_pos_weight", type=float, default=None)
    parser.add_argument("--attr_prior_video_only_pos_weight", type=float, default=None)
    parser.add_argument("--scene_prior_caption_pos_weight", type=float, default=None)
    parser.add_argument("--scene_prior_video_only_pos_weight", type=float, default=None)
    parser.add_argument("--lambda_phrase", type=float, default=0.08)
    parser.add_argument("--lambda_phrase_gen", type=float, default=0.0)
    parser.add_argument("--lambda_phrase_pred_gen", type=float, default=0.0)
    parser.add_argument("--phrase_pred_gen_start_epoch", type=int, default=1)
    parser.add_argument("--lambda_phrase_slot_presence", type=float, default=0.0)
    parser.add_argument("--lambda_phrase_slot_div", type=float, default=0.0)
    parser.add_argument("--lambda_phrase_ref_slot_align", type=float, default=0.0)
    parser.add_argument("--lambda_phrase_ref_bridge", type=float, default=0.0)
    parser.add_argument("--lambda_phrase_bridge", type=float, default=0.0)
    parser.add_argument("--lambda_phrase_slot_source_align", type=float, default=0.0)
    parser.add_argument("--lambda_phrase_slot_source_comp", type=float, default=0.0)
    parser.add_argument("--phrase_slot_source_comp_margin", type=float, default=0.05)
    parser.add_argument("--prior_dropout", type=float, default=0.1)
    parser.add_argument("--struct_condition", type=int, default=1)
    parser.add_argument("--struct_condition_scale", type=float, default=0.35)
    parser.add_argument("--struct_condition_query_bridge_enable", type=int, default=0)
    parser.add_argument("--struct_condition_query_bridge_num_queries", type=int, default=4)
    parser.add_argument("--struct_condition_query_bridge_scale", type=float, default=0.15)
    parser.add_argument("--struct_condition_query_bridge_memory_enable", type=int, default=0)
    parser.add_argument("--struct_condition_query_bridge_memory_scale", type=float, default=0.15)
    parser.add_argument("--struct_condition_query_bridge_hidden_enable", type=int, default=0)
    parser.add_argument("--struct_condition_query_bridge_hidden_scale", type=float, default=0.15)
    parser.add_argument("--phrase_decoder_enable", type=int, default=0)
    parser.add_argument("--phrase_condition_enable", type=int, default=0)
    parser.add_argument("--phrase_condition_slot_aware_enable", type=int, default=0)
    parser.add_argument("--phrase_condition_slot_selective_enable", type=int, default=0)
    parser.add_argument("--phrase_condition_slot_residual_enable", type=int, default=0)
    parser.add_argument("--phrase_condition_train_use_predicted", type=int, default=0)
    parser.add_argument("--phrase_condition_pred_detach", type=int, default=1)
    parser.add_argument("--phrase_condition_pred_detach_until_epoch", type=int, default=0)
    parser.add_argument(
        "--phrase_condition_teacher_source",
        type=str,
        default="single_ref",
        choices=["single_ref", "ref_bank"],
    )
    parser.add_argument("--phrase_decoder_layers", type=int, default=2)
    parser.add_argument("--phrase_condition_scale", type=float, default=0.25)
    parser.add_argument("--phrase_condition_aux_scale", type=float, default=0.15)
    parser.add_argument("--phrase_condition_slot_residual_scale", type=float, default=0.15)
    parser.add_argument("--phrase_condition_family_bridge_enable", type=int, default=0)
    parser.add_argument("--phrase_condition_family_bridge_scale", type=float, default=0.20)
    parser.add_argument("--phrase_condition_core_slot_types", type=str, default="")
    parser.add_argument("--phrase_condition_aux_slot_types", type=str, default="")
    parser.add_argument("--phrase_condition_slot_residual_slot_types", type=str, default="")
    parser.add_argument("--phrase_condition_candidate_bias_enable", type=int, default=0)
    parser.add_argument("--phrase_condition_candidate_bias_scale", type=float, default=0.10)
    parser.add_argument("--phrase_condition_candidate_topk", type=int, default=12)
    parser.add_argument("--phrase_condition_candidate_slot_types", type=str, default="")
    parser.add_argument("--phrase_condition_query_bridge_enable", type=int, default=0)
    parser.add_argument("--phrase_condition_query_bridge_num_queries", type=int, default=4)
    parser.add_argument("--phrase_condition_query_bridge_scale", type=float, default=0.15)
    parser.add_argument("--phrase_gen_max_len", type=int, default=48)
    parser.add_argument("--phrase_memory_mode", type=str, default="pooled", choices=["pooled", "temporal"])
    parser.add_argument("--phrase_target_mode", type=str, default="flat", choices=["flat", "slot"])
    parser.add_argument(
        "--phrase_slot_schema",
        type=str,
        default="raw",
        choices=["raw", "typed", "typed_rich", "typed_rich_semantic", "typed_rich_roleaware", "family4_compact"],
    )
    parser.add_argument("--phrase_include_attr_units", type=int, default=0)
    parser.add_argument("--phrase_include_scene_units", type=int, default=0)
    parser.add_argument("--phrase_attr_scene_units_start_epoch", type=int, default=0)
    parser.add_argument("--phrase_attr_scene_units_end_epoch", type=int, default=0)
    parser.add_argument("--phrase_include_video_phrase_units", type=int, default=0)
    parser.add_argument("--phrase_include_video_attr_units", type=int, default=0)
    parser.add_argument("--phrase_include_video_scene_units", type=int, default=0)
    parser.add_argument("--phrase_video_phrase_min_support", type=int, default=2)
    parser.add_argument("--phrase_video_phrase_max_units", type=int, default=4)
    parser.add_argument("--max_phrase_slots", type=int, default=4)
    parser.add_argument("--phrase_slot_max_len", type=int, default=24)
    parser.add_argument("--phrase_slot_planner_enable", type=int, default=0)
    parser.add_argument("--phrase_slot_planner_flow_enable", type=int, default=0)
    parser.add_argument("--phrase_slot_planner_flow_scale", type=float, default=0.20)
    parser.add_argument("--phrase_slot_planner_flow_slot_types", type=str, default="")
    parser.add_argument("--phrase_slot_guidance_enable", type=int, default=0)
    parser.add_argument("--phrase_slot_role_anchor_enable", type=int, default=0)
    parser.add_argument("--phrase_slot_role_anchor_topk", type=int, default=4)
    parser.add_argument("--phrase_slot_role_anchor_scale", type=float, default=1.0)
    parser.add_argument("--phrase_slot_role_anchor_slot_types", type=str, default="")
    parser.add_argument("--phrase_slot_decode_anchor_enable", type=int, default=0)
    parser.add_argument("--phrase_slot_decode_anchor_topk", type=int, default=8)
    parser.add_argument("--phrase_slot_decode_anchor_scale", type=float, default=1.0)
    parser.add_argument("--phrase_slot_decode_anchor_early_scale", type=float, default=1.25)
    parser.add_argument("--phrase_slot_decode_anchor_family_scale", type=float, default=0.75)
    parser.add_argument("--phrase_slot_decode_anchor_family_topk", type=int, default=64)
    parser.add_argument("--phrase_slot_decode_anchor_family_min_count", type=int, default=2)
    parser.add_argument("--phrase_slot_decode_anchor_stopword_penalty", type=float, default=0.75)
    parser.add_argument("--phrase_slot_decode_anchor_stopword_steps", type=int, default=2)
    parser.add_argument("--phrase_slot_decode_anchor_debug_topk", type=int, default=8)
    parser.add_argument("--phrase_slot_presence_enable", type=int, default=0)
    parser.add_argument("--phrase_slot_presence_support_enable", type=int, default=0)
    parser.add_argument("--phrase_slot_presence_evidence_enable", type=int, default=0)
    parser.add_argument("--phrase_slot_presence_context_slot_types", type=str, default="")
    parser.add_argument("--phrase_slot_presence_threshold", type=float, default=0.5)
    parser.add_argument("--phrase_slot_active_slot_types", type=str, default="")
    parser.add_argument(
        "--phrase_slot_presence_calibration_mode",
        type=str,
        default="none",
        choices=["none", "pos_weight"],
    )
    parser.add_argument("--phrase_slot_presence_calibration_slot_types", type=str, default="")
    parser.add_argument("--phrase_slot_presence_threshold_min", type=float, default=0.35)
    parser.add_argument("--phrase_slot_presence_threshold_max", type=float, default=0.65)
    parser.add_argument("--phrase_slot_reweight_enable", type=int, default=0)
    parser.add_argument("--phrase_slot_reweight_power", type=float, default=0.5)
    parser.add_argument("--phrase_slot_reweight_min", type=float, default=1.0)
    parser.add_argument("--phrase_slot_reweight_max", type=float, default=4.0)
    parser.add_argument("--phrase_slot_multiref_enable", type=int, default=0)
    parser.add_argument("--phrase_slot_multiref_max_refs", type=int, default=0)
    parser.add_argument("--phrase_slot_multiref_reduce", type=str, default="mean", choices=["mean", "sum"])
    parser.add_argument("--phrase_slot_multiref_gain", type=float, default=1.0)
    parser.add_argument("--phrase_slot_multiref_chunk_size", type=int, default=4)
    parser.add_argument(
        "--phrase_slot_pred_multiref_reduce",
        type=str,
        default="inherit",
        choices=["inherit", "mean", "sum", "min", "softmin"],
    )
    parser.add_argument("--phrase_slot_pred_multiref_softmin_temp", type=float, default=1.0)
    parser.add_argument(
        "--phrase_slot_family_sample_mode",
        type=str,
        default="first",
        choices=["first", "seeded_hash", "epoch_seeded_hash"],
    )
    parser.add_argument("--phrase_slot_family_sample_seed", type=int, default=-1)
    parser.add_argument(
        "--phrase_slot_family_expand_mode",
        type=str,
        default="none",
        choices=["none", "parallel"],
    )
    parser.add_argument(
        "--prior_head_type",
        type=str,
        default="simple",
        choices=["simple", "multi_semantic", "attn_nextvlad"],
    )
    parser.add_argument("--prior_head_num_heads", type=int, default=8)
    parser.add_argument("--prior_head_hidden_dim", type=int, default=2048)
    parser.add_argument("--prior_head_num_blocks", type=int, default=4)
    parser.add_argument("--prior_head_num_clusters", type=int, default=16)
    parser.add_argument("--prior_head_expansion", type=int, default=2)
    parser.add_argument("--prior_head_groups", type=int, default=8)
    parser.add_argument("--prior_loss_type", type=str, default="bce", choices=["bce", "asl"])
    parser.add_argument("--prior_asl_gamma_neg", type=float, default=4.0)
    parser.add_argument("--prior_asl_gamma_pos", type=float, default=1.0)
    parser.add_argument("--prior_asl_clip", type=float, default=0.05)
    parser.add_argument("--prior_asl_eps", type=float, default=1e-8)
    parser.add_argument("--phrase_progress_enable", type=int, default=0)
    parser.add_argument("--phrase_progress_memory_scale", type=float, default=0.10)
    parser.add_argument("--phrase_progress_source_scale", type=float, default=0.10)
    parser.add_argument("--max_train_steps_per_epoch", type=int, default=0)
    parser.add_argument("--phrase_max_len", type=int, default=77)
    parser.add_argument("--phrase_fallback_to_caption", type=int, default=0)
    parser.add_argument(
        "--training_stage",
        type=str,
        default="joint",
        choices=["joint", "stage1_word", "stage2_phrase", "stage3_sentence"],
    )
    parser.add_argument("--init_model_ckpt", type=str, default=None)
    parser.add_argument("--init_caption_ckpt", type=str, default=None)
    parser.add_argument("--init_skip_structured_vocab_modules", type=int, default=0)
    parser.add_argument("--freeze_caption_model", type=int, default=0)
    parser.add_argument("--freeze_entity_action_heads", type=int, default=0)
    parser.add_argument("--freeze_struct_condition", type=int, default=0)
    return parser


def _fill_default_paths(args):
    if args.dataset_type == "msrvtt":
        if args.clip_global_vision_feats_path is None:
            args.clip_global_vision_feats_path = "../datasets/MSRVTT/feats/ViT-B-32_k_split_ks12_features.pickle"
        if args.annotations_path is None:
            args.annotations_path = "../datasets/MSRVTT/MSRVTT_data.json"
        if args.structured_gt_path is None:
            args.structured_gt_path = "./annotations/msrvtt_structured_train.json"
    elif args.dataset_type == "msvd":
        if args.clip_global_vision_feats_path is None:
            args.clip_global_vision_feats_path = "../datasets/MSVD/feats/ViT-B-32_k_split_ks12_features.pickle"
        if args.annotations_path is None:
            args.annotations_path = "../datasets/MSVD/annotations_preprocessed.txt"
        if args.out_dir == "./runs/base_mean_ks20":
            args.out_dir = "./runs/msvd_structured"
        if args.structured_gt_path is None:
            args.structured_gt_path = "./annotations/msvd_structured_train.json"
    else:
        raise ValueError(f"Unsupported dataset_type: {args.dataset_type}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    _fill_default_paths(args)
    _apply_training_stage_loss_profile(args)

    if not Path(args.structured_gt_path).exists():
        raise FileNotFoundError(
            f"structured_gt_path not found: {args.structured_gt_path}. "
            f"Please run build_structured_gt_api.py first."
        )

    if bool(args.ddp) and base.is_torchrun_env():
        main_worker(local_rank=int(os.environ.get("LOCAL_RANK", 0)), args=args)
    else:
        main_worker(local_rank=0, args=args)


if __name__ == "__main__":
    main()
