#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate structured-caption checkpoints with optional ISCR-style reranking.

This script keeps strict no-leak inference:
- Video features only at inference time.
- No oracle text evidence.
- Rerank evidence is built from model-internal entity/action priors.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import train_base_mean_monitored as base
from dataloaders.dataset_msvd_feats import MSVD_FeaturesDataset
from dataloaders.dataset_msrvtt_feats import MSRVTT_FeaturesDataset
from dataloaders.dataset_structured_caption import StructuredCaptionDataset
from load_tokenizers import CLIPTokenizer_Custom
from models_structured import StructuredCaptionModel
from phrase_lexical_anchors import build_phrase_lexical_anchor_kwargs


SPECIAL_TOKEN_IDS = {0, 49406, 49407}
SOFT_SUPPORT_NOTE = (
    "Planner-weighted stage-1 priors; support evidence only, not decoded phrase output."
)
SOFT_SUPPORT_STAGE_KEYS = {
    "entity": "top_entity_priors",
    "action": "top_action_priors",
    "attribute": "top_attribute_priors",
    "scene": "top_scene_priors",
}
SOFT_SUPPORT_DEFAULT_SOURCE_PREFERENCE: Tuple[str, ...] = (
    "entity",
    "action",
    "attribute",
    "scene",
    "struct",
)
SOFT_SUPPORT_SLOT_SOURCE_PREFERENCES: Dict[str, Tuple[str, ...]] = {
    "subject_action": ("action", "entity", "attribute", "scene", "struct"),
    "object_passive": ("action", "entity", "attribute", "scene", "struct"),
    "relation_scene": ("scene", "action", "entity", "attribute", "struct"),
    "attribute_misc": ("attribute", "entity", "action", "scene", "struct"),
    "subject_entity": ("entity", "attribute", "action", "scene", "struct"),
    "object_entity": ("entity", "attribute", "action", "scene", "struct"),
    "relation_detail": ("action", "entity", "attribute", "scene", "struct"),
    "instrument_detail": ("entity", "attribute", "action", "scene", "struct"),
    "entity_modifier": ("attribute", "entity", "action", "scene", "struct"),
    "subject_modifier": ("attribute", "entity", "action", "scene", "struct"),
    "object_modifier": ("attribute", "entity", "action", "scene", "struct"),
    "scene_context": ("scene", "entity", "attribute", "action", "struct"),
}
SOFT_SUPPORT_TEXT_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "to",
        "of",
        "on",
        "in",
        "with",
        "at",
        "by",
        "for",
        "from",
        "into",
        "onto",
        "over",
        "under",
        "and",
    }
)


@torch.no_grad()
def beam_search_candidates_batch(
    model: torch.nn.Module,
    vid_feat: torch.Tensor,  # [B, T, D]
    vid_mask: torch.Tensor,  # [B, T]
    generation_state=None,
    beam_size: int = 5,
    max_new_tokens: int = 76,
    alpha: float = 0.7,
) -> List[List[Tuple[List[int], float]]]:
    """Return per-sample beam candidates as (token_ids, normalized_logprob)."""
    device = vid_feat.device
    batch_size = vid_feat.size(0)

    bos_id = getattr(model, "bos_token_id", 49406)
    eos_id = getattr(model, "eos_token_id", 49407)
    pad_id = getattr(model, "pad_token_id", 0)

    def length_penalty(length: int) -> float:
        return ((5.0 + length) ** alpha) / ((5.0 + 1.0) ** alpha)

    if generation_state is None:
        generation_state = base.prepare_generation_state_if_available(model, vid_feat, vid_mask)

    all_candidates: List[List[Tuple[List[int], float]]] = []

    for i in range(batch_size):
        vf = vid_feat[i : i + 1]
        vm = vid_mask[i : i + 1]
        sample_generation_state = base.select_generation_state_if_available(model, generation_state, i)
        beams: List[Tuple[List[int], float, bool]] = [([bos_id], 0.0, False)]

        for _ in range(max_new_tokens):
            max_len = max(len(seq) for seq, _, _ in beams)
            m = len(beams)

            captions = torch.full((m, max_len + 1), pad_id, dtype=torch.long, device=device)
            caption_mask = torch.zeros((m, max_len + 1), dtype=torch.bool, device=device)
            for j, (seq, _, _) in enumerate(beams):
                sl = len(seq)
                captions[j, :sl] = torch.tensor(seq, dtype=torch.long, device=device)
                caption_mask[j, :sl] = True

            vf_rep = vf.expand(m, -1, -1)
            vm_rep = vm.expand(m, -1)
            beam_generation_state = base.expand_generation_state_if_available(model, sample_generation_state, m)
            logits = base.forward_with_generation_state(
                model,
                vf_rep,
                vm_rep,
                captions,
                caption_mask,
                generation_state=beam_generation_state,
            )
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)

            expanded: List[Tuple[List[int], float, bool]] = []
            for j, (seq, score, ended) in enumerate(beams):
                if ended:
                    expanded.append((seq, score, True))
                    continue

                values, indices = torch.topk(log_probs[j], k=min(beam_size, log_probs.size(-1)))
                for k in range(values.size(0)):
                    nid = int(indices[k].item())
                    if len(seq) == 1 and nid == eos_id:
                        continue
                    nseq = seq + [nid]
                    nscore = float(score + values[k].item())
                    expanded.append((nseq, nscore, nid == eos_id))

            expanded.sort(key=lambda x: x[1], reverse=True)
            beams = expanded[:beam_size]
            if all(end for _, _, end in beams):
                break

        scored: List[Tuple[List[int], float]] = []
        for seq, score, _ in beams:
            seq_out = seq
            if eos_id in seq_out:
                seq_out = seq_out[: seq_out.index(eos_id) + 1]
            norm = score / length_penalty(max(1, len(seq_out) - 1))
            scored.append((seq_out, float(norm)))
        scored.sort(key=lambda x: x[1], reverse=True)
        all_candidates.append(scored[: max(1, beam_size)])

    return all_candidates


def _safe_encode_phrase_token_ids(tokenizer: CLIPTokenizer_Custom, phrase: str) -> List[int]:
    seq_ids = _safe_encode_phrase_token_sequence(tokenizer, phrase)

    out: List[int] = []
    seen = set()
    for tid_i in seq_ids:
        if tid_i not in seen:
            out.append(tid_i)
            seen.add(tid_i)
    return out


def _safe_encode_phrase_token_sequence(tokenizer: CLIPTokenizer_Custom, phrase: str) -> List[int]:
    try:
        ids = tokenizer.encode(str(phrase), add_special_tokens=False)
    except TypeError:
        ids = tokenizer.encode(str(phrase))
    if not isinstance(ids, list):
        ids = list(ids)
    return [int(tid) for tid in ids if int(tid) not in SPECIAL_TOKEN_IDS]


def decode_predicted_phrase_slots(
    tokenizer: CLIPTokenizer_Custom,
    generation_state,
    index: int,
) -> List[str]:
    if generation_state is None:
        return []

    phrase_slot_ids = generation_state.get("phrase_slot_ids")
    if phrase_slot_ids is None or not torch.is_tensor(phrase_slot_ids) or phrase_slot_ids.dim() < 3:
        return []
    if phrase_slot_ids.size(0) <= index:
        return []

    phrase_slot_mask = generation_state.get("phrase_slot_mask")
    slot_texts: List[str] = []
    for slot_idx in range(phrase_slot_ids.size(1)):
        row_ids = phrase_slot_ids[index, slot_idx]
        if torch.is_tensor(phrase_slot_mask) and phrase_slot_mask.dim() >= 3 and phrase_slot_mask.size(0) > index:
            row_ids = row_ids[phrase_slot_mask[index, slot_idx].bool()]
        token_ids = [int(t) for t in row_ids.tolist() if int(t) not in SPECIAL_TOKEN_IDS]
        slot_text = ""
        if token_ids:
            slot_text = tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
        slot_texts.append(slot_text)
    return slot_texts


def decode_predicted_phrase(
    tokenizer: CLIPTokenizer_Custom,
    generation_state,
    index: int,
) -> Tuple[str, List[str], List[int]]:
    if generation_state is None:
        return "", [], []

    slot_texts = decode_predicted_phrase_slots(
        tokenizer=tokenizer,
        generation_state=generation_state,
        index=index,
    )
    if slot_texts:
        slot_units = [slot_text for slot_text in slot_texts if slot_text]
        phrase_text = " ; ".join(slot_units)
        token_ids = _safe_encode_phrase_token_ids(tokenizer, phrase_text) if phrase_text else []
        return phrase_text, slot_units, token_ids

    phrase_ids = generation_state.get("phrase_decoder_ids")
    if phrase_ids is None or not torch.is_tensor(phrase_ids) or phrase_ids.dim() < 2 or phrase_ids.size(0) <= index:
        return "", [], []

    phrase_mask = generation_state.get("phrase_decoder_mask")
    row_ids = phrase_ids[index]
    if torch.is_tensor(phrase_mask) and phrase_mask.dim() >= 2 and phrase_mask.size(0) > index:
        row_ids = row_ids[phrase_mask[index].bool()]

    token_ids = [int(t) for t in row_ids.tolist() if int(t) not in SPECIAL_TOKEN_IDS]
    if not token_ids:
        return "", [], []

    phrase_text = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
    phrase_units = [part.strip() for part in phrase_text.split(";") if part.strip()]
    return phrase_text, phrase_units, token_ids


def decode_predicted_slot_presence(
    generation_state,
    index: int,
) -> Tuple[List[float], List[int], List[int], List[float], bool, int]:
    if generation_state is None:
        return [], [], [], [], False, -1

    probs = generation_state.get("phrase_slot_presence_probs")
    raw_preds = generation_state.get("phrase_slot_presence_raw_pred")
    preds = generation_state.get("phrase_slot_presence_pred")
    thresholds = generation_state.get("phrase_slot_presence_thresholds")
    fallback_mask = generation_state.get("phrase_slot_presence_fallback_mask")
    fallback_index = generation_state.get("phrase_slot_presence_fallback_index")

    prob_list: List[float] = []
    raw_pred_list: List[int] = []
    pred_list: List[int] = []
    threshold_list: List[float] = []
    fallback_applied = False
    fallback_slot = -1
    if torch.is_tensor(probs) and probs.dim() >= 2 and probs.size(0) > index:
        prob_list = [float(x) for x in probs[index].detach().cpu().tolist()]
    if torch.is_tensor(raw_preds) and raw_preds.dim() >= 2 and raw_preds.size(0) > index:
        raw_pred_list = [int(x) for x in raw_preds[index].detach().cpu().int().tolist()]
    if torch.is_tensor(preds) and preds.dim() >= 2 and preds.size(0) > index:
        pred_list = [int(x) for x in preds[index].detach().cpu().int().tolist()]
    if torch.is_tensor(thresholds) and thresholds.dim() >= 2 and thresholds.size(0) > index:
        threshold_list = [float(x) for x in thresholds[index].detach().cpu().tolist()]
    if torch.is_tensor(fallback_mask) and fallback_mask.dim() >= 1 and fallback_mask.size(0) > index:
        fallback_applied = bool(fallback_mask[index].detach().cpu().item())
    if torch.is_tensor(fallback_index) and fallback_index.dim() >= 1 and fallback_index.size(0) > index:
        fallback_slot = int(fallback_index[index].detach().cpu().item())
    if not raw_pred_list:
        raw_pred_list = list(pred_list)
    return prob_list, raw_pred_list, pred_list, threshold_list, fallback_applied, fallback_slot


def decode_predicted_slot_planner_sources(
    generation_state,
    index: int,
) -> Tuple[List[str], List[List[float]]]:
    if generation_state is None:
        return [], []

    source_names = generation_state.get("phrase_slot_source_names")
    if not isinstance(source_names, (list, tuple)):
        source_names = []

    source_weights = generation_state.get("phrase_slot_source_weights")
    if not torch.is_tensor(source_weights) or source_weights.dim() < 3 or source_weights.size(0) <= index:
        return [str(name) for name in source_names], []

    weight_rows = [
        [float(weight) for weight in row]
        for row in source_weights[index].detach().cpu().tolist()
    ]
    return [str(name) for name in source_names], weight_rows


def decode_predicted_slot_anchor_candidates(
    tokenizer: CLIPTokenizer_Custom,
    generation_state,
    index: int,
) -> List[List[dict]]:
    if generation_state is None:
        return []

    token_ids = generation_state.get("phrase_slot_anchor_token_ids")
    token_scores = generation_state.get("phrase_slot_anchor_token_scores")
    if (
        not torch.is_tensor(token_ids)
        or not torch.is_tensor(token_scores)
        or token_ids.dim() < 3
        or token_scores.dim() < 3
        or token_ids.size(0) <= index
        or token_scores.size(0) <= index
    ):
        return []

    anchor_rows: List[List[dict]] = []
    slot_token_ids = token_ids[index].detach().cpu().tolist()
    slot_token_scores = token_scores[index].detach().cpu().tolist()
    for token_id_row, score_row in zip(slot_token_ids, slot_token_scores):
        candidates: List[dict] = []
        for token_id, score in zip(token_id_row, score_row):
            token_id = int(token_id)
            if token_id in SPECIAL_TOKEN_IDS or float(score) <= 0.0:
                continue
            token_text = tokenizer.decode(
                [token_id],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
            if not token_text:
                continue
            candidates.append(
                {
                    "token": token_text,
                    "token_id": token_id,
                    "score": float(score),
                }
            )
        anchor_rows.append(candidates)
    return anchor_rows


def _word_stage_prior_map(word_stage: Optional[Dict[str, List[dict]]]) -> Dict[str, List[dict]]:
    if not isinstance(word_stage, dict):
        return {}

    prior_map: Dict[str, List[dict]] = {}
    for source_name, stage_key in SOFT_SUPPORT_STAGE_KEYS.items():
        raw_items = word_stage.get(stage_key, [])
        if not isinstance(raw_items, list):
            continue
        prior_map[source_name] = [dict(item) for item in raw_items if isinstance(item, dict)]
    return prior_map


def _slot_soft_support_source_rank(slot_type: str, source_name: str) -> int:
    slot_key = str(slot_type or "").strip()
    source_key = str(source_name or "").strip()

    preferred_sources = SOFT_SUPPORT_SLOT_SOURCE_PREFERENCES.get(slot_key)
    if preferred_sources is None:
        if "scene" in slot_key:
            preferred_sources = ("scene", "entity", "attribute", "action", "struct")
        elif "modifier" in slot_key or "attribute" in slot_key:
            preferred_sources = ("attribute", "entity", "action", "scene", "struct")
        elif "entity" in slot_key or "instrument" in slot_key:
            preferred_sources = ("entity", "attribute", "action", "scene", "struct")
        elif "action" in slot_key or "passive" in slot_key or "relation" in slot_key:
            preferred_sources = ("action", "entity", "attribute", "scene", "struct")
        else:
            preferred_sources = SOFT_SUPPORT_DEFAULT_SOURCE_PREFERENCE

    try:
        return preferred_sources.index(source_key)
    except ValueError:
        return len(preferred_sources) + 1


def _surface_content_tokens(text: str) -> List[str]:
    cleaned = "".join(ch if str(ch).isalnum() else " " for ch in str(text or "").lower())
    return [
        token
        for token in cleaned.split()
        if token and token not in SOFT_SUPPORT_TEXT_STOPWORDS
    ]


def _surface_token_variants(token: str) -> Set[str]:
    token = str(token or "").strip().lower()
    if not token:
        return set()

    variants = {token}
    if len(token) > 5 and token.endswith("ing"):
        base = token[:-3]
        if len(base) >= 4:
            variants.add(base)
            if not base.endswith("e"):
                variants.add(base + "e")
    if len(token) > 4 and token.endswith("ied"):
        variants.add(token[:-3] + "y")
    if len(token) > 4 and token.endswith("ed"):
        base = token[:-2]
        if len(base) >= 3:
            variants.add(base)
            if not base.endswith("e"):
                variants.add(base + "e")
    if len(token) > 4 and token.endswith("es"):
        base = token[:-2]
        if len(base) >= 3:
            variants.add(base)
    if len(token) > 3 and token.endswith("s"):
        base = token[:-1]
        if len(base) >= 3:
            variants.add(base)
    return {variant for variant in variants if len(variant) >= 2}


def _surface_profile(text: str) -> Dict[str, object]:
    tokens = _surface_content_tokens(text)
    variants: Set[str] = set()
    for token in tokens:
        variants.update(_surface_token_variants(token))
    return {
        "tokens": tokens,
        "token_set": set(tokens),
        "variant_set": variants,
        "phrase": " ".join(tokens),
    }


def _slot_story_alignment(slot_text: str, candidate_text: str) -> Dict[str, object]:
    slot_profile = _surface_profile(slot_text)
    candidate_profile = _surface_profile(candidate_text)
    slot_tokens = slot_profile["token_set"]
    candidate_tokens = candidate_profile["token_set"]
    slot_variants = slot_profile["variant_set"]
    candidate_variants = candidate_profile["variant_set"]
    slot_phrase = str(slot_profile["phrase"])
    candidate_phrase = str(candidate_profile["phrase"])

    if not slot_tokens or not candidate_tokens:
        return {
            "alignment_score": 0.0,
            "alignment_label": "",
            "token_overlap": 0,
            "variant_overlap": 0,
        }

    exact_match = int(bool(slot_phrase) and slot_phrase == candidate_phrase)
    token_overlap = int(len(slot_tokens & candidate_tokens))
    variant_overlap = int(len(slot_variants & candidate_variants))
    phrase_overlap = int(
        bool(slot_phrase)
        and bool(candidate_phrase)
        and (candidate_phrase in slot_phrase or slot_phrase in candidate_phrase)
    )
    alignment_score = float(
        6.0 * exact_match
        + 2.0 * phrase_overlap
        + 1.5 * token_overlap
        + 0.5 * variant_overlap
    )
    alignment_label = ""
    if exact_match:
        alignment_label = "exact"
    elif phrase_overlap or token_overlap:
        alignment_label = "token"
    elif variant_overlap:
        alignment_label = "variant"
    return {
        "alignment_score": alignment_score,
        "alignment_label": alignment_label,
        "token_overlap": token_overlap,
        "variant_overlap": variant_overlap,
    }


def _dedupe_story_candidates(rows: Sequence[dict]) -> List[dict]:
    deduped: List[dict] = []
    seen_tokens: Set[str] = set()
    for row in rows:
        token_key = str(row.get("token", "")).strip().lower()
        if not token_key or token_key in seen_tokens:
            continue
        seen_tokens.add(token_key)
        deduped.append(dict(row))
    return deduped


def _build_slot_soft_support(
    planner_sources: Sequence[dict],
    word_stage: Optional[Dict[str, List[dict]]],
    *,
    slot_type: str = "",
    slot_label: str = "",
    slot_text: str = "",
    per_source_topk: int = 3,
    story_scan_topk: int = 64,
    story_topk: int = 6,
    merged_topk: int = 6,
) -> Optional[dict]:
    if not planner_sources:
        return None

    prior_map = _word_stage_prior_map(word_stage)
    source_breakdown: List[dict] = []
    merged_candidates: List[dict] = []
    story_candidates: List[dict] = []
    story_scan_limit = max(max(1, int(per_source_topk)), max(1, int(story_scan_topk)))

    for planner_item in planner_sources:
        source_name = str(planner_item.get("source", "")).strip()
        planner_weight = float(planner_item.get("weight", 0.0))
        source_rank = _slot_soft_support_source_rank(slot_type=slot_type, source_name=source_name)
        raw_candidates = prior_map.get(source_name, [])
        candidate_rows: List[dict] = []
        for candidate_idx, prior_item in enumerate(raw_candidates[:story_scan_limit]):
            token = str(prior_item.get("token", "")).strip()
            if not token:
                continue
            prob = float(prior_item.get("prob", 0.0))
            weighted_score = float(planner_weight * prob)
            merged_row = {
                "token": token,
                "source": source_name,
                "prob": prob,
                "planner_weight": planner_weight,
                "weighted_score": weighted_score,
                "semantic_rank": source_rank,
            }
            if "index" in prior_item:
                merged_row["index"] = int(prior_item["index"])

            if candidate_idx < max(1, int(per_source_topk)):
                candidate_row = {
                    "token": token,
                    "prob": prob,
                    "weighted_score": weighted_score,
                }
                if "index" in prior_item:
                    candidate_row["index"] = int(prior_item["index"])
                candidate_rows.append(candidate_row)
                merged_candidates.append(dict(merged_row))

            if str(slot_text or "").strip():
                alignment = _slot_story_alignment(slot_text=slot_text, candidate_text=token)
                if float(alignment.get("alignment_score", 0.0)) > 0.0:
                    story_candidates.append(
                        {
                            **merged_row,
                            "alignment_score": float(alignment["alignment_score"]),
                            "alignment_label": str(alignment["alignment_label"]),
                            "token_overlap": int(alignment["token_overlap"]),
                            "variant_overlap": int(alignment["variant_overlap"]),
                        }
                    )

        source_breakdown.append(
            {
                "source": source_name,
                "planner_weight": planner_weight,
                "candidates": candidate_rows,
            }
        )

    merged_candidates.sort(
        key=lambda item: (
            int(item.get("semantic_rank", len(SOFT_SUPPORT_DEFAULT_SOURCE_PREFERENCE) + 1)),
            -float(item.get("weighted_score", 0.0)),
            -float(item.get("planner_weight", 0.0)),
            -float(item.get("prob", 0.0)),
            str(item.get("token", "")),
            str(item.get("source", "")),
        )
    )
    story_candidates.sort(
        key=lambda item: (
            -float(item.get("alignment_score", 0.0)),
            int(item.get("semantic_rank", len(SOFT_SUPPORT_DEFAULT_SOURCE_PREFERENCE) + 1)),
            -float(item.get("weighted_score", 0.0)),
            -float(item.get("planner_weight", 0.0)),
            -float(item.get("prob", 0.0)),
            str(item.get("token", "")),
            str(item.get("source", "")),
        )
    )
    story_candidates = _dedupe_story_candidates(story_candidates)
    return {
        "evidence_only": True,
        "note": SOFT_SUPPORT_NOTE,
        "slot_type": str(slot_type or ""),
        "slot_label": str(slot_label or ""),
        "slot_text": str(slot_text or ""),
        "story_candidates": story_candidates[: max(1, int(story_topk))],
        "source_breakdown": source_breakdown,
        "merged_candidates": merged_candidates[: max(1, int(merged_topk))],
    }


def build_phrase_slot_records(
    slot_texts: Sequence[str],
    presence_probs: Sequence[float],
    presence_pred: Sequence[int],
    presence_forced_pred: Optional[Sequence[int]] = None,
    presence_thresholds: Optional[Sequence[float]] = None,
    fallback_applied: bool = False,
    fallback_slot: int = -1,
    phrase_slot_schema: str = "raw",
    max_phrase_slots: int = 4,
    slot_source_names: Optional[Sequence[str]] = None,
    slot_source_weights: Optional[Sequence[Sequence[float]]] = None,
    slot_anchor_candidates: Optional[Sequence[Sequence[dict]]] = None,
    word_stage: Optional[Dict[str, List[dict]]] = None,
) -> List[dict]:
    slot_texts = [str(text) for text in slot_texts]
    presence_probs = [float(x) for x in presence_probs]
    presence_pred = [int(x) for x in presence_pred]
    presence_forced_pred = [int(x) for x in (presence_forced_pred or [])]
    presence_thresholds = [float(x) for x in (presence_thresholds or [])]
    slot_source_names = [str(name) for name in (slot_source_names or [])]
    slot_source_weights = [
        [float(weight) for weight in row]
        for row in (slot_source_weights or [])
    ]
    slot_count = max(
        max(1, int(max_phrase_slots)),
        len(slot_texts),
        len(presence_probs),
        len(presence_pred),
        len(slot_source_weights),
    )
    slot_specs = StructuredCaptionDataset.get_phrase_slot_type_specs(
        max_phrase_slots=slot_count,
        phrase_slot_schema=phrase_slot_schema,
    )

    records: List[dict] = []
    for slot_idx in range(slot_count):
        spec = dict(slot_specs[slot_idx])
        text = str(slot_texts[slot_idx]).strip() if slot_idx < len(slot_texts) else ""
        prob = float(presence_probs[slot_idx]) if slot_idx < len(presence_probs) else 0.0
        pred = int(presence_pred[slot_idx]) if slot_idx < len(presence_pred) else int(bool(text))
        forced_pred = int(presence_forced_pred[slot_idx]) if slot_idx < len(presence_forced_pred) else pred
        threshold = float(presence_thresholds[slot_idx]) if slot_idx < len(presence_thresholds) else 0.0
        slot_fallback = bool(fallback_applied and slot_idx == int(fallback_slot))
        anchor_candidates = []
        if slot_anchor_candidates and slot_idx < len(slot_anchor_candidates):
            anchor_candidates = [
                dict(item)
                for item in slot_anchor_candidates[slot_idx]
                if isinstance(item, dict)
            ]
        planner_sources: List[dict] = []
        if slot_source_names and slot_idx < len(slot_source_weights):
            planner_sources = [
                {
                    "source": str(source_name),
                    "weight": float(source_weight),
                }
                for source_name, source_weight in zip(slot_source_names, slot_source_weights[slot_idx])
            ]
            planner_sources.sort(key=lambda item: item["weight"], reverse=True)
        soft_support = _build_slot_soft_support(
            planner_sources=planner_sources,
            word_stage=word_stage,
            slot_type=str(spec.get("slot_type", "")),
            slot_label=str(spec.get("slot_label", "")),
            slot_text=text,
        )
        records.append(
            {
                **spec,
                "presence_prob": prob,
                "presence_threshold": threshold,
                "presence_pred": pred,
                "presence_forced_pred": forced_pred,
                "presence_fallback_applied": slot_fallback,
                "has_text": bool(text),
                "active": bool(pred),
                "text": text,
                "decode_anchor_candidates": anchor_candidates,
                "planner_top_source": planner_sources[0]["source"] if planner_sources else "",
                "planner_sources": planner_sources,
                **({"soft_support": soft_support} if soft_support else {}),
            }
        )
    return records


def summarize_top_priors(
    prob_row: Optional[torch.Tensor],
    vocab: Sequence[str],
    topk: int,
    min_prob: float = 0.0,
) -> List[dict]:
    if prob_row is None or not vocab:
        return []
    if prob_row.numel() == 0:
        return []

    k = min(max(1, int(topk)), int(prob_row.numel()))
    values, indices = torch.topk(prob_row, k=k)
    rows: List[dict] = []
    for prob, idx in zip(values.tolist(), indices.tolist()):
        if float(prob) < float(min_prob):
            continue
        if idx < 0 or idx >= len(vocab):
            continue
        rows.append(
            {
                "token": str(vocab[int(idx)]),
                "prob": float(prob),
                "index": int(idx),
            }
        )

    if rows:
        return rows

    fallback_prob, fallback_idx = torch.topk(prob_row, k=k)
    for prob, idx in zip(fallback_prob.tolist(), fallback_idx.tolist()):
        if idx < 0 or idx >= len(vocab):
            continue
        rows.append(
            {
                "token": str(vocab[int(idx)]),
                "prob": float(prob),
                "index": int(idx),
            }
        )
    return rows


def build_multistage_prediction_record(
    *,
    video_id: str,
    final_caption: str,
    predicted_phrase_text: str,
    predicted_phrase_units: Sequence[str],
    predicted_phrase_token_ids: Sequence[int],
    predicted_slot_presence_probs: Sequence[float],
    predicted_slot_presence_pred: Sequence[int],
    predicted_slot_presence_forced_pred: Sequence[int] = (),
    predicted_slot_presence_thresholds: Sequence[float] = (),
    predicted_slot_presence_fallback_applied: bool = False,
    predicted_slot_presence_fallback_slot: int = -1,
    phrase_slots: Sequence[dict] = (),
    word_stage: Optional[Dict[str, List[dict]]] = None,
) -> dict:
    phrase_stage = build_phrase_stage_payload(
        predicted_phrase_text=predicted_phrase_text,
        predicted_phrase_units=predicted_phrase_units,
        predicted_phrase_token_ids=predicted_phrase_token_ids,
        predicted_slot_presence_probs=predicted_slot_presence_probs,
        predicted_slot_presence_pred=predicted_slot_presence_pred,
        predicted_slot_presence_forced_pred=predicted_slot_presence_forced_pred,
        predicted_slot_presence_thresholds=predicted_slot_presence_thresholds,
        predicted_slot_presence_fallback_applied=predicted_slot_presence_fallback_applied,
        predicted_slot_presence_fallback_slot=predicted_slot_presence_fallback_slot,
        phrase_slots=phrase_slots,
    )
    sentence_stage = {"final_caption": str(final_caption)}
    return {
        "video_id": str(video_id),
        "top_entity_priors": [dict(item) for item in (word_stage or {}).get("top_entity_priors", [])],
        "top_action_priors": [dict(item) for item in (word_stage or {}).get("top_action_priors", [])],
        "top_attribute_priors": [dict(item) for item in (word_stage or {}).get("top_attribute_priors", [])],
        "top_scene_priors": [dict(item) for item in (word_stage or {}).get("top_scene_priors", [])],
        "predicted_phrase_text": phrase_stage["predicted_phrase_text"],
        "predicted_phrase_units": list(phrase_stage["predicted_phrase_units"]),
        "predicted_phrase_token_ids": list(phrase_stage["predicted_phrase_token_ids"]),
        "predicted_phrase_slot_presence_probs": list(phrase_stage["predicted_phrase_slot_presence_probs"]),
        "predicted_phrase_slot_presence_pred": list(phrase_stage["predicted_phrase_slot_presence_pred"]),
        "predicted_phrase_slot_presence_forced_pred": list(phrase_stage["predicted_phrase_slot_presence_forced_pred"]),
        "predicted_phrase_slot_presence_thresholds": list(phrase_stage["predicted_phrase_slot_presence_thresholds"]),
        "predicted_phrase_slot_presence_fallback_applied": bool(
            phrase_stage["predicted_phrase_slot_presence_fallback_applied"]
        ),
        "predicted_phrase_slot_presence_fallback_slot": int(
            phrase_stage["predicted_phrase_slot_presence_fallback_slot"]
        ),
        "phrase_slots": list(phrase_stage["phrase_slots"]),
        "final_caption": sentence_stage["final_caption"],
        "word_stage": {
            key: [dict(item) for item in value]
            for key, value in (word_stage or {}).items()
        },
        "phrase_stage": phrase_stage,
        "sentence_stage": sentence_stage,
    }


def build_phrase_stage_payload(
    *,
    predicted_phrase_text: str,
    predicted_phrase_units: Sequence[str],
    predicted_phrase_token_ids: Sequence[int],
    predicted_slot_presence_probs: Sequence[float],
    predicted_slot_presence_pred: Sequence[int],
    predicted_slot_presence_forced_pred: Sequence[int],
    predicted_slot_presence_thresholds: Sequence[float],
    predicted_slot_presence_fallback_applied: bool,
    predicted_slot_presence_fallback_slot: int,
    phrase_slots: Sequence[dict],
) -> dict:
    phrase_stage = {
        "predicted_phrase_text": str(predicted_phrase_text),
        "predicted_phrase_units": [str(unit) for unit in predicted_phrase_units],
        "predicted_phrase_token_ids": [int(t) for t in predicted_phrase_token_ids],
        "predicted_phrase_slot_presence_probs": [float(x) for x in predicted_slot_presence_probs],
        "predicted_phrase_slot_presence_pred": [int(x) for x in predicted_slot_presence_pred],
        "predicted_phrase_slot_presence_forced_pred": [int(x) for x in predicted_slot_presence_forced_pred],
        "predicted_phrase_slot_presence_thresholds": [float(x) for x in predicted_slot_presence_thresholds],
        "predicted_phrase_slot_presence_fallback_applied": bool(predicted_slot_presence_fallback_applied),
        "predicted_phrase_slot_presence_fallback_slot": int(predicted_slot_presence_fallback_slot),
        "phrase_slots": [dict(slot) for slot in phrase_slots],
    }
    for slot in phrase_stage["phrase_slots"]:
        soft_support = slot.get("soft_support")
        if isinstance(soft_support, dict):
            phrase_stage["soft_support_note"] = str(soft_support.get("note") or SOFT_SUPPORT_NOTE)
            break
    return phrase_stage


def build_phrase_rerank_targets(
    tokenizer: CLIPTokenizer_Custom,
    predicted_phrase_units: Sequence[str],
    predicted_phrase_token_ids: Sequence[int],
) -> Tuple[List[List[int]], Set[int]]:
    unit_token_lists: List[List[int]] = []
    evidence_ids: Set[int] = set()

    for unit in predicted_phrase_units:
        unit_text = str(unit).strip()
        if not unit_text:
            continue
        unit_ids = _safe_encode_phrase_token_sequence(tokenizer, unit_text)
        if not unit_ids:
            continue
        unit_token_lists.append(unit_ids)
        evidence_ids.update(int(t) for t in unit_ids)

    if not evidence_ids:
        evidence_ids.update(int(t) for t in predicted_phrase_token_ids if int(t) not in SPECIAL_TOKEN_IDS)

    return unit_token_lists, evidence_ids


def _contains_token_subsequence(seq_tokens: Sequence[int], unit_tokens: Sequence[int]) -> bool:
    if not unit_tokens or len(seq_tokens) < len(unit_tokens):
        return False
    unit_len = len(unit_tokens)
    head = int(unit_tokens[0])
    for start, token in enumerate(seq_tokens):
        if int(token) != head:
            continue
        if [int(t) for t in seq_tokens[start : start + unit_len]] == [int(t) for t in unit_tokens]:
            return True
    return False


def _phrase_reward_stats(
    seq: Sequence[int],
    phrase_evidence_ids: Set[int],
    phrase_unit_token_lists: Sequence[Sequence[int]],
) -> Tuple[float, float, float]:
    seq_tokens = [int(tok) for tok in seq if int(tok) not in SPECIAL_TOKEN_IDS]
    seq_token_set = set(seq_tokens)

    token_coverage = 0.0
    if phrase_evidence_ids:
        token_coverage = len(seq_token_set & phrase_evidence_ids) / max(1, len(phrase_evidence_ids))

    unit_coverage = 0.0
    if phrase_unit_token_lists:
        matched_units = sum(1 for unit_ids in phrase_unit_token_lists if _contains_token_subsequence(seq_tokens, unit_ids))
        unit_coverage = matched_units / max(1, len(phrase_unit_token_lists))

    if phrase_evidence_ids and phrase_unit_token_lists:
        reward = 0.5 * float(token_coverage) + 0.5 * float(unit_coverage)
    elif phrase_evidence_ids:
        reward = float(token_coverage)
    else:
        reward = float(unit_coverage)

    return float(token_coverage), float(unit_coverage), float(reward)


def build_semantic_token_lists(
    tokenizer: CLIPTokenizer_Custom,
    entity_vocab: Sequence[str],
    action_vocab: Sequence[str],
    attribute_vocab: Sequence[str],
    scene_vocab: Sequence[str],
) -> Tuple[List[List[int]], List[List[int]], List[List[int]], List[List[int]], Set[int]]:
    entity_lists: List[List[int]] = []
    action_lists: List[List[int]] = []
    attribute_lists: List[List[int]] = []
    scene_lists: List[List[int]] = []
    semantic_ids: Set[int] = set()

    for phrase in entity_vocab:
        ids = _safe_encode_phrase_token_ids(tokenizer, phrase)
        entity_lists.append(ids)
        semantic_ids.update(ids)

    for phrase in action_vocab:
        ids = _safe_encode_phrase_token_ids(tokenizer, phrase)
        action_lists.append(ids)
        semantic_ids.update(ids)

    for phrase in attribute_vocab:
        ids = _safe_encode_phrase_token_ids(tokenizer, phrase)
        attribute_lists.append(ids)
        semantic_ids.update(ids)

    for phrase in scene_vocab:
        ids = _safe_encode_phrase_token_ids(tokenizer, phrase)
        scene_lists.append(ids)
        semantic_ids.update(ids)

    return entity_lists, action_lists, attribute_lists, scene_lists, semantic_ids


def build_weighted_semantic_token_ids(
    entity_token_lists: Sequence[Sequence[int]],
    action_token_lists: Sequence[Sequence[int]],
    attribute_token_lists: Sequence[Sequence[int]],
    scene_token_lists: Sequence[Sequence[int]],
    entity_weight: float,
    action_weight: float,
    attribute_weight: float,
    scene_weight: float,
) -> Set[int]:
    scoped_ids: Set[int] = set()
    if float(entity_weight) > 0.0:
        for ids in entity_token_lists:
            scoped_ids.update(int(t) for t in ids)
    if float(action_weight) > 0.0:
        for ids in action_token_lists:
            scoped_ids.update(int(t) for t in ids)
    if float(attribute_weight) > 0.0:
        for ids in attribute_token_lists:
            scoped_ids.update(int(t) for t in ids)
    if float(scene_weight) > 0.0:
        for ids in scene_token_lists:
            scoped_ids.update(int(t) for t in ids)
    return scoped_ids


@torch.no_grad()
def predict_struct_prior_probs(
    model: StructuredCaptionModel,
    vid_feat: torch.Tensor,
    vid_mask: torch.Tensor,
    aux_raw_global_feats: Optional[torch.Tensor] = None,
    aux_raw_global_mask: Optional[torch.Tensor] = None,
    aux_patch_feats: Optional[torch.Tensor] = None,
    aux_patch_mask: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    _pooled, entity_logits, action_logits, attribute_logits, scene_logits = model._compute_priors(
        vid_feat,
        vid_mask,
        aux_raw_global_feats=aux_raw_global_feats,
        aux_raw_global_mask=aux_raw_global_mask,
        aux_patch_feats=aux_patch_feats,
        aux_patch_mask=aux_patch_mask,
    )
    entity_prob = torch.sigmoid(entity_logits) if entity_logits is not None else None
    action_prob = torch.sigmoid(action_logits) if action_logits is not None else None
    attribute_prob = torch.sigmoid(attribute_logits) if attribute_logits is not None else None
    scene_prob = torch.sigmoid(scene_logits) if scene_logits is not None else None

    return entity_prob, action_prob, attribute_prob, scene_prob


def build_evidence_token_ids(
    entity_prob_row: Optional[torch.Tensor],
    action_prob_row: Optional[torch.Tensor],
    attribute_prob_row: Optional[torch.Tensor],
    scene_prob_row: Optional[torch.Tensor],
    entity_token_lists: Sequence[Sequence[int]],
    action_token_lists: Sequence[Sequence[int]],
    attribute_token_lists: Sequence[Sequence[int]],
    scene_token_lists: Sequence[Sequence[int]],
    prior_topk: int,
    token_topk: int,
    threshold: float,
    entity_threshold: float,
    action_threshold: float,
    attribute_threshold: float,
    scene_threshold: float,
    entity_weight: float,
    action_weight: float,
    attribute_weight: float,
    scene_weight: float,
    entity_min_prob: float,
    action_min_prob: float,
    attribute_min_prob: float,
    scene_min_prob: float,
) -> Set[int]:
    token_score: Dict[int, float] = {}

    def _accumulate(
        prob_row: Optional[torch.Tensor],
        token_lists: Sequence[Sequence[int]],
        branch_weight: float,
        branch_threshold: float,
        branch_min_prob: float,
    ) -> None:
        if prob_row is None:
            return
        if prob_row.numel() == 0:
            return
        if branch_weight <= 0.0:
            return
        max_prob = float(prob_row.max().item())
        if max_prob < float(branch_min_prob):
            return

        k = int(prior_topk)
        if k > 0 and k < prob_row.numel():
            vals, idxs = torch.topk(prob_row, k=k)
            pairs = zip(idxs.tolist(), vals.tolist())
        else:
            pairs = ((i, float(prob_row[i].item())) for i in range(prob_row.numel()))

        for idx, score in pairs:
            if score <= float(branch_threshold):
                continue
            if idx < 0 or idx >= len(token_lists):
                continue
            tids = token_lists[idx]
            if not tids:
                continue
            share = (float(score) * float(branch_weight)) / float(len(tids))
            for tid in tids:
                token_score[int(tid)] = token_score.get(int(tid), 0.0) + share

    _accumulate(
        entity_prob_row,
        entity_token_lists,
        float(entity_weight),
        float(entity_threshold if entity_threshold >= 0.0 else threshold),
        float(entity_min_prob),
    )
    _accumulate(
        action_prob_row,
        action_token_lists,
        float(action_weight),
        float(action_threshold if action_threshold >= 0.0 else threshold),
        float(action_min_prob),
    )
    _accumulate(
        attribute_prob_row,
        attribute_token_lists,
        float(attribute_weight),
        float(attribute_threshold if attribute_threshold >= 0.0 else threshold),
        float(attribute_min_prob),
    )
    _accumulate(
        scene_prob_row,
        scene_token_lists,
        float(scene_weight),
        float(scene_threshold if scene_threshold >= 0.0 else threshold),
        float(scene_min_prob),
    )

    if not token_score:
        return set()

    scored = sorted(token_score.items(), key=lambda x: x[1], reverse=True)
    if int(token_topk) > 0:
        scored = scored[: int(token_topk)]
    return {int(tid) for tid, _ in scored}


def _normalize_base_scores(base_scores: Sequence[float], mode: str) -> List[float]:
    mode_norm = str(mode).lower()
    if not base_scores:
        return []
    if mode_norm == "minmax":
        bmin = float(min(base_scores))
        bmax = float(max(base_scores))
        denom = max(1e-8, bmax - bmin)
        return [(float(x) - bmin) / denom for x in base_scores]
    if mode_norm == "zscore":
        mu = float(sum(base_scores) / max(1, len(base_scores)))
        var = float(sum((float(x) - mu) ** 2 for x in base_scores) / max(1, len(base_scores)))
        std = max(1e-8, math.sqrt(var))
        return [(float(x) - mu) / std for x in base_scores]
    return [float(x) for x in base_scores]


def _collect_branch_evidence(
    prob_row: Optional[torch.Tensor],
    token_lists: Sequence[Sequence[int]],
    prior_topk: int,
    token_topk: int,
    branch_threshold: float,
    branch_min_prob: float,
    branch_weight: float,
) -> Tuple[Set[int], float]:
    if prob_row is None or prob_row.numel() == 0:
        return set(), 0.0
    if float(branch_weight) <= 0.0:
        return set(), 0.0

    max_prob = float(prob_row.max().item())
    if max_prob < float(branch_min_prob):
        return set(), 0.0

    token_score: Dict[int, float] = {}
    k = int(prior_topk)
    if k > 0 and k < prob_row.numel():
        vals, idxs = torch.topk(prob_row, k=k)
        pairs = zip(idxs.tolist(), vals.tolist())
    else:
        pairs = ((i, float(prob_row[i].item())) for i in range(prob_row.numel()))

    for idx, score in pairs:
        if score <= float(branch_threshold):
            continue
        if idx < 0 or idx >= len(token_lists):
            continue
        tids = token_lists[idx]
        if not tids:
            continue
        share = float(score) / max(1, len(tids))
        for tid in tids:
            token_score[int(tid)] = token_score.get(int(tid), 0.0) + share

    if not token_score:
        return set(), 0.0

    scored = sorted(token_score.items(), key=lambda x: x[1], reverse=True)
    if int(token_topk) > 0:
        scored = scored[: int(token_topk)]
    evidence_ids = {int(tid) for tid, _ in scored}
    confidence = max(0.0, min(1.0, (max_prob - float(branch_min_prob)) / max(1e-6, 1.0 - float(branch_min_prob))))
    return evidence_ids, float(confidence)


def build_two_stage_evidence_ids(
    entity_prob_row: Optional[torch.Tensor],
    action_prob_row: Optional[torch.Tensor],
    attribute_prob_row: Optional[torch.Tensor],
    scene_prob_row: Optional[torch.Tensor],
    entity_token_lists: Sequence[Sequence[int]],
    action_token_lists: Sequence[Sequence[int]],
    attribute_token_lists: Sequence[Sequence[int]],
    scene_token_lists: Sequence[Sequence[int]],
    prior_topk: int,
    token_topk: int,
    threshold: float,
    entity_threshold: float,
    action_threshold: float,
    attribute_threshold: float,
    scene_threshold: float,
    entity_weight: float,
    action_weight: float,
    attribute_weight: float,
    scene_weight: float,
    entity_min_prob: float,
    action_min_prob: float,
    attribute_min_prob: float,
    scene_min_prob: float,
) -> Tuple[Set[int], Set[int], Set[int], float, float]:
    entity_ids, _ = _collect_branch_evidence(
        prob_row=entity_prob_row,
        token_lists=entity_token_lists,
        prior_topk=int(prior_topk),
        token_topk=int(token_topk),
        branch_threshold=float(entity_threshold if entity_threshold >= 0.0 else threshold),
        branch_min_prob=float(entity_min_prob),
        branch_weight=float(entity_weight),
    )
    action_ids, _ = _collect_branch_evidence(
        prob_row=action_prob_row,
        token_lists=action_token_lists,
        prior_topk=int(prior_topk),
        token_topk=int(token_topk),
        branch_threshold=float(action_threshold if action_threshold >= 0.0 else threshold),
        branch_min_prob=float(action_min_prob),
        branch_weight=float(action_weight),
    )
    attribute_ids, attribute_conf = _collect_branch_evidence(
        prob_row=attribute_prob_row,
        token_lists=attribute_token_lists,
        prior_topk=int(prior_topk),
        token_topk=int(token_topk),
        branch_threshold=float(attribute_threshold if attribute_threshold >= 0.0 else threshold),
        branch_min_prob=float(attribute_min_prob),
        branch_weight=float(attribute_weight),
    )
    scene_ids, scene_conf = _collect_branch_evidence(
        prob_row=scene_prob_row,
        token_lists=scene_token_lists,
        prior_topk=int(prior_topk),
        token_topk=int(token_topk),
        branch_threshold=float(scene_threshold if scene_threshold >= 0.0 else threshold),
        branch_min_prob=float(scene_min_prob),
        branch_weight=float(scene_weight),
    )
    stage1_ids = set(entity_ids)
    stage1_ids.update(action_ids)
    return stage1_ids, attribute_ids, scene_ids, float(attribute_conf), float(scene_conf)


def evidence_rerank_tokens(
    candidates: Sequence[Tuple[List[int], float]],
    evidence_ids: Set[int],
    semantic_token_ids: Set[int],
    alpha: float,
    lambda_cov: float,
    lambda_hall: float,
    phrase_evidence_ids: Optional[Set[int]] = None,
    phrase_unit_token_lists: Optional[Sequence[Sequence[int]]] = None,
    phrase_weight: float = 0.0,
    base_norm_mode: str = "none",
    return_meta: bool = False,
):
    if not candidates:
        if return_meta:
            return [], 0, {"evidence_count": 0, "candidates": []}
        return [], 0

    base_scores = [float(base_score) for _seq, base_score in candidates]
    base_scores_used = _normalize_base_scores(base_scores, mode=base_norm_mode)
    best_idx = 0
    best_score = -1e9
    best_seq = candidates[0][0]
    cand_meta = []
    phrase_evidence_ids = set(int(t) for t in (phrase_evidence_ids or set()))
    phrase_unit_token_lists = [list(int(t) for t in unit_ids) for unit_ids in (phrase_unit_token_lists or []) if unit_ids]
    phrase_scale = max(0.0, float(phrase_weight)) if (phrase_evidence_ids or phrase_unit_token_lists) else 0.0

    for rank, (seq, _base_score_raw) in enumerate(candidates):
        base_raw = float(base_scores[rank])
        base_score = float(base_scores_used[rank])
        pred_sem = {int(tok) for tok in seq if int(tok) in semantic_token_ids}

        if evidence_ids:
            coverage = len(pred_sem & evidence_ids) / max(1, len(evidence_ids))
        else:
            coverage = 0.0

        if pred_sem:
            hall = len(pred_sem - evidence_ids) / max(1, len(pred_sem))
        else:
            hall = 0.0

        score = float(alpha) * float(base_score) + float(lambda_cov) * float(coverage) - float(lambda_hall) * float(hall)
        phrase_token_coverage = 0.0
        phrase_unit_coverage = 0.0
        phrase_reward = 0.0
        if phrase_scale > 0.0:
            phrase_token_coverage, phrase_unit_coverage, phrase_reward = _phrase_reward_stats(
                seq=seq,
                phrase_evidence_ids=phrase_evidence_ids,
                phrase_unit_token_lists=phrase_unit_token_lists,
            )
            score += float(phrase_scale) * float(phrase_reward)
        cand_meta.append(
            {
                "rank": int(rank),
                "base_score_raw": float(base_raw),
                "base_score_used": float(base_score),
                "coverage_stage1": float(coverage),
                "hall_stage1": float(hall),
                "attr_coverage": 0.0,
                "attr_hall": 0.0,
                "scene_coverage": 0.0,
                "scene_hall": 0.0,
                "phrase_token_coverage": float(phrase_token_coverage),
                "phrase_unit_coverage": float(phrase_unit_coverage),
                "phrase_reward": float(phrase_reward),
                "final_score": float(score),
                "token_ids": [int(t) for t in seq],
            }
        )
        if score > best_score:
            best_score = score
            best_idx = int(rank)
            best_seq = seq

    if return_meta:
        return best_seq, best_idx, {
            "mode": "single",
            "base_norm_mode": str(base_norm_mode).lower(),
            "evidence_count": int(len(evidence_ids)),
            "phrase_evidence_count": int(len(phrase_evidence_ids)),
            "phrase_unit_count": int(len(phrase_unit_token_lists)),
            "phrase_scale": float(phrase_scale),
            "best_score": float(best_score),
            "candidates": cand_meta,
        }
    return best_seq, best_idx


def evidence_rerank_tokens_two_stage(
    candidates: Sequence[Tuple[List[int], float]],
    stage1_evidence_ids: Set[int],
    stage1_semantic_ids: Set[int],
    attribute_evidence_ids: Set[int],
    attribute_semantic_ids: Set[int],
    scene_evidence_ids: Set[int],
    scene_semantic_ids: Set[int],
    alpha: float,
    lambda_cov: float,
    lambda_hall: float,
    attribute_weight: float,
    scene_weight: float,
    attribute_confidence: float,
    scene_confidence: float,
    phrase_evidence_ids: Optional[Set[int]] = None,
    phrase_unit_token_lists: Optional[Sequence[Sequence[int]]] = None,
    phrase_weight: float = 0.0,
    base_norm_mode: str = "none",
    return_meta: bool = False,
):
    if not candidates:
        if return_meta:
            return [], 0, {"stage1_evidence_count": 0, "candidates": []}
        return [], 0

    base_scores = [float(base_score) for _seq, base_score in candidates]
    base_scores_used = _normalize_base_scores(base_scores, mode=base_norm_mode)
    best_idx = 0
    best_score = -1e9
    best_seq = candidates[0][0]
    cand_meta = []

    attr_scale = max(0.0, float(attribute_weight)) * max(0.0, float(attribute_confidence))
    scene_scale = max(0.0, float(scene_weight)) * max(0.0, float(scene_confidence))
    phrase_evidence_ids = set(int(t) for t in (phrase_evidence_ids or set()))
    phrase_unit_token_lists = [list(int(t) for t in unit_ids) for unit_ids in (phrase_unit_token_lists or []) if unit_ids]
    phrase_scale = max(0.0, float(phrase_weight)) if (phrase_evidence_ids or phrase_unit_token_lists) else 0.0

    for rank, (seq, _base_score_raw) in enumerate(candidates):
        base_raw = float(base_scores[rank])
        base_score = float(base_scores_used[rank])
        pred_stage1 = {int(tok) for tok in seq if int(tok) in stage1_semantic_ids}
        if stage1_evidence_ids:
            coverage_stage1 = len(pred_stage1 & stage1_evidence_ids) / max(1, len(stage1_evidence_ids))
        else:
            coverage_stage1 = 0.0
        if pred_stage1:
            hall_stage1 = len(pred_stage1 - stage1_evidence_ids) / max(1, len(pred_stage1))
        else:
            hall_stage1 = 0.0
        score = float(alpha) * float(base_score) + float(lambda_cov) * float(coverage_stage1) - float(lambda_hall) * float(hall_stage1)
        coverage_attr = 0.0
        hall_attr = 0.0

        if attr_scale > 0.0 and attribute_evidence_ids:
            pred_attr = {int(tok) for tok in seq if int(tok) in attribute_semantic_ids}
            coverage_attr = len(pred_attr & attribute_evidence_ids) / max(1, len(attribute_evidence_ids))
            hall_attr = len(pred_attr - attribute_evidence_ids) / max(1, len(pred_attr)) if pred_attr else 0.0
            score += float(attr_scale) * (float(lambda_cov) * float(coverage_attr) - float(lambda_hall) * float(hall_attr))

        coverage_scene = 0.0
        hall_scene = 0.0
        if scene_scale > 0.0 and scene_evidence_ids:
            pred_scene = {int(tok) for tok in seq if int(tok) in scene_semantic_ids}
            coverage_scene = len(pred_scene & scene_evidence_ids) / max(1, len(scene_evidence_ids))
            hall_scene = len(pred_scene - scene_evidence_ids) / max(1, len(pred_scene)) if pred_scene else 0.0
            score += float(scene_scale) * (float(lambda_cov) * float(coverage_scene) - float(lambda_hall) * float(hall_scene))

        phrase_token_coverage = 0.0
        phrase_unit_coverage = 0.0
        phrase_reward = 0.0
        if phrase_scale > 0.0:
            phrase_token_coverage, phrase_unit_coverage, phrase_reward = _phrase_reward_stats(
                seq=seq,
                phrase_evidence_ids=phrase_evidence_ids,
                phrase_unit_token_lists=phrase_unit_token_lists,
            )
            score += float(phrase_scale) * float(phrase_reward)

        cand_meta.append(
            {
                "rank": int(rank),
                "base_score_raw": float(base_raw),
                "base_score_used": float(base_score),
                "coverage_stage1": float(coverage_stage1),
                "hall_stage1": float(hall_stage1),
                "attr_coverage": float(coverage_attr),
                "attr_hall": float(hall_attr),
                "scene_coverage": float(coverage_scene),
                "scene_hall": float(hall_scene),
                "phrase_token_coverage": float(phrase_token_coverage),
                "phrase_unit_coverage": float(phrase_unit_coverage),
                "phrase_reward": float(phrase_reward),
                "final_score": float(score),
                "token_ids": [int(t) for t in seq],
            }
        )

        if score > best_score:
            best_score = score
            best_idx = int(rank)
            best_seq = seq

    if return_meta:
        return best_seq, best_idx, {
            "mode": "two_stage",
            "base_norm_mode": str(base_norm_mode).lower(),
            "stage1_evidence_count": int(len(stage1_evidence_ids)),
            "attr_evidence_count": int(len(attribute_evidence_ids)),
            "scene_evidence_count": int(len(scene_evidence_ids)),
            "attr_scale": float(attr_scale),
            "scene_scale": float(scene_scale),
            "phrase_evidence_count": int(len(phrase_evidence_ids)),
            "phrase_unit_count": int(len(phrase_unit_token_lists)),
            "phrase_scale": float(phrase_scale),
            "best_score": float(best_score),
            "candidates": cand_meta,
        }
    return best_seq, best_idx


def build_dataset(dataset_type: str, features_path: str, annotations_path: str, split: str):
    if dataset_type == "msvd":
        return MSVD_FeaturesDataset(
            features_path=features_path,
            annotations_path=annotations_path,
            split=split,
        )
    if dataset_type == "msrvtt":
        return MSRVTT_FeaturesDataset(
            features_path=features_path,
            json_path=annotations_path,
            split=split,
        )
    raise ValueError(f"Unsupported dataset_type: {dataset_type}")


def get_references_from_dataset(dataset) -> Dict[str, List[str]]:
    dataset = base.unwrap_dataset(dataset)
    refs: Dict[str, List[str]] = {}
    for vid in dataset.video_ids:
        caps = [c for (c, _sid) in dataset.captions_data.get(vid, [])]
        refs[str(vid)] = caps if caps else [""]
    return refs


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["val", "test"])
    ap.add_argument("--dataset_type", type=str, default=None, choices=["msvd", "msrvtt"])
    ap.add_argument("--clip_global_vision_feats_path", type=str, default=None)
    ap.add_argument("--annotations_path", type=str, default=None)
    ap.add_argument("--structured_gt_path", type=str, default=None)
    ap.add_argument("--aux_visual_enable", type=int, default=None)
    ap.add_argument("--aux_raw_global_enable", type=int, default=None)
    ap.add_argument("--aux_raw_global_feats_path", type=str, default=None)
    ap.add_argument("--aux_patch_enable", type=int, default=None)
    ap.add_argument("--aux_patch_root", type=str, default=None)
    ap.add_argument("--aux_patch_block", type=int, default=None)
    ap.add_argument("--aux_visual_raw_global_dim", type=int, default=None)
    ap.add_argument("--aux_visual_patch_dim", type=int, default=None)
    ap.add_argument("--aux_visual_prior_scale", type=float, default=None)
    ap.add_argument("--aux_visual_struct_scale", type=float, default=None)
    ap.add_argument("--aux_visual_memory_scale", type=float, default=None)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument("--beam_alpha", type=float, default=0.7)

    ap.add_argument("--iscr_rerank", type=int, default=1)
    ap.add_argument("--iscr_rerank_alpha", type=float, default=1.0)
    ap.add_argument("--iscr_rerank_lambda_cov", type=float, default=0.8)
    ap.add_argument("--iscr_rerank_lambda_hall", type=float, default=1.0)
    ap.add_argument("--iscr_rerank_base_norm", type=str, default="none", choices=["none", "minmax", "zscore"])
    ap.add_argument("--iscr_prior_topk", type=int, default=64)
    ap.add_argument("--iscr_rerank_topk", type=int, default=20)
    ap.add_argument("--iscr_evidence_threshold", type=float, default=0.0)
    ap.add_argument("--iscr_entity_threshold", type=float, default=-1.0)
    ap.add_argument("--iscr_action_threshold", type=float, default=-1.0)
    ap.add_argument("--iscr_attribute_threshold", type=float, default=-1.0)
    ap.add_argument("--iscr_scene_threshold", type=float, default=-1.0)
    ap.add_argument("--iscr_entity_weight", type=float, default=1.0)
    ap.add_argument("--iscr_action_weight", type=float, default=1.0)
    ap.add_argument("--iscr_attribute_weight", type=float, default=1.0)
    ap.add_argument("--iscr_scene_weight", type=float, default=1.0)
    ap.add_argument(
        "--iscr_halluc_semantic_scope",
        type=str,
        default="weighted",
        choices=["global", "weighted"],
        help="global: use all branch semantic tokens for hallucination penalty; weighted: only use branches with non-zero weights.",
    )
    ap.add_argument("--iscr_entity_min_prob", type=float, default=0.0)
    ap.add_argument("--iscr_action_min_prob", type=float, default=0.0)
    ap.add_argument("--iscr_attribute_min_prob", type=float, default=0.0)
    ap.add_argument("--iscr_scene_min_prob", type=float, default=0.0)
    ap.add_argument("--iscr_two_stage_fusion", type=int, default=0)
    ap.add_argument("--iscr_two_stage_attr_weight", type=float, default=0.0)
    ap.add_argument("--iscr_two_stage_scene_weight", type=float, default=0.0)
    ap.add_argument("--iscr_phrase_weight", type=float, default=0.0)
    ap.add_argument("--iscr_explain_jsonl", type=str, default="")
    ap.add_argument("--phrase_predictions_jsonl", type=str, default="")
    ap.add_argument("--iscr_explain_topk", type=int, default=4)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--output_json", type=str, default="")
    return ap


def main():
    args = build_parser().parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    payload = torch.load(str(ckpt_path), map_location="cpu")
    ckpt_args = payload.get("args", {})
    state = payload.get("model", payload)

    dataset_type = args.dataset_type or ckpt_args.get("dataset_type", "msvd")
    features_path = args.clip_global_vision_feats_path or ckpt_args.get("clip_global_vision_feats_path")
    annotations_path = args.annotations_path or ckpt_args.get("annotations_path")
    structured_gt_path = args.structured_gt_path or ckpt_args.get("structured_gt_path")
    aux_override_names = [
        "aux_visual_enable",
        "aux_raw_global_enable",
        "aux_raw_global_feats_path",
        "aux_patch_enable",
        "aux_patch_root",
        "aux_patch_block",
        "aux_visual_raw_global_dim",
        "aux_visual_patch_dim",
        "aux_visual_prior_scale",
        "aux_visual_struct_scale",
        "aux_visual_memory_scale",
    ]
    for aux_name in aux_override_names:
        if getattr(args, aux_name, None) in (None, "") and aux_name in ckpt_args:
            setattr(args, aux_name, ckpt_args.get(aux_name))

    if not features_path:
        raise ValueError("clip_global_vision_feats_path is required (not found in args/checkpoint).")
    if not annotations_path:
        raise ValueError("annotations_path is required (not found in args/checkpoint).")
    if not structured_gt_path:
        raise ValueError("structured_gt_path is required (not found in args/checkpoint).")

    with Path(structured_gt_path).open("r", encoding="utf-8") as f:
        structured_payload = json.load(f)
    entity_vocab = structured_payload.get("entity_vocab", [])
    action_vocab = structured_payload.get("action_vocab", [])
    attribute_vocab = structured_payload.get("attribute_vocab", [])
    scene_vocab = structured_payload.get("scene_vocab", [])
    lexical_anchor_kwargs = {}
    if bool(int(ckpt_args.get("phrase_slot_decode_anchor_enable", 0) or 0)):
        tokenizer_for_anchors = CLIPTokenizer_Custom()
        lexical_anchor_kwargs = build_phrase_lexical_anchor_kwargs(
            tokenizer=tokenizer_for_anchors,
            structured_payload_or_videos=structured_payload.get("videos", {}),
            entity_vocab=entity_vocab,
            action_vocab=action_vocab,
            attribute_vocab=attribute_vocab,
            scene_vocab=scene_vocab,
            phrase_slot_schema=str(ckpt_args.get("phrase_slot_schema", "raw")),
            max_phrase_slots=int(ckpt_args.get("max_phrase_slots", 4)),
            family_topk_tokens=int(ckpt_args.get("phrase_slot_decode_anchor_family_topk", 64)),
            family_min_count=int(ckpt_args.get("phrase_slot_decode_anchor_family_min_count", 2)),
        )

    state_has_slot_guidance = any(str(key).startswith("slot_guidance_") for key in state.keys())
    state_has_slot_role_anchor = any(str(key).startswith("slot_role_anchor_") for key in state.keys())
    state_has_slot_residual = any(
        str(key).startswith("phrase_condition_slot_residual_") for key in state.keys()
    )
    state_has_candidate_bias = any(
        str(key).startswith("phrase_condition_candidate_") for key in state.keys()
    )
    state_has_struct_query_bridge = any(
        str(key).startswith("struct_condition_query_bridge_") for key in state.keys()
    )
    state_has_struct_query_bridge_memory = "struct_condition_query_bridge_memory_scale" in state
    state_has_struct_query_bridge_hidden = any(
        str(key).startswith("struct_condition_query_bridge_hidden_") for key in state.keys()
    )
    state_has_query_bridge = any(
        str(key).startswith("phrase_condition_query_bridge_") for key in state.keys()
    )
    phrase_query_bridge_num_queries = int(
        ckpt_args.get(
            "phrase_condition_query_bridge_num_queries",
            state.get("phrase_condition_query_bridge_queries", torch.empty(4, 1)).shape[0]
            if "phrase_condition_query_bridge_queries" in state
            else 4,
        )
    )
    struct_query_bridge_num_queries = int(
        ckpt_args.get(
            "struct_condition_query_bridge_num_queries",
            state.get("struct_condition_query_bridge_queries", torch.empty(4, 1)).shape[0]
            if "struct_condition_query_bridge_queries" in state
            else 4,
        )
    )

    model = StructuredCaptionModel(
        entity_dim=len(entity_vocab),
        action_dim=len(action_vocab),
        attribute_dim=len(attribute_vocab),
        scene_dim=len(scene_vocab),
        prior_dropout=float(ckpt_args.get("prior_dropout", 0.1)),
        struct_condition=bool(int(ckpt_args.get("struct_condition", 1))),
        struct_condition_scale=float(ckpt_args.get("struct_condition_scale", 0.35)),
        struct_condition_query_bridge_enable=bool(
            int(ckpt_args.get("struct_condition_query_bridge_enable", 0))
        ) or state_has_struct_query_bridge,
        struct_condition_query_bridge_num_queries=struct_query_bridge_num_queries,
        struct_condition_query_bridge_scale=float(
            ckpt_args.get("struct_condition_query_bridge_scale", 0.15)
        ),
        struct_condition_query_bridge_memory_enable=bool(
            int(ckpt_args.get("struct_condition_query_bridge_memory_enable", 0))
        ) or state_has_struct_query_bridge_memory,
        struct_condition_query_bridge_memory_scale=float(
            ckpt_args.get("struct_condition_query_bridge_memory_scale", 0.15)
        ),
        struct_condition_query_bridge_hidden_enable=bool(
            int(ckpt_args.get("struct_condition_query_bridge_hidden_enable", 0))
        ) or state_has_struct_query_bridge_hidden,
        struct_condition_query_bridge_hidden_scale=float(
            ckpt_args.get("struct_condition_query_bridge_hidden_scale", 0.15)
        ),
        phrase_decoder_enable=bool(int(ckpt_args.get("phrase_decoder_enable", 0))),
        phrase_condition_enable=bool(int(ckpt_args.get("phrase_condition_enable", 0))),
        phrase_condition_slot_aware_enable=bool(int(ckpt_args.get("phrase_condition_slot_aware_enable", 0))),
        phrase_condition_slot_selective_enable=bool(
            int(ckpt_args.get("phrase_condition_slot_selective_enable", 0))
        ),
        phrase_condition_slot_residual_enable=bool(
            int(ckpt_args.get("phrase_condition_slot_residual_enable", 0))
        ) or state_has_slot_residual,
        phrase_condition_pred_detach=bool(int(ckpt_args.get("phrase_condition_pred_detach", 1))),
        phrase_decoder_layers=int(ckpt_args.get("phrase_decoder_layers", 2)),
        phrase_condition_scale=float(ckpt_args.get("phrase_condition_scale", 0.25)),
        phrase_condition_aux_scale=float(ckpt_args.get("phrase_condition_aux_scale", 0.15)),
        phrase_condition_slot_residual_scale=float(
            ckpt_args.get("phrase_condition_slot_residual_scale", 0.15)
        ),
        phrase_condition_core_slot_types=str(ckpt_args.get("phrase_condition_core_slot_types", "")),
        phrase_condition_aux_slot_types=str(ckpt_args.get("phrase_condition_aux_slot_types", "")),
        phrase_condition_slot_residual_slot_types=str(
            ckpt_args.get("phrase_condition_slot_residual_slot_types", "")
        ),
        phrase_condition_family_bridge_enable=bool(
            int(ckpt_args.get("phrase_condition_family_bridge_enable", 0))
        ),
        phrase_condition_family_bridge_scale=float(
            ckpt_args.get("phrase_condition_family_bridge_scale", 0.20)
        ),
        phrase_condition_candidate_bias_enable=bool(
            int(ckpt_args.get("phrase_condition_candidate_bias_enable", 0))
        ) or state_has_candidate_bias,
        phrase_condition_candidate_bias_scale=float(
            ckpt_args.get("phrase_condition_candidate_bias_scale", 0.10)
        ),
        phrase_condition_candidate_topk=int(ckpt_args.get("phrase_condition_candidate_topk", 12)),
        phrase_condition_candidate_slot_types=str(
            ckpt_args.get("phrase_condition_candidate_slot_types", "")
        ),
        phrase_condition_query_bridge_enable=bool(
            int(ckpt_args.get("phrase_condition_query_bridge_enable", 0))
        ) or state_has_query_bridge,
        phrase_condition_query_bridge_num_queries=phrase_query_bridge_num_queries,
        phrase_condition_query_bridge_scale=float(
            ckpt_args.get("phrase_condition_query_bridge_scale", 0.15)
        ),
        phrase_condition_train_use_predicted=bool(int(ckpt_args.get("phrase_condition_train_use_predicted", 0))),
        phrase_gen_max_len=int(ckpt_args.get("phrase_gen_max_len", 48)),
        phrase_memory_mode=str(ckpt_args.get("phrase_memory_mode", "pooled")),
        phrase_target_mode=str(ckpt_args.get("phrase_target_mode", "flat")),
        max_phrase_slots=int(ckpt_args.get("max_phrase_slots", 4)),
        phrase_slot_max_len=int(ckpt_args.get("phrase_slot_max_len", 24)),
        phrase_slot_schema=str(ckpt_args.get("phrase_slot_schema", "raw")),
        phrase_slot_planner_enable=bool(int(ckpt_args.get("phrase_slot_planner_enable", 0))),
        phrase_slot_planner_flow_enable=bool(int(ckpt_args.get("phrase_slot_planner_flow_enable", 0))),
        phrase_slot_planner_flow_scale=float(ckpt_args.get("phrase_slot_planner_flow_scale", 0.20)),
        phrase_slot_planner_flow_slot_types=str(
            ckpt_args.get("phrase_slot_planner_flow_slot_types", "")
        ),
        phrase_slot_guidance_enable=bool(
            int(ckpt_args.get("phrase_slot_guidance_enable", 0))
        ) or state_has_slot_guidance,
        phrase_slot_role_anchor_enable=bool(
            int(ckpt_args.get("phrase_slot_role_anchor_enable", 0))
        ) or state_has_slot_role_anchor,
        phrase_slot_role_anchor_topk=int(ckpt_args.get("phrase_slot_role_anchor_topk", 4)),
        phrase_slot_role_anchor_scale=float(ckpt_args.get("phrase_slot_role_anchor_scale", 1.0)),
        phrase_slot_role_anchor_slot_types=str(ckpt_args.get("phrase_slot_role_anchor_slot_types", "")),
        phrase_slot_decode_anchor_enable=bool(int(ckpt_args.get("phrase_slot_decode_anchor_enable", 0))),
        phrase_slot_decode_anchor_topk=int(ckpt_args.get("phrase_slot_decode_anchor_topk", 8)),
        phrase_slot_decode_anchor_scale=float(ckpt_args.get("phrase_slot_decode_anchor_scale", 1.0)),
        phrase_slot_decode_anchor_early_scale=float(ckpt_args.get("phrase_slot_decode_anchor_early_scale", 1.25)),
        phrase_slot_decode_anchor_family_scale=float(
            ckpt_args.get("phrase_slot_decode_anchor_family_scale", 0.75)
        ),
        phrase_slot_decode_anchor_stopword_penalty=float(
            ckpt_args.get("phrase_slot_decode_anchor_stopword_penalty", 0.75)
        ),
        phrase_slot_decode_anchor_stopword_steps=int(
            ckpt_args.get("phrase_slot_decode_anchor_stopword_steps", 2)
        ),
        phrase_slot_decode_anchor_debug_topk=int(ckpt_args.get("phrase_slot_decode_anchor_debug_topk", 8)),
        phrase_slot_presence_enable=bool(int(ckpt_args.get("phrase_slot_presence_enable", 0))),
        phrase_slot_presence_support_enable=bool(int(ckpt_args.get("phrase_slot_presence_support_enable", 0))),
        phrase_slot_presence_evidence_enable=bool(int(ckpt_args.get("phrase_slot_presence_evidence_enable", 0))),
        phrase_slot_presence_context_slot_types=str(
            ckpt_args.get("phrase_slot_presence_context_slot_types", "")
        ),
        phrase_slot_presence_threshold=float(ckpt_args.get("phrase_slot_presence_threshold", 0.5)),
        phrase_slot_presence_thresholds=ckpt_args.get("phrase_slot_presence_thresholds"),
        phrase_slot_active_slot_types=str(ckpt_args.get("phrase_slot_active_slot_types", "")),
        prior_head_type=str(ckpt_args.get("prior_head_type", "simple")),
        prior_head_num_heads=int(ckpt_args.get("prior_head_num_heads", 8)),
        prior_head_hidden_dim=int(ckpt_args.get("prior_head_hidden_dim", 2048)),
        prior_head_num_blocks=int(ckpt_args.get("prior_head_num_blocks", 4)),
        prior_head_num_clusters=int(ckpt_args.get("prior_head_num_clusters", 16)),
        prior_head_expansion=int(ckpt_args.get("prior_head_expansion", 2)),
        prior_head_groups=int(ckpt_args.get("prior_head_groups", 8)),
        aux_visual_enable=bool(int(getattr(args, "aux_visual_enable", 0) or 0)),
        aux_raw_global_enable=bool(int(getattr(args, "aux_raw_global_enable", 0) or 0)),
        aux_patch_enable=bool(int(getattr(args, "aux_patch_enable", 0) or 0)),
        aux_visual_raw_global_dim=int(getattr(args, "aux_visual_raw_global_dim", 512) or 512),
        aux_visual_patch_dim=int(getattr(args, "aux_visual_patch_dim", 768) or 768),
        aux_visual_prior_scale=float(getattr(args, "aux_visual_prior_scale", 0.15) or 0.15),
        aux_visual_struct_scale=float(getattr(args, "aux_visual_struct_scale", 0.10) or 0.10),
        aux_visual_memory_scale=float(getattr(args, "aux_visual_memory_scale", 0.10) or 0.10),
        vocab_size=49408,
        decoder_nhead=int(ckpt_args.get("decoder_nhead", 8)),
        d_model=int(ckpt_args.get("d_model", 512)),
        deocder_layer_nums=int(ckpt_args.get("num_layers", 3)),
        init_we=int(ckpt_args.get("init_we", 1)),
        init_lmhead=int(ckpt_args.get("init_lmhead", 1)),
        pad_token_id=0,
        bos_token_id=49406,
        eos_token_id=49407,
        frozen_we=int(ckpt_args.get("frozen_we", 1)),
        frozen_lmhead=int(ckpt_args.get("frozen_lmhead", 0)),
        **lexical_anchor_kwargs,
    )
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()

    dataset = build_dataset(
        dataset_type=dataset_type,
        features_path=features_path,
        annotations_path=annotations_path,
        split=args.split,
    )
    dataset = base.maybe_wrap_with_visual_evidence(dataset, args, split=args.split)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=False,
    )

    tokenizer = CLIPTokenizer_Custom()
    entity_token_lists, action_token_lists, attribute_token_lists, scene_token_lists, semantic_token_ids = build_semantic_token_lists(
        tokenizer=tokenizer,
        entity_vocab=entity_vocab,
        action_vocab=action_vocab,
        attribute_vocab=attribute_vocab,
        scene_vocab=scene_vocab,
    )
    print(
        f"[ISCR] semantic_token_ids={len(semantic_token_ids)} "
        f"entity_vocab={len(entity_vocab)} action_vocab={len(action_vocab)} "
        f"attribute_vocab={len(attribute_vocab)} scene_vocab={len(scene_vocab)}"
    )
    entity_semantic_ids = {int(t) for ids in entity_token_lists for t in ids}
    action_semantic_ids = {int(t) for ids in action_token_lists for t in ids}
    attribute_semantic_ids = {int(t) for ids in attribute_token_lists for t in ids}
    scene_semantic_ids = {int(t) for ids in scene_token_lists for t in ids}
    stage1_semantic_ids_default = set(entity_semantic_ids)
    stage1_semantic_ids_default.update(action_semantic_ids)

    if args.iscr_halluc_semantic_scope == "weighted":
        semantic_token_ids_hall = build_weighted_semantic_token_ids(
            entity_token_lists=entity_token_lists,
            action_token_lists=action_token_lists,
            attribute_token_lists=attribute_token_lists,
            scene_token_lists=scene_token_lists,
            entity_weight=float(args.iscr_entity_weight),
            action_weight=float(args.iscr_action_weight),
            attribute_weight=float(args.iscr_attribute_weight),
            scene_weight=float(args.iscr_scene_weight),
        )
        if not semantic_token_ids_hall:
            semantic_token_ids_hall = set(semantic_token_ids)
    else:
        semantic_token_ids_hall = set(semantic_token_ids)

    print(
        f"[ISCR] halluc_scope={args.iscr_halluc_semantic_scope} "
        f"semantic_token_ids_hall={len(semantic_token_ids_hall)}"
    )

    use_rerank = bool(int(args.iscr_rerank)) and int(args.beam_size) > 1
    use_two_stage = bool(use_rerank and int(args.iscr_two_stage_fusion))
    phrase_target_mode = str(ckpt_args.get("phrase_target_mode", "flat")).strip().lower() or "flat"
    phrase_slot_schema = str(ckpt_args.get("phrase_slot_schema", "raw")).strip().lower() or "raw"
    explain_prior_topk = max(1, int(args.iscr_prior_topk))
    if use_two_stage:
        stage1_semantic_ids = build_weighted_semantic_token_ids(
            entity_token_lists=entity_token_lists,
            action_token_lists=action_token_lists,
            attribute_token_lists=attribute_token_lists,
            scene_token_lists=scene_token_lists,
            entity_weight=float(args.iscr_entity_weight),
            action_weight=float(args.iscr_action_weight),
            attribute_weight=0.0,
            scene_weight=0.0,
        )
        if not stage1_semantic_ids:
            stage1_semantic_ids = set(stage1_semantic_ids_default)
        print(
            f"[ISCR] two_stage_fusion=1 "
            f"stage1_semantic_ids={len(stage1_semantic_ids)} "
            f"attr_weight={float(args.iscr_two_stage_attr_weight):.3f} "
            f"scene_weight={float(args.iscr_two_stage_scene_weight):.3f}"
        )
    else:
        stage1_semantic_ids = set(stage1_semantic_ids_default)
    hyp: Dict[str, List[str]] = {}
    test_loss_sum = 0.0
    test_loss_n = 0
    rerank_changed = 0
    rerank_rank_hist = Counter()
    stage2_attr_active = 0
    stage2_scene_active = 0
    stage2_any_active = 0
    stage2_attr_conf_sum = 0.0
    stage2_scene_conf_sum = 0.0
    phrase_active = 0
    collect_explain = bool(str(args.iscr_explain_jsonl).strip())
    explain_topk = max(1, int(args.iscr_explain_topk))
    explain_records: List[dict] = []
    phrase_prediction_records: List[dict] = []
    phrase_slot_text_total = 0
    phrase_slot_active_total = 0

    pbar = tqdm(loader, desc=f"Eval {args.split}")
    for batch in pbar:
        batch, aux_inputs = base.split_batch_and_aux(batch)
        vid_feat, vid_mask, caption_ids, caption_mask, _, vids, _ = batch
        vid_feat = vid_feat.to(device, non_blocking=True)
        vid_mask = vid_mask.to(device, non_blocking=True).bool()
        caption_ids = caption_ids.to(device, non_blocking=True)
        caption_mask = caption_mask.to(device, non_blocking=True)
        aux_inputs = base.move_aux_tensors_to_device(aux_inputs, device)

        generation_state = base.prepare_generation_state_if_available(model, vid_feat, vid_mask, **aux_inputs)
        entity_prob, action_prob, attribute_prob, scene_prob = predict_struct_prior_probs(
            model,
            vid_feat,
            vid_mask,
            **aux_inputs,
        )
        logits_tf = base.forward_with_generation_state(
            model,
            vid_feat,
            vid_mask,
            caption_ids,
            caption_mask,
            generation_state=generation_state,
            **aux_inputs,
        )
        target = caption_ids[:, 1:].contiguous()
        loss_test = F.cross_entropy(
            logits_tf.reshape(-1, logits_tf.size(-1)),
            target.reshape(-1),
            ignore_index=0,
        )
        test_loss_sum += float(loss_test.item())
        test_loss_n += 1

        if int(args.beam_size) > 1:
            candidates = beam_search_candidates_batch(
                model=model,
                vid_feat=vid_feat,
                vid_mask=vid_mask,
                generation_state=generation_state,
                beam_size=max(2, int(args.beam_size)),
                max_new_tokens=76,
                alpha=float(args.beam_alpha),
            )

            for i, vid in enumerate(vids):
                best_rank = 0
                best_seq = candidates[i][0][0]
                rerank_meta = None
                evidence_ids: Set[int] = set()
                stage1_evidence_ids: Set[int] = set()
                attr_evidence_ids: Set[int] = set()
                scene_evidence_ids: Set[int] = set()
                attr_conf = 0.0
                scene_conf = 0.0
                attr_active = False
                scene_active = False
                predicted_phrase_text, predicted_phrase_units, predicted_phrase_token_ids = decode_predicted_phrase(
                    tokenizer=tokenizer,
                    generation_state=generation_state,
                    index=i,
                )
                predicted_phrase_slots = decode_predicted_phrase_slots(
                    tokenizer=tokenizer,
                    generation_state=generation_state,
                    index=i,
                )
                (
                    predicted_slot_presence_probs,
                    predicted_slot_presence_pred,
                    predicted_slot_presence_forced_pred,
                    predicted_slot_presence_thresholds,
                    predicted_slot_presence_fallback_applied,
                    predicted_slot_presence_fallback_slot,
                ) = decode_predicted_slot_presence(
                    generation_state=generation_state,
                    index=i,
                )
                predicted_slot_source_names, predicted_slot_source_weights = decode_predicted_slot_planner_sources(
                    generation_state=generation_state,
                    index=i,
                )
                predicted_slot_anchor_candidates = decode_predicted_slot_anchor_candidates(
                    tokenizer=tokenizer,
                    generation_state=generation_state,
                    index=i,
                )
                word_stage = {
                    "top_entity_priors": summarize_top_priors(
                        prob_row=None if entity_prob is None else entity_prob[i],
                        vocab=entity_vocab,
                        topk=explain_prior_topk,
                        min_prob=float(args.iscr_entity_min_prob),
                    ),
                    "top_action_priors": summarize_top_priors(
                        prob_row=None if action_prob is None else action_prob[i],
                        vocab=action_vocab,
                        topk=explain_prior_topk,
                        min_prob=float(args.iscr_action_min_prob),
                    ),
                    "top_attribute_priors": summarize_top_priors(
                        prob_row=None if attribute_prob is None else attribute_prob[i],
                        vocab=attribute_vocab,
                        topk=explain_prior_topk,
                        min_prob=float(args.iscr_attribute_min_prob),
                    ),
                    "top_scene_priors": summarize_top_priors(
                        prob_row=None if scene_prob is None else scene_prob[i],
                        vocab=scene_vocab,
                        topk=explain_prior_topk,
                        min_prob=float(args.iscr_scene_min_prob),
                    ),
                }
                phrase_slots = build_phrase_slot_records(
                    slot_texts=predicted_phrase_slots,
                    presence_probs=predicted_slot_presence_probs,
                    presence_pred=predicted_slot_presence_pred,
                    presence_forced_pred=predicted_slot_presence_forced_pred,
                    presence_thresholds=predicted_slot_presence_thresholds,
                    fallback_applied=predicted_slot_presence_fallback_applied,
                    fallback_slot=predicted_slot_presence_fallback_slot,
                    phrase_slot_schema=phrase_slot_schema,
                    max_phrase_slots=max(
                        int(ckpt_args.get("max_phrase_slots", 4)),
                        len(predicted_phrase_slots),
                    ),
                    slot_source_names=predicted_slot_source_names,
                    slot_source_weights=predicted_slot_source_weights,
                    slot_anchor_candidates=predicted_slot_anchor_candidates,
                    word_stage=word_stage,
                )
                phrase_slot_text_total += sum(1 for slot in phrase_slots if slot.get("has_text"))
                phrase_slot_active_total += sum(1 for slot in phrase_slots if slot.get("active"))
                phrase_unit_token_lists, phrase_evidence_ids = build_phrase_rerank_targets(
                    tokenizer=tokenizer,
                    predicted_phrase_units=predicted_phrase_units,
                    predicted_phrase_token_ids=predicted_phrase_token_ids,
                )
                if phrase_evidence_ids or phrase_unit_token_lists:
                    phrase_active += 1
                if use_rerank:
                    if use_two_stage:
                        stage1_evidence_ids, attr_evidence_ids, scene_evidence_ids, attr_conf, scene_conf = build_two_stage_evidence_ids(
                            entity_prob_row=None if entity_prob is None else entity_prob[i],
                            action_prob_row=None if action_prob is None else action_prob[i],
                            attribute_prob_row=None if attribute_prob is None else attribute_prob[i],
                            scene_prob_row=None if scene_prob is None else scene_prob[i],
                            entity_token_lists=entity_token_lists,
                            action_token_lists=action_token_lists,
                            attribute_token_lists=attribute_token_lists,
                            scene_token_lists=scene_token_lists,
                            prior_topk=int(args.iscr_prior_topk),
                            token_topk=int(args.iscr_rerank_topk),
                            threshold=float(args.iscr_evidence_threshold),
                            entity_threshold=float(args.iscr_entity_threshold),
                            action_threshold=float(args.iscr_action_threshold),
                            attribute_threshold=float(args.iscr_attribute_threshold),
                            scene_threshold=float(args.iscr_scene_threshold),
                            entity_weight=float(args.iscr_entity_weight),
                            action_weight=float(args.iscr_action_weight),
                            attribute_weight=float(args.iscr_two_stage_attr_weight),
                            scene_weight=float(args.iscr_two_stage_scene_weight),
                            entity_min_prob=float(args.iscr_entity_min_prob),
                            action_min_prob=float(args.iscr_action_min_prob),
                            attribute_min_prob=float(args.iscr_attribute_min_prob),
                            scene_min_prob=float(args.iscr_scene_min_prob),
                        )
                        attr_active = bool(attr_evidence_ids) and float(attr_conf) > 0.0
                        scene_active = bool(scene_evidence_ids) and float(scene_conf) > 0.0
                        if attr_active:
                            stage2_attr_active += 1
                        if scene_active:
                            stage2_scene_active += 1
                        if attr_active or scene_active:
                            stage2_any_active += 1
                        stage2_attr_conf_sum += float(attr_conf)
                        stage2_scene_conf_sum += float(scene_conf)
                        if collect_explain:
                            best_seq, best_rank, rerank_meta = evidence_rerank_tokens_two_stage(
                                candidates=candidates[i],
                                stage1_evidence_ids=stage1_evidence_ids,
                                stage1_semantic_ids=stage1_semantic_ids,
                                attribute_evidence_ids=attr_evidence_ids,
                                attribute_semantic_ids=attribute_semantic_ids,
                                scene_evidence_ids=scene_evidence_ids,
                                scene_semantic_ids=scene_semantic_ids,
                                alpha=float(args.iscr_rerank_alpha),
                                lambda_cov=float(args.iscr_rerank_lambda_cov),
                                lambda_hall=float(args.iscr_rerank_lambda_hall),
                                attribute_weight=float(args.iscr_two_stage_attr_weight),
                                scene_weight=float(args.iscr_two_stage_scene_weight),
                                attribute_confidence=float(attr_conf),
                                scene_confidence=float(scene_conf),
                                phrase_evidence_ids=phrase_evidence_ids,
                                phrase_unit_token_lists=phrase_unit_token_lists,
                                phrase_weight=float(args.iscr_phrase_weight),
                                base_norm_mode=str(args.iscr_rerank_base_norm),
                                return_meta=True,
                            )
                        else:
                            best_seq, best_rank = evidence_rerank_tokens_two_stage(
                                candidates=candidates[i],
                                stage1_evidence_ids=stage1_evidence_ids,
                                stage1_semantic_ids=stage1_semantic_ids,
                                attribute_evidence_ids=attr_evidence_ids,
                                attribute_semantic_ids=attribute_semantic_ids,
                                scene_evidence_ids=scene_evidence_ids,
                                scene_semantic_ids=scene_semantic_ids,
                                alpha=float(args.iscr_rerank_alpha),
                                lambda_cov=float(args.iscr_rerank_lambda_cov),
                                lambda_hall=float(args.iscr_rerank_lambda_hall),
                                attribute_weight=float(args.iscr_two_stage_attr_weight),
                                scene_weight=float(args.iscr_two_stage_scene_weight),
                                attribute_confidence=float(attr_conf),
                                scene_confidence=float(scene_conf),
                                phrase_evidence_ids=phrase_evidence_ids,
                                phrase_unit_token_lists=phrase_unit_token_lists,
                                phrase_weight=float(args.iscr_phrase_weight),
                                base_norm_mode=str(args.iscr_rerank_base_norm),
                                return_meta=False,
                            )
                    else:
                        evidence_ids = build_evidence_token_ids(
                            entity_prob_row=None if entity_prob is None else entity_prob[i],
                            action_prob_row=None if action_prob is None else action_prob[i],
                            attribute_prob_row=None if attribute_prob is None else attribute_prob[i],
                            scene_prob_row=None if scene_prob is None else scene_prob[i],
                            entity_token_lists=entity_token_lists,
                            action_token_lists=action_token_lists,
                            attribute_token_lists=attribute_token_lists,
                            scene_token_lists=scene_token_lists,
                            prior_topk=int(args.iscr_prior_topk),
                            token_topk=int(args.iscr_rerank_topk),
                            threshold=float(args.iscr_evidence_threshold),
                            entity_threshold=float(args.iscr_entity_threshold),
                            action_threshold=float(args.iscr_action_threshold),
                            attribute_threshold=float(args.iscr_attribute_threshold),
                            scene_threshold=float(args.iscr_scene_threshold),
                            entity_weight=float(args.iscr_entity_weight),
                            action_weight=float(args.iscr_action_weight),
                            attribute_weight=float(args.iscr_attribute_weight),
                            scene_weight=float(args.iscr_scene_weight),
                            entity_min_prob=float(args.iscr_entity_min_prob),
                            action_min_prob=float(args.iscr_action_min_prob),
                            attribute_min_prob=float(args.iscr_attribute_min_prob),
                            scene_min_prob=float(args.iscr_scene_min_prob),
                        )
                        if collect_explain:
                            best_seq, best_rank, rerank_meta = evidence_rerank_tokens(
                                candidates=candidates[i],
                                evidence_ids=evidence_ids,
                                semantic_token_ids=semantic_token_ids_hall,
                                alpha=float(args.iscr_rerank_alpha),
                                lambda_cov=float(args.iscr_rerank_lambda_cov),
                                lambda_hall=float(args.iscr_rerank_lambda_hall),
                                phrase_evidence_ids=phrase_evidence_ids,
                                phrase_unit_token_lists=phrase_unit_token_lists,
                                phrase_weight=float(args.iscr_phrase_weight),
                                base_norm_mode=str(args.iscr_rerank_base_norm),
                                return_meta=True,
                            )
                        else:
                            best_seq, best_rank = evidence_rerank_tokens(
                                candidates=candidates[i],
                                evidence_ids=evidence_ids,
                                semantic_token_ids=semantic_token_ids_hall,
                                alpha=float(args.iscr_rerank_alpha),
                                lambda_cov=float(args.iscr_rerank_lambda_cov),
                                lambda_hall=float(args.iscr_rerank_lambda_hall),
                                phrase_evidence_ids=phrase_evidence_ids,
                                phrase_unit_token_lists=phrase_unit_token_lists,
                                phrase_weight=float(args.iscr_phrase_weight),
                                base_norm_mode=str(args.iscr_rerank_base_norm),
                                return_meta=False,
                            )

                if best_rank != 0:
                    rerank_changed += 1
                rerank_rank_hist[int(best_rank)] += 1

                text = tokenizer.decode(best_seq, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                hyp[str(vid)] = [text]
                phrase_prediction_records.append(
                    build_multistage_prediction_record(
                        video_id=str(vid),
                        final_caption=str(text),
                        predicted_phrase_text=str(predicted_phrase_text),
                        predicted_phrase_units=[str(unit) for unit in predicted_phrase_units],
                        predicted_phrase_token_ids=[int(t) for t in predicted_phrase_token_ids],
                        predicted_slot_presence_probs=[float(x) for x in predicted_slot_presence_probs],
                        predicted_slot_presence_pred=[int(x) for x in predicted_slot_presence_pred],
                        predicted_slot_presence_forced_pred=[int(x) for x in predicted_slot_presence_forced_pred],
                        predicted_slot_presence_thresholds=[float(x) for x in predicted_slot_presence_thresholds],
                        predicted_slot_presence_fallback_applied=bool(predicted_slot_presence_fallback_applied),
                        predicted_slot_presence_fallback_slot=int(predicted_slot_presence_fallback_slot),
                        phrase_slots=phrase_slots,
                        word_stage=word_stage,
                    )
                )

                if collect_explain:
                    if rerank_meta is None:
                        base_scores = [float(score) for _seq, score in candidates[i]]
                        base_scores_used = _normalize_base_scores(base_scores, mode=str(args.iscr_rerank_base_norm))
                        rerank_meta = {
                            "mode": "disabled",
                            "base_norm_mode": str(args.iscr_rerank_base_norm).lower(),
                            "candidates": [
                                {
                                    "rank": int(rank),
                                    "base_score_raw": float(base_scores[rank]),
                                    "base_score_used": float(base_scores_used[rank]),
                                    "coverage_stage1": 0.0,
                                    "hall_stage1": 0.0,
                                    "attr_coverage": 0.0,
                                    "attr_hall": 0.0,
                                    "scene_coverage": 0.0,
                                    "scene_hall": 0.0,
                                    "phrase_token_coverage": 0.0,
                                    "phrase_unit_coverage": 0.0,
                                    "phrase_reward": 0.0,
                                    "final_score": float(base_scores_used[rank]),
                                    "token_ids": [int(t) for t in seq],
                                }
                                for rank, (seq, _score) in enumerate(candidates[i])
                            ],
                        }

                    cand_details = []
                    for cand in rerank_meta.get("candidates", [])[:explain_topk]:
                        seq_ids = [int(t) for t in cand.get("token_ids", [])]
                        cand_text = tokenizer.decode(seq_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        cand_details.append(
                            {
                                "rank": int(cand.get("rank", 0)),
                                "base_score_raw": float(cand.get("base_score_raw", 0.0)),
                                "base_score_used": float(cand.get("base_score_used", 0.0)),
                                "coverage_stage1": float(cand.get("coverage_stage1", 0.0)),
                                "hall_stage1": float(cand.get("hall_stage1", 0.0)),
                                "attr_coverage": float(cand.get("attr_coverage", 0.0)),
                                "attr_hall": float(cand.get("attr_hall", 0.0)),
                                "scene_coverage": float(cand.get("scene_coverage", 0.0)),
                                "scene_hall": float(cand.get("scene_hall", 0.0)),
                                "phrase_token_coverage": float(cand.get("phrase_token_coverage", 0.0)),
                                "phrase_unit_coverage": float(cand.get("phrase_unit_coverage", 0.0)),
                                "phrase_reward": float(cand.get("phrase_reward", 0.0)),
                                "final_score": float(cand.get("final_score", 0.0)),
                                "token_count": int(len(seq_ids)),
                                "text": str(cand_text),
                            }
                        )

                    record = {
                        "video_id": str(vid),
                        "mode": str(rerank_meta.get("mode", "disabled")),
                        "rerank_enabled": bool(use_rerank),
                        "two_stage_enabled": bool(use_two_stage),
                        "chosen_rank": int(best_rank),
                        "chosen_text": str(text),
                        "final_caption": str(text),
                        "predicted_phrase_text": str(predicted_phrase_text),
                        "predicted_phrase_units": [str(unit) for unit in predicted_phrase_units],
                        "beam_size": int(args.beam_size),
                        "base_norm_mode": str(rerank_meta.get("base_norm_mode", str(args.iscr_rerank_base_norm).lower())),
                        "candidates": cand_details,
                        "phrase_evidence_count": int(len(phrase_evidence_ids)),
                        "phrase_unit_count": int(len(phrase_unit_token_lists)),
                        "phrase_scale": float(rerank_meta.get("phrase_scale", 0.0)),
                        "phrase_target_mode": phrase_target_mode,
                        "phrase_slot_schema": phrase_slot_schema,
                    }
                    record.update(
                        {
                            "top_entity_priors": [dict(item) for item in word_stage["top_entity_priors"]],
                            "top_action_priors": [dict(item) for item in word_stage["top_action_priors"]],
                            "top_attribute_priors": [dict(item) for item in word_stage["top_attribute_priors"]],
                            "top_scene_priors": [dict(item) for item in word_stage["top_scene_priors"]],
                            "phrase_slots": [dict(slot) for slot in phrase_slots],
                            "word_stage": {
                                key: [dict(item) for item in value]
                                for key, value in word_stage.items()
                            },
                            "phrase_stage": build_phrase_stage_payload(
                                predicted_phrase_text=str(predicted_phrase_text),
                                predicted_phrase_units=[str(unit) for unit in predicted_phrase_units],
                                predicted_phrase_token_ids=[int(t) for t in predicted_phrase_token_ids],
                                predicted_slot_presence_probs=[float(x) for x in predicted_slot_presence_probs],
                                predicted_slot_presence_pred=[int(x) for x in predicted_slot_presence_pred],
                                predicted_slot_presence_forced_pred=[int(x) for x in predicted_slot_presence_forced_pred],
                                predicted_slot_presence_thresholds=[float(x) for x in predicted_slot_presence_thresholds],
                                predicted_slot_presence_fallback_applied=bool(predicted_slot_presence_fallback_applied),
                                predicted_slot_presence_fallback_slot=int(predicted_slot_presence_fallback_slot),
                                phrase_slots=phrase_slots,
                            ),
                            "sentence_stage": {
                                "final_caption": str(text),
                            },
                        }
                    )
                    if use_two_stage:
                        record.update(
                            {
                                "stage1_evidence_count": int(len(stage1_evidence_ids)),
                                "attr_evidence_count": int(len(attr_evidence_ids)),
                                "scene_evidence_count": int(len(scene_evidence_ids)),
                                "attr_active": bool(attr_active),
                                "scene_active": bool(scene_active),
                                "attr_confidence": float(attr_conf),
                                "scene_confidence": float(scene_conf),
                                "attr_scale": float(rerank_meta.get("attr_scale", 0.0)),
                                "scene_scale": float(rerank_meta.get("scene_scale", 0.0)),
                            }
                        )
                    else:
                        record.update(
                            {
                                "evidence_count": int(len(evidence_ids)),
                            }
                        )
                    explain_records.append(record)
        else:
            texts = base.greedy_generate_batch(
                model=model,
                vid_feat=vid_feat,
                vid_mask=vid_mask,
                tokenizer=tokenizer,
                generation_state=generation_state,
                max_new_tokens=76,
                top_p=0.9,
                temperature=0.0,
            )
            for i, (vid, text) in enumerate(zip(vids, texts)):
                hyp[str(vid)] = [text]
                predicted_phrase_text, predicted_phrase_units, predicted_phrase_token_ids = decode_predicted_phrase(
                    tokenizer=tokenizer,
                    generation_state=generation_state,
                    index=i,
                )
                predicted_phrase_slots = decode_predicted_phrase_slots(
                    tokenizer=tokenizer,
                    generation_state=generation_state,
                    index=i,
                )
                (
                    predicted_slot_presence_probs,
                    predicted_slot_presence_pred,
                    predicted_slot_presence_forced_pred,
                    predicted_slot_presence_thresholds,
                    predicted_slot_presence_fallback_applied,
                    predicted_slot_presence_fallback_slot,
                ) = decode_predicted_slot_presence(
                    generation_state=generation_state,
                    index=i,
                )
                predicted_slot_source_names, predicted_slot_source_weights = decode_predicted_slot_planner_sources(
                    generation_state=generation_state,
                    index=i,
                )
                predicted_slot_anchor_candidates = decode_predicted_slot_anchor_candidates(
                    tokenizer=tokenizer,
                    generation_state=generation_state,
                    index=i,
                )
                word_stage = {
                    "top_entity_priors": summarize_top_priors(
                        prob_row=None if entity_prob is None else entity_prob[i],
                        vocab=entity_vocab,
                        topk=explain_prior_topk,
                        min_prob=float(args.iscr_entity_min_prob),
                    ),
                    "top_action_priors": summarize_top_priors(
                        prob_row=None if action_prob is None else action_prob[i],
                        vocab=action_vocab,
                        topk=explain_prior_topk,
                        min_prob=float(args.iscr_action_min_prob),
                    ),
                    "top_attribute_priors": summarize_top_priors(
                        prob_row=None if attribute_prob is None else attribute_prob[i],
                        vocab=attribute_vocab,
                        topk=explain_prior_topk,
                        min_prob=float(args.iscr_attribute_min_prob),
                    ),
                    "top_scene_priors": summarize_top_priors(
                        prob_row=None if scene_prob is None else scene_prob[i],
                        vocab=scene_vocab,
                        topk=explain_prior_topk,
                        min_prob=float(args.iscr_scene_min_prob),
                    ),
                }
                phrase_slots = build_phrase_slot_records(
                    slot_texts=predicted_phrase_slots,
                    presence_probs=predicted_slot_presence_probs,
                    presence_pred=predicted_slot_presence_pred,
                    presence_forced_pred=predicted_slot_presence_forced_pred,
                    presence_thresholds=predicted_slot_presence_thresholds,
                    fallback_applied=predicted_slot_presence_fallback_applied,
                    fallback_slot=predicted_slot_presence_fallback_slot,
                    phrase_slot_schema=phrase_slot_schema,
                    max_phrase_slots=max(
                        int(ckpt_args.get("max_phrase_slots", 4)),
                        len(predicted_phrase_slots),
                    ),
                    slot_source_names=predicted_slot_source_names,
                    slot_source_weights=predicted_slot_source_weights,
                    slot_anchor_candidates=predicted_slot_anchor_candidates,
                    word_stage=word_stage,
                )
                phrase_slot_text_total += sum(1 for slot in phrase_slots if slot.get("has_text"))
                phrase_slot_active_total += sum(1 for slot in phrase_slots if slot.get("active"))
                phrase_prediction_records.append(
                    build_multistage_prediction_record(
                        video_id=str(vid),
                        final_caption=str(text),
                        predicted_phrase_text=str(predicted_phrase_text),
                        predicted_phrase_units=[str(unit) for unit in predicted_phrase_units],
                        predicted_phrase_token_ids=[int(t) for t in predicted_phrase_token_ids],
                        predicted_slot_presence_probs=[float(x) for x in predicted_slot_presence_probs],
                        predicted_slot_presence_pred=[int(x) for x in predicted_slot_presence_pred],
                        predicted_slot_presence_forced_pred=[int(x) for x in predicted_slot_presence_forced_pred],
                        predicted_slot_presence_thresholds=[float(x) for x in predicted_slot_presence_thresholds],
                        predicted_slot_presence_fallback_applied=bool(predicted_slot_presence_fallback_applied),
                        predicted_slot_presence_fallback_slot=int(predicted_slot_presence_fallback_slot),
                        phrase_slots=phrase_slots,
                        word_stage=word_stage,
                    )
                )
                if collect_explain:
                    explain_records.append(
                        {
                            "video_id": str(vid),
                            "mode": "greedy",
                            "rerank_enabled": False,
                            "two_stage_enabled": False,
                            "chosen_rank": 0,
                            "chosen_text": str(text),
                            "final_caption": str(text),
                            "predicted_phrase_text": str(predicted_phrase_text),
                            "predicted_phrase_units": [str(unit) for unit in predicted_phrase_units],
                            "beam_size": int(args.beam_size),
                            "base_norm_mode": str(args.iscr_rerank_base_norm).lower(),
                            "candidates": [],
                            "phrase_evidence_count": 0,
                            "phrase_unit_count": int(len(predicted_phrase_units)),
                            "phrase_scale": 0.0,
                            "phrase_target_mode": phrase_target_mode,
                            "phrase_slot_schema": phrase_slot_schema,
                            "top_entity_priors": [dict(item) for item in word_stage["top_entity_priors"]],
                            "top_action_priors": [dict(item) for item in word_stage["top_action_priors"]],
                            "top_attribute_priors": [dict(item) for item in word_stage["top_attribute_priors"]],
                            "top_scene_priors": [dict(item) for item in word_stage["top_scene_priors"]],
                            "phrase_slots": [dict(slot) for slot in phrase_slots],
                            "word_stage": {
                                key: [dict(item) for item in value]
                                for key, value in word_stage.items()
                            },
                            "phrase_stage": build_phrase_stage_payload(
                                predicted_phrase_text=str(predicted_phrase_text),
                                predicted_phrase_units=[str(unit) for unit in predicted_phrase_units],
                                predicted_phrase_token_ids=[int(t) for t in predicted_phrase_token_ids],
                                predicted_slot_presence_probs=[float(x) for x in predicted_slot_presence_probs],
                                predicted_slot_presence_pred=[int(x) for x in predicted_slot_presence_pred],
                                predicted_slot_presence_forced_pred=[int(x) for x in predicted_slot_presence_forced_pred],
                                predicted_slot_presence_thresholds=[float(x) for x in predicted_slot_presence_thresholds],
                                predicted_slot_presence_fallback_applied=bool(predicted_slot_presence_fallback_applied),
                                predicted_slot_presence_fallback_slot=int(predicted_slot_presence_fallback_slot),
                                phrase_slots=phrase_slots,
                            ),
                            "sentence_stage": {
                                "final_caption": str(text),
                            },
                        }
                    )

    refs = get_references_from_dataset(dataset)
    scores = base.compute_metrics_no_spice(refs, hyp)
    loss_key = f"{args.split.upper()}_LOSS"
    scores[loss_key] = float(test_loss_sum / max(1, test_loss_n))

    run_dir = ckpt_path.parent.parent
    eval_dir = run_dir / f"{args.split}_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.split}_from_{ckpt_path.stem}_beam{int(args.beam_size)}"
    if use_rerank:
        suffix += "_iscr_pred"
    hyp_path = eval_dir / f"{suffix}_hyp.json"
    ref_path = eval_dir / f"{suffix}_ref.json"
    with hyp_path.open("w", encoding="utf-8") as f:
        json.dump(hyp, f, ensure_ascii=False, indent=2)
    with ref_path.open("w", encoding="utf-8") as f:
        json.dump(refs, f, ensure_ascii=False, indent=2)

    phrase_predictions_path = None
    has_phrase_predictions = any(rec.get("predicted_phrase_token_ids") for rec in phrase_prediction_records)
    if has_phrase_predictions:
        phrase_predictions_path = (
            Path(args.phrase_predictions_jsonl).resolve()
            if str(args.phrase_predictions_jsonl).strip()
            else eval_dir / f"{suffix}_phrases.jsonl"
        )
        phrase_predictions_path.parent.mkdir(parents=True, exist_ok=True)
        with phrase_predictions_path.open("w", encoding="utf-8") as f:
            for rec in phrase_prediction_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    explain_path = None
    if collect_explain:
        explain_path = Path(args.iscr_explain_jsonl).resolve()
        explain_path.parent.mkdir(parents=True, exist_ok=True)
        with explain_path.open("w", encoding="utf-8") as f:
            for rec in explain_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    out_json = Path(args.output_json).resolve() if args.output_json else run_dir / f"eval_{args.split}_{ckpt_path.stem}_iscr.json"
    result = {
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "dataset_type": dataset_type,
        "beam_size": int(args.beam_size),
        "beam_alpha": float(args.beam_alpha),
        "phrase_target_mode": phrase_target_mode,
        "phrase_slot_schema": phrase_slot_schema,
        "iscr_rerank": bool(use_rerank),
        "iscr_rerank_alpha": float(args.iscr_rerank_alpha),
        "iscr_rerank_lambda_cov": float(args.iscr_rerank_lambda_cov),
        "iscr_rerank_lambda_hall": float(args.iscr_rerank_lambda_hall),
        "iscr_rerank_base_norm": str(args.iscr_rerank_base_norm).lower(),
        "iscr_prior_topk": int(args.iscr_prior_topk),
        "iscr_rerank_topk": int(args.iscr_rerank_topk),
        "iscr_evidence_threshold": float(args.iscr_evidence_threshold),
        "iscr_entity_threshold": float(args.iscr_entity_threshold),
        "iscr_action_threshold": float(args.iscr_action_threshold),
        "iscr_attribute_threshold": float(args.iscr_attribute_threshold),
        "iscr_scene_threshold": float(args.iscr_scene_threshold),
        "iscr_entity_weight": float(args.iscr_entity_weight),
        "iscr_action_weight": float(args.iscr_action_weight),
        "iscr_attribute_weight": float(args.iscr_attribute_weight),
        "iscr_scene_weight": float(args.iscr_scene_weight),
        "iscr_halluc_semantic_scope": str(args.iscr_halluc_semantic_scope),
        "iscr_entity_min_prob": float(args.iscr_entity_min_prob),
        "iscr_action_min_prob": float(args.iscr_action_min_prob),
        "iscr_attribute_min_prob": float(args.iscr_attribute_min_prob),
        "iscr_scene_min_prob": float(args.iscr_scene_min_prob),
        "iscr_two_stage_fusion": bool(use_two_stage),
        "iscr_two_stage_attr_weight": float(args.iscr_two_stage_attr_weight),
        "iscr_two_stage_scene_weight": float(args.iscr_two_stage_scene_weight),
        "iscr_phrase_weight": float(args.iscr_phrase_weight),
        "strict_no_text_leak": True,
        "evidence_source": "pred_struct_prior",
        "metrics": {k: float(v) for k, v in scores.items()},
        "rerank_stats": {
            "changed_count": int(rerank_changed),
            "total": int(sum(rerank_rank_hist.values())),
            "changed_ratio": float(rerank_changed / max(1, sum(rerank_rank_hist.values()))),
            "best_rank_hist": {str(k): int(v) for k, v in sorted(rerank_rank_hist.items())},
            "semantic_token_ids": int(len(semantic_token_ids)),
            "semantic_token_ids_hall": int(len(semantic_token_ids_hall)),
            "two_stage_attr_active_count": int(stage2_attr_active),
            "two_stage_scene_active_count": int(stage2_scene_active),
            "two_stage_any_active_count": int(stage2_any_active),
            "phrase_active_count": int(phrase_active),
            "two_stage_attr_active_ratio": float(stage2_attr_active / max(1, sum(rerank_rank_hist.values()))),
            "two_stage_scene_active_ratio": float(stage2_scene_active / max(1, sum(rerank_rank_hist.values()))),
            "two_stage_any_active_ratio": float(stage2_any_active / max(1, sum(rerank_rank_hist.values()))),
            "phrase_active_ratio": float(phrase_active / max(1, sum(rerank_rank_hist.values()))),
            "two_stage_attr_conf_mean": float(stage2_attr_conf_sum / max(1, sum(rerank_rank_hist.values()))),
            "two_stage_scene_conf_mean": float(stage2_scene_conf_sum / max(1, sum(rerank_rank_hist.values()))),
            "phrase_slot_text_mean": float(phrase_slot_text_total / max(1, len(phrase_prediction_records))),
            "phrase_slot_active_mean": float(phrase_slot_active_total / max(1, len(phrase_prediction_records))),
        },
        "hyp_path": str(hyp_path),
        "ref_path": str(ref_path),
    }
    if phrase_predictions_path is not None:
        result["phrase_predictions_jsonl"] = str(phrase_predictions_path)
        result["phrase_prediction_records"] = int(len(phrase_prediction_records))
    if collect_explain and explain_path is not None:
        result["iscr_explain_jsonl"] = str(explain_path)
        result["iscr_explain_records"] = int(len(explain_records))
        result["iscr_explain_topk"] = int(explain_topk)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
