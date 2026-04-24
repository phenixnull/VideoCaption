# -*- coding: utf-8 -*-
"""Structured-caption model wrapper.

Adds video-conditioned structured priors plus an optional real phrase decoder.
The phrase decoder can emit phrase text explicitly and later expose a no-leak
conditioning signal to the final sentence decoder.
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from dataloaders.dataset_structured_caption import StructuredCaptionDataset
from models import CaptionModel_Base
from structured_prior_heads import StructuredAttentionNextVLADPriorHead, StructuredMultiSemanticPriorHead


class StructuredCaptionModel(nn.Module):
    _TYPED_SLOT_SOURCE_PRIORS = {
        "subject_action": {
            "entity": 1.25,
            "action": 1.75,
            "attribute": 0.25,
            "scene": -0.25,
            "struct": 0.50,
        },
        "object_passive": {
            "entity": 1.60,
            "action": 1.20,
            "attribute": 0.10,
            "scene": -0.30,
            "struct": 0.35,
        },
        "relation_scene": {
            "entity": -0.25,
            "action": 0.55,
            "attribute": 0.20,
            "scene": 1.80,
            "struct": 0.55,
        },
        "attribute_misc": {
            "entity": 0.45,
            "action": -0.25,
            "attribute": 1.80,
            "scene": 0.35,
            "struct": 0.25,
        },
        "extra": {
            "entity": 0.40,
            "action": 0.40,
            "attribute": 0.25,
            "scene": 0.25,
            "struct": 0.50,
        },
        "generic": {
            "entity": 0.30,
            "action": 0.30,
            "attribute": 0.30,
            "scene": 0.30,
            "struct": 0.30,
        },
    }
    _TYPED_RICH_SLOT_SOURCE_PRIORS = {
        "subject_action": {
            "entity": 1.10,
            "action": 1.85,
            "attribute": 0.20,
            "scene": -0.30,
            "struct": 0.50,
        },
        "object_passive": {
            "entity": 1.75,
            "action": 1.10,
            "attribute": 0.10,
            "scene": -0.30,
            "struct": 0.35,
        },
        "subject_entity": {
            "entity": 1.80,
            "action": 0.15,
            "attribute": 0.65,
            "scene": -0.20,
            "struct": 0.40,
        },
        "object_entity": {
            "entity": 1.65,
            "action": 0.10,
            "attribute": 0.90,
            "scene": -0.15,
            "struct": 0.35,
        },
        "relation_detail": {
            "entity": 0.45,
            "action": 0.85,
            "attribute": 0.95,
            "scene": 0.20,
            "struct": 0.60,
        },
        "scene_context": {
            "entity": -0.15,
            "action": 0.15,
            "attribute": 0.35,
            "scene": 1.95,
            "struct": 0.65,
        },
        "extra": {
            "entity": 0.40,
            "action": 0.40,
            "attribute": 0.25,
            "scene": 0.25,
            "struct": 0.50,
        },
        "generic": {
            "entity": 0.30,
            "action": 0.30,
            "attribute": 0.30,
            "scene": 0.30,
            "struct": 0.30,
        },
    }
    _TYPED_RICH_SEMANTIC_SLOT_SOURCE_PRIORS = {
        "subject_action": {
            "entity": 1.10,
            "action": 1.85,
            "attribute": 0.20,
            "scene": -0.30,
            "struct": 0.50,
        },
        "object_passive": {
            "entity": 1.75,
            "action": 1.10,
            "attribute": 0.10,
            "scene": -0.30,
            "struct": 0.35,
        },
        "subject_entity": {
            "entity": 1.80,
            "action": 0.15,
            "attribute": 0.65,
            "scene": -0.20,
            "struct": 0.40,
        },
        "object_entity": {
            "entity": 1.65,
            "action": 0.10,
            "attribute": 0.90,
            "scene": -0.15,
            "struct": 0.35,
        },
        "relation_detail": {
            "entity": 0.35,
            "action": 0.95,
            "attribute": 0.55,
            "scene": 0.15,
            "struct": 0.60,
        },
        "instrument_detail": {
            "entity": 1.10,
            "action": 0.95,
            "attribute": 0.75,
            "scene": -0.20,
            "struct": 0.65,
        },
        "entity_modifier": {
            "entity": 1.20,
            "action": 0.10,
            "attribute": 1.25,
            "scene": 0.10,
            "struct": 0.45,
        },
        "scene_context": {
            "entity": -0.15,
            "action": 0.15,
            "attribute": 0.35,
            "scene": 1.95,
            "struct": 0.65,
        },
        "extra": {
            "entity": 0.40,
            "action": 0.40,
            "attribute": 0.25,
            "scene": 0.25,
            "struct": 0.50,
        },
        "generic": {
            "entity": 0.30,
            "action": 0.30,
            "attribute": 0.30,
            "scene": 0.30,
            "struct": 0.30,
        },
    }
    _TYPED_RICH_ROLEAWARE_SLOT_SOURCE_PRIORS = {
        "subject_action": {
            "entity": 1.10,
            "action": 1.85,
            "attribute": 0.20,
            "scene": -0.30,
            "struct": 0.50,
        },
        "object_passive": {
            "entity": 1.75,
            "action": 1.10,
            "attribute": 0.10,
            "scene": -0.30,
            "struct": 0.35,
        },
        "subject_entity": {
            "entity": 1.80,
            "action": 0.15,
            "attribute": 0.65,
            "scene": -0.20,
            "struct": 0.40,
        },
        "object_entity": {
            "entity": 1.65,
            "action": 0.10,
            "attribute": 0.90,
            "scene": -0.15,
            "struct": 0.35,
        },
        "subject_modifier": {
            "entity": 1.25,
            "action": 0.10,
            "attribute": 1.35,
            "scene": 0.05,
            "struct": 0.45,
        },
        "object_modifier": {
            "entity": 1.15,
            "action": 0.10,
            "attribute": 1.45,
            "scene": 0.10,
            "struct": 0.45,
        },
        "relation_detail": {
            "entity": 0.35,
            "action": 0.95,
            "attribute": 0.55,
            "scene": 0.15,
            "struct": 0.60,
        },
        "instrument_detail": {
            "entity": 1.10,
            "action": 0.95,
            "attribute": 0.75,
            "scene": -0.20,
            "struct": 0.65,
        },
        "scene_context": {
            "entity": -0.15,
            "action": 0.15,
            "attribute": 0.35,
            "scene": 1.95,
            "struct": 0.65,
        },
        "extra": {
            "entity": 0.40,
            "action": 0.40,
            "attribute": 0.25,
            "scene": 0.25,
            "struct": 0.50,
        },
        "generic": {
            "entity": 0.30,
            "action": 0.30,
            "attribute": 0.30,
            "scene": 0.30,
            "struct": 0.30,
        },
    }
    _FAMILY4_COMPACT_SLOT_SOURCE_PRIORS = {
        "subject_entity": {
            "entity": 1.80,
            "action": 0.15,
            "attribute": 0.65,
            "scene": -0.20,
            "struct": 0.40,
        },
        "object_entity": {
            "entity": 1.65,
            "action": 0.10,
            "attribute": 0.90,
            "scene": -0.15,
            "struct": 0.35,
        },
        "subject_action": {
            "entity": 1.10,
            "action": 1.85,
            "attribute": 0.20,
            "scene": -0.30,
            "struct": 0.50,
        },
        "scene_context": {
            "entity": -0.15,
            "action": 0.15,
            "attribute": 0.35,
            "scene": 1.95,
            "struct": 0.65,
        },
        "extra": {
            "entity": 0.40,
            "action": 0.40,
            "attribute": 0.25,
            "scene": 0.25,
            "struct": 0.50,
        },
        "generic": {
            "entity": 0.30,
            "action": 0.30,
            "attribute": 0.30,
            "scene": 0.30,
            "struct": 0.30,
        },
    }
    _DEFAULT_PHRASE_CONDITION_CORE_SLOT_TYPES = (
        "subject_action",
        "subject_entity",
        "object_entity",
        "object_passive",
    )
    _DEFAULT_PHRASE_CONDITION_AUX_SLOT_TYPES = (
        "subject_modifier",
        "object_modifier",
        "entity_modifier",
        "relation_detail",
        "instrument_detail",
        "scene_context",
    )
    _DEFAULT_PHRASE_CONDITION_FAMILY_GROUPS = (
        (
            "subject_family",
            (
                "subject_modifier",
                "subject_entity",
            ),
        ),
        (
            "action_family",
            (
                "subject_action",
            ),
        ),
        (
            "object_family",
            (
                "object_modifier",
                "object_entity",
                "object_passive",
                "instrument_detail",
                "entity_modifier",
            ),
        ),
        (
            "context_family",
            (
                "relation_detail",
                "scene_context",
            ),
        ),
    )

    @classmethod
    def get_slot_source_prior_map(cls, phrase_slot_schema: str) -> Optional[Dict[str, Dict[str, float]]]:
        schema = str(phrase_slot_schema).strip().lower() or "raw"
        if schema == "typed":
            return cls._TYPED_SLOT_SOURCE_PRIORS
        if schema == "typed_rich":
            return cls._TYPED_RICH_SLOT_SOURCE_PRIORS
        if schema == "typed_rich_semantic":
            return cls._TYPED_RICH_SEMANTIC_SLOT_SOURCE_PRIORS
        if schema == "typed_rich_roleaware":
            return cls._TYPED_RICH_ROLEAWARE_SLOT_SOURCE_PRIORS
        if schema == "family4_compact":
            return cls._FAMILY4_COMPACT_SLOT_SOURCE_PRIORS
        return None

    @staticmethod
    def _parse_slot_type_list(slot_types: str) -> Tuple[str, ...]:
        parsed: List[str] = []
        if slot_types:
            for raw_slot_type in str(slot_types).split(","):
                slot_type = str(raw_slot_type).strip().lower()
                if slot_type and slot_type not in parsed:
                    parsed.append(slot_type)
        return tuple(parsed)

    def __init__(
        self,
        *,
        entity_dim: int,
        action_dim: int,
        attribute_dim: int = 0,
        scene_dim: int = 0,
        prior_dropout: float = 0.1,
        struct_condition: bool = True,
        struct_condition_scale: float = 0.35,
        struct_condition_query_bridge_enable: bool = False,
        struct_condition_query_bridge_num_queries: int = 4,
        struct_condition_query_bridge_scale: float = 0.15,
        struct_condition_query_bridge_memory_enable: bool = False,
        struct_condition_query_bridge_memory_scale: float = 0.15,
        struct_condition_query_bridge_hidden_enable: bool = False,
        struct_condition_query_bridge_hidden_scale: float = 0.15,
        phrase_decoder_enable: bool = False,
        phrase_condition_enable: bool = False,
        phrase_condition_slot_aware_enable: bool = False,
        phrase_condition_slot_selective_enable: bool = False,
        phrase_condition_slot_residual_enable: bool = False,
        phrase_decoder_layers: int = 2,
        phrase_condition_scale: float = 0.25,
        phrase_condition_aux_scale: float = 0.15,
        phrase_condition_slot_residual_scale: float = 0.15,
        phrase_condition_core_slot_types: str = "",
        phrase_condition_aux_slot_types: str = "",
        phrase_condition_slot_residual_slot_types: str = "",
        phrase_condition_family_bridge_enable: bool = False,
        phrase_condition_family_bridge_scale: float = 0.20,
        phrase_condition_candidate_bias_enable: bool = False,
        phrase_condition_candidate_bias_scale: float = 0.10,
        phrase_condition_candidate_topk: int = 12,
        phrase_condition_candidate_slot_types: str = "",
        phrase_condition_query_bridge_enable: bool = False,
        phrase_condition_query_bridge_num_queries: int = 4,
        phrase_condition_query_bridge_scale: float = 0.15,
        phrase_condition_train_use_predicted: bool = False,
        phrase_condition_pred_detach: bool = True,
        phrase_condition_teacher_source: str = "single_ref",
        phrase_gen_max_len: int = 48,
        phrase_memory_mode: str = "pooled",
        phrase_target_mode: str = "flat",
        phrase_slot_schema: str = "raw",
        max_phrase_slots: int = 4,
        phrase_slot_max_len: int = 24,
        phrase_slot_planner_enable: bool = False,
        phrase_slot_planner_flow_enable: bool = False,
        phrase_slot_planner_flow_scale: float = 0.20,
        phrase_slot_planner_flow_slot_types: str = "",
        phrase_slot_guidance_enable: bool = False,
        phrase_slot_role_anchor_enable: bool = False,
        phrase_slot_role_anchor_topk: int = 4,
        phrase_slot_role_anchor_scale: float = 1.0,
        phrase_slot_role_anchor_slot_types: str = "",
        phrase_slot_decode_anchor_enable: bool = False,
        phrase_slot_decode_anchor_topk: int = 8,
        phrase_slot_decode_anchor_scale: float = 1.0,
        phrase_slot_decode_anchor_early_scale: float = 1.25,
        phrase_slot_decode_anchor_family_scale: float = 0.75,
        phrase_slot_decode_anchor_stopword_penalty: float = 0.75,
        phrase_slot_decode_anchor_stopword_steps: int = 2,
        phrase_slot_decode_anchor_debug_topk: int = 8,
        phrase_slot_presence_enable: bool = False,
        phrase_slot_presence_support_enable: bool = False,
        phrase_slot_presence_evidence_enable: bool = False,
        phrase_slot_presence_context_slot_types: str = "",
        phrase_slot_presence_threshold: float = 0.5,
        phrase_slot_presence_thresholds: Optional[Sequence[float]] = None,
        phrase_slot_active_slot_types: str = "",
        prior_head_type: str = "simple",
        prior_head_num_heads: int = 8,
        prior_head_hidden_dim: int = 2048,
        prior_head_num_blocks: int = 4,
        prior_head_num_clusters: int = 16,
        prior_head_expansion: int = 2,
        prior_head_groups: int = 8,
        aux_visual_enable: bool = False,
        aux_raw_global_enable: bool = False,
        aux_patch_enable: bool = False,
        aux_visual_raw_global_dim: int = 512,
        aux_visual_patch_dim: int = 768,
        aux_visual_prior_scale: float = 0.15,
        aux_visual_struct_scale: float = 0.10,
        aux_visual_memory_scale: float = 0.10,
        phrase_progress_enable: bool = False,
        phrase_progress_memory_scale: float = 0.10,
        phrase_progress_source_scale: float = 0.10,
        entity_label_token_ids: Optional[Sequence[Sequence[int]]] = None,
        action_label_token_ids: Optional[Sequence[Sequence[int]]] = None,
        attribute_label_token_ids: Optional[Sequence[Sequence[int]]] = None,
        scene_label_token_ids: Optional[Sequence[Sequence[int]]] = None,
        slot_family_anchor_token_ids: Optional[Dict[str, Sequence[int]]] = None,
        slot_family_anchor_token_weights: Optional[Dict[str, Sequence[float]]] = None,
        decode_stopword_token_ids: Optional[Sequence[int]] = None,
        **caption_model_kwargs,
    ):
        super().__init__()
        self.caption_model = CaptionModel_Base(**caption_model_kwargs)

        d_model = int(caption_model_kwargs.get("d_model", 512))
        decoder_nhead = int(caption_model_kwargs.get("decoder_nhead", 8))

        self.pad_token_id = int(caption_model_kwargs.get("pad_token_id", 0))
        self.bos_token_id = int(caption_model_kwargs.get("bos_token_id", 49406))
        self.eos_token_id = int(caption_model_kwargs.get("eos_token_id", 49407))

        self.entity_dim = int(entity_dim)
        self.action_dim = int(action_dim)
        self.attribute_dim = int(attribute_dim)
        self.scene_dim = int(scene_dim)
        self.struct_condition = bool(struct_condition)
        self.struct_condition_query_bridge_enable = bool(
            struct_condition_query_bridge_enable and self.struct_condition
        )
        self.struct_condition_query_bridge_num_queries = max(1, int(struct_condition_query_bridge_num_queries))
        self.struct_condition_query_bridge_scale_init = float(struct_condition_query_bridge_scale)
        self.struct_condition_query_bridge_memory_enable = bool(
            struct_condition_query_bridge_memory_enable
            and self.struct_condition_query_bridge_enable
            and phrase_decoder_enable
        )
        self.struct_condition_query_bridge_memory_scale_init = float(struct_condition_query_bridge_memory_scale)
        self.struct_condition_query_bridge_hidden_enable = bool(
            struct_condition_query_bridge_hidden_enable
            and self.struct_condition_query_bridge_enable
            and self.struct_condition
        )
        self.struct_condition_query_bridge_hidden_scale_init = float(struct_condition_query_bridge_hidden_scale)
        self.phrase_decoder_enable = bool(phrase_decoder_enable)
        self.phrase_condition_enable = bool(phrase_condition_enable and phrase_decoder_enable)
        self.phrase_condition_train_use_predicted = bool(
            phrase_condition_train_use_predicted and self.phrase_condition_enable
        )
        self.phrase_condition_pred_detach = bool(
            phrase_condition_pred_detach and self.phrase_condition_train_use_predicted
        )
        teacher_source = str(phrase_condition_teacher_source).strip().lower() or "single_ref"
        if teacher_source not in {"single_ref", "ref_bank"}:
            raise ValueError(
                "phrase_condition_teacher_source must be one of {'single_ref', 'ref_bank'}, "
                f"got {phrase_condition_teacher_source}"
            )
        self.phrase_condition_teacher_source = teacher_source
        self.phrase_gen_max_len = max(1, int(phrase_gen_max_len))
        self.phrase_memory_mode = str(phrase_memory_mode).strip().lower() or "pooled"
        if self.phrase_memory_mode not in {"pooled", "temporal"}:
            raise ValueError(f"Unsupported phrase_memory_mode: {phrase_memory_mode}")
        self.phrase_target_mode = str(phrase_target_mode).strip().lower() or "flat"
        if self.phrase_target_mode not in {"flat", "slot"}:
            raise ValueError(f"Unsupported phrase_target_mode: {phrase_target_mode}")
        self.phrase_condition_slot_aware_enable = bool(
            phrase_condition_slot_aware_enable
            and self.phrase_condition_enable
            and self.phrase_target_mode == "slot"
        )
        self.phrase_condition_slot_selective_enable = bool(
            phrase_condition_slot_selective_enable
            and self.phrase_condition_enable
            and self.phrase_target_mode == "slot"
        )
        self.phrase_condition_slot_residual_enable = bool(
            phrase_condition_slot_residual_enable
            and self.phrase_condition_enable
            and self.phrase_target_mode == "slot"
        )
        core_slot_types = self._parse_slot_type_list(phrase_condition_core_slot_types)
        aux_slot_types = self._parse_slot_type_list(phrase_condition_aux_slot_types)
        slot_residual_slot_types = self._parse_slot_type_list(phrase_condition_slot_residual_slot_types)
        candidate_slot_types = self._parse_slot_type_list(phrase_condition_candidate_slot_types)
        if self.phrase_condition_slot_selective_enable and not core_slot_types:
            core_slot_types = self._DEFAULT_PHRASE_CONDITION_CORE_SLOT_TYPES
        core_slot_type_set = set(core_slot_types)
        if self.phrase_condition_slot_selective_enable and not aux_slot_types:
            aux_slot_types = tuple(
                slot_type
                for slot_type in self._DEFAULT_PHRASE_CONDITION_AUX_SLOT_TYPES
                if slot_type not in core_slot_type_set
            )
        else:
            aux_slot_types = tuple(slot_type for slot_type in aux_slot_types if slot_type not in core_slot_type_set)
        self.phrase_condition_core_slot_types = tuple(core_slot_types)
        self.phrase_condition_aux_slot_types = tuple(aux_slot_types)
        self.phrase_condition_aux_scale_init = float(phrase_condition_aux_scale)
        self.phrase_condition_slot_residual_slot_types = tuple(slot_residual_slot_types)
        self.phrase_condition_slot_residual_scale_init = float(phrase_condition_slot_residual_scale)
        self.phrase_condition_family_bridge_enable = bool(
            phrase_condition_family_bridge_enable and self.phrase_condition_enable and self.phrase_target_mode == "slot"
        )
        self.phrase_condition_family_bridge_scale_init = float(phrase_condition_family_bridge_scale)
        self.phrase_condition_candidate_bias_enable = bool(
            phrase_condition_candidate_bias_enable and self.phrase_condition_enable
        )
        self.phrase_condition_candidate_bias_scale_init = float(phrase_condition_candidate_bias_scale)
        self.phrase_condition_candidate_topk = max(1, int(phrase_condition_candidate_topk))
        self.phrase_condition_candidate_slot_types = tuple(candidate_slot_types)
        self.phrase_condition_query_bridge_enable = bool(
            phrase_condition_query_bridge_enable and self.phrase_condition_enable
        )
        self.phrase_condition_query_bridge_num_queries = max(1, int(phrase_condition_query_bridge_num_queries))
        self.phrase_condition_query_bridge_scale_init = float(phrase_condition_query_bridge_scale)
        self.phrase_condition_family_groups = tuple(
            (str(family_name), tuple(str(slot_type) for slot_type in slot_types))
            for family_name, slot_types in self._DEFAULT_PHRASE_CONDITION_FAMILY_GROUPS
        )
        self.phrase_condition_family_count = len(self.phrase_condition_family_groups)
        self.phrase_slot_schema = str(phrase_slot_schema).strip().lower() or "raw"
        if self.phrase_slot_schema not in {
            "raw",
            "typed",
            "typed_rich",
            "typed_rich_semantic",
            "typed_rich_roleaware",
            "family4_compact",
        }:
            raise ValueError(f"Unsupported phrase_slot_schema: {phrase_slot_schema}")
        self.max_phrase_slots = max(1, int(max_phrase_slots))
        self.phrase_slot_max_len = max(4, int(phrase_slot_max_len))
        self.phrase_slot_planner_enable = bool(
            phrase_slot_planner_enable and self.phrase_decoder_enable and self.phrase_target_mode == "slot"
        )
        self.phrase_slot_planner_flow_enable = bool(
            phrase_slot_planner_flow_enable and self.phrase_slot_planner_enable and self.max_phrase_slots > 1
        )
        self.phrase_slot_planner_flow_scale = max(0.0, float(phrase_slot_planner_flow_scale))
        self.phrase_slot_planner_flow_slot_types = self._parse_slot_type_list(
            phrase_slot_planner_flow_slot_types
        )
        self.phrase_slot_guidance_enable = bool(
            phrase_slot_guidance_enable and self.phrase_decoder_enable and self.phrase_target_mode == "slot"
        )
        self.phrase_slot_role_anchor_enable = bool(
            phrase_slot_role_anchor_enable and self.phrase_decoder_enable and self.phrase_target_mode == "slot"
        )
        self.phrase_slot_role_anchor_topk = max(1, int(phrase_slot_role_anchor_topk))
        self.phrase_slot_role_anchor_scale = float(phrase_slot_role_anchor_scale)
        self.phrase_slot_role_anchor_slot_types = self._parse_slot_type_list(
            phrase_slot_role_anchor_slot_types
        )
        self.phrase_slot_decode_anchor_enable = bool(
            phrase_slot_decode_anchor_enable and self.phrase_decoder_enable and self.phrase_target_mode == "slot"
        )
        self.phrase_slot_decode_anchor_topk = max(1, int(phrase_slot_decode_anchor_topk))
        self.phrase_slot_decode_anchor_scale = max(0.0, float(phrase_slot_decode_anchor_scale))
        self.phrase_slot_decode_anchor_early_scale = max(0.0, float(phrase_slot_decode_anchor_early_scale))
        self.phrase_slot_decode_anchor_family_scale = max(0.0, float(phrase_slot_decode_anchor_family_scale))
        self.phrase_slot_decode_anchor_stopword_penalty = max(0.0, float(phrase_slot_decode_anchor_stopword_penalty))
        self.phrase_slot_decode_anchor_stopword_steps = max(0, int(phrase_slot_decode_anchor_stopword_steps))
        self.phrase_slot_decode_anchor_debug_topk = max(1, int(phrase_slot_decode_anchor_debug_topk))
        self.phrase_slot_presence_enable = bool(
            phrase_slot_presence_enable and self.phrase_decoder_enable and self.phrase_target_mode == "slot"
        )
        self.phrase_slot_presence_support_enable = bool(
            phrase_slot_presence_support_enable
            and self.phrase_slot_presence_enable
            and self.phrase_slot_planner_enable
        )
        self.phrase_slot_presence_evidence_enable = bool(
            phrase_slot_presence_evidence_enable and self.phrase_slot_presence_enable
        )
        self.phrase_slot_presence_context_slot_types = self._parse_slot_type_list(
            phrase_slot_presence_context_slot_types
        )
        self.phrase_slot_active_slot_types = self._parse_slot_type_list(phrase_slot_active_slot_types)
        self.phrase_slot_presence_threshold = float(phrase_slot_presence_threshold)
        if not 0.0 <= self.phrase_slot_presence_threshold <= 1.0:
            raise ValueError(
                f"phrase_slot_presence_threshold must be within [0, 1], got {phrase_slot_presence_threshold}"
            )
        threshold_tensor = None
        if phrase_slot_presence_thresholds is not None:
            threshold_tensor = torch.as_tensor(phrase_slot_presence_thresholds, dtype=torch.float32).flatten()
            if threshold_tensor.numel() == 0:
                threshold_tensor = None
            elif bool(((threshold_tensor < 0.0) | (threshold_tensor > 1.0)).any()):
                raise ValueError(
                    "phrase_slot_presence_thresholds must contain values within [0, 1], "
                    f"got {phrase_slot_presence_thresholds}"
                )
        self.register_buffer("_phrase_slot_presence_thresholds_tensor", threshold_tensor, persistent=False)
        self.prior_head_type = str(prior_head_type).strip().lower() or "simple"
        if self.prior_head_type not in {"simple", "multi_semantic", "attn_nextvlad"}:
            raise ValueError(f"Unsupported prior_head_type: {prior_head_type}")
        self.prior_head_num_heads = max(1, int(prior_head_num_heads))
        self.prior_head_hidden_dim = max(d_model, int(prior_head_hidden_dim))
        self.prior_head_num_blocks = max(1, int(prior_head_num_blocks))
        self.prior_head_num_clusters = max(4, int(prior_head_num_clusters))
        self.prior_head_expansion = max(1, int(prior_head_expansion))
        self.prior_head_groups = max(1, int(prior_head_groups))

        self.aux_raw_global_enable = bool(aux_raw_global_enable)
        self.aux_patch_enable = bool(aux_patch_enable)
        self.aux_visual_enable = bool(aux_visual_enable and (self.aux_raw_global_enable or self.aux_patch_enable))
        self.aux_visual_raw_global_dim = int(aux_visual_raw_global_dim)
        self.aux_visual_patch_dim = int(aux_visual_patch_dim)
        self.aux_visual_prior_scale = max(0.0, float(aux_visual_prior_scale))
        self.aux_visual_struct_scale = max(0.0, float(aux_visual_struct_scale))
        self.aux_visual_memory_scale = max(0.0, float(aux_visual_memory_scale))
        self.phrase_progress_enable = bool(phrase_progress_enable and self.phrase_decoder_enable)
        self.phrase_progress_memory_scale = max(0.0, float(phrase_progress_memory_scale))
        self.phrase_progress_source_scale = max(0.0, float(phrase_progress_source_scale))
        self.aux_raw_global_proj = None
        self.aux_patch_proj = None
        self.aux_visual_token_norm = None
        self.aux_visual_prior_proj = None
        self.aux_visual_struct_proj = None
        self.aux_visual_memory_proj = None
        self.phrase_progress_proj = None
        if self.aux_visual_enable:
            if self.aux_raw_global_enable:
                self.aux_raw_global_proj = nn.Sequential(
                    nn.LayerNorm(self.aux_visual_raw_global_dim),
                    nn.Linear(self.aux_visual_raw_global_dim, d_model),
                    nn.Tanh(),
                )
            if self.aux_patch_enable:
                self.aux_patch_proj = nn.Sequential(
                    nn.LayerNorm(self.aux_visual_patch_dim),
                    nn.Linear(self.aux_visual_patch_dim, d_model),
                    nn.Tanh(),
                )
            self.aux_visual_token_norm = nn.LayerNorm(d_model)
            self.aux_visual_prior_proj = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.Tanh(),
            )
            self.aux_visual_struct_proj = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.Tanh(),
            )
            self.aux_visual_memory_proj = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.Tanh(),
            )
        if self.phrase_progress_enable:
            self.phrase_progress_proj = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.Tanh(),
            )

        def _build_prior_head(num_classes: int) -> nn.Module:
            if self.prior_head_type == "multi_semantic":
                return StructuredMultiSemanticPriorHead(
                    d_model=d_model,
                    num_classes=num_classes,
                    num_heads=self.prior_head_num_heads,
                    hidden_dim=self.prior_head_hidden_dim,
                    num_mlp_blocks=self.prior_head_num_blocks,
                    dropout=prior_dropout,
                )
            if self.prior_head_type == "attn_nextvlad":
                return StructuredAttentionNextVLADPriorHead(
                    d_model=d_model,
                    num_classes=num_classes,
                    num_heads=self.prior_head_num_heads,
                    hidden_dim=self.prior_head_hidden_dim,
                    num_mlp_blocks=self.prior_head_num_blocks,
                    num_clusters=self.prior_head_num_clusters,
                    expansion=self.prior_head_expansion,
                    groups=self.prior_head_groups,
                    dropout=prior_dropout,
                )
            return nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Dropout(prior_dropout),
                nn.Linear(d_model, num_classes),
            )

        self.entity_prior_head = None
        if self.entity_dim > 0:
            self.entity_prior_head = _build_prior_head(self.entity_dim)

        self.action_prior_head = None
        if self.action_dim > 0:
            self.action_prior_head = _build_prior_head(self.action_dim)

        self.attribute_prior_head = None
        if self.attribute_dim > 0:
            self.attribute_prior_head = _build_prior_head(self.attribute_dim)

        self.scene_prior_head = None
        if self.scene_dim > 0:
            self.scene_prior_head = _build_prior_head(self.scene_dim)

        # Planner-to-decoder bridge: convert structured priors to a decoder context vector.
        self.entity_to_context = nn.Linear(self.entity_dim, d_model) if self.entity_dim > 0 else None
        self.action_to_context = nn.Linear(self.action_dim, d_model) if self.action_dim > 0 else None
        self.attribute_to_context = nn.Linear(self.attribute_dim, d_model) if self.attribute_dim > 0 else None
        self.scene_to_context = nn.Linear(self.scene_dim, d_model) if self.scene_dim > 0 else None
        self.attribute_context_scale = (
            nn.Parameter(torch.tensor(0.0, dtype=torch.float32)) if self.attribute_dim > 0 else None
        )
        self.scene_context_scale = (
            nn.Parameter(torch.tensor(0.0, dtype=torch.float32)) if self.scene_dim > 0 else None
        )
        self.condition_gate = nn.Linear(d_model, d_model)
        self.condition_dropout = nn.Dropout(prior_dropout)
        self.condition_norm = nn.LayerNorm(d_model)
        self.condition_post_norm = nn.LayerNorm(d_model)
        self.struct_scale = nn.Parameter(torch.tensor(float(struct_condition_scale), dtype=torch.float32))
        self.struct_condition_query_bridge_queries = None
        self.struct_condition_query_bridge_source_embeddings = None
        self.struct_condition_query_bridge_source_key = None
        self.struct_condition_query_bridge_source_value = None
        self.struct_condition_query_bridge_token_norm = None
        self.struct_condition_query_bridge_scale = None
        self.struct_condition_query_bridge_memory_scale = None
        self.struct_condition_query_bridge_hidden_query = None
        self.struct_condition_query_bridge_hidden_key = None
        self.struct_condition_query_bridge_hidden_value = None
        self.struct_condition_query_bridge_hidden_gate = None
        self.struct_condition_query_bridge_hidden_norm = None
        self.struct_condition_query_bridge_hidden_scale = None
        if self.struct_condition_query_bridge_enable:
            self.struct_condition_query_bridge_queries = nn.Parameter(
                torch.randn(self.struct_condition_query_bridge_num_queries, d_model, dtype=torch.float32) * 0.02
            )
            self.struct_condition_query_bridge_source_embeddings = nn.Embedding(5, d_model)
            self.struct_condition_query_bridge_source_key = nn.Linear(d_model, d_model)
            self.struct_condition_query_bridge_source_value = nn.Linear(d_model, d_model)
            self.struct_condition_query_bridge_token_norm = nn.LayerNorm(d_model)
            self.struct_condition_query_bridge_scale = nn.Parameter(
                torch.tensor(self.struct_condition_query_bridge_scale_init, dtype=torch.float32)
            )
            if self.struct_condition_query_bridge_memory_enable:
                self.struct_condition_query_bridge_memory_scale = nn.Parameter(
                    torch.tensor(self.struct_condition_query_bridge_memory_scale_init, dtype=torch.float32)
                )
            if self.struct_condition_query_bridge_hidden_enable:
                self.struct_condition_query_bridge_hidden_query = nn.Linear(d_model, d_model)
                self.struct_condition_query_bridge_hidden_key = nn.Linear(d_model, d_model)
                self.struct_condition_query_bridge_hidden_value = nn.Linear(d_model, d_model)
                self.struct_condition_query_bridge_hidden_gate = nn.Linear(d_model * 2, d_model)
                self.struct_condition_query_bridge_hidden_norm = nn.LayerNorm(d_model)
                self.struct_condition_query_bridge_hidden_scale = nn.Parameter(
                    torch.tensor(self.struct_condition_query_bridge_hidden_scale_init, dtype=torch.float32)
                )

        self.phrase_decoder = None
        self.phrase_output_norm = None
        self.phrase_to_context = None
        self.phrase_condition_gate = None
        self.phrase_condition_norm = None
        self.phrase_slot_condition_proj = None
        self.phrase_slot_condition_presence_proj = None
        self.phrase_slot_condition_norm = None
        self.phrase_condition_core_proj = None
        self.phrase_condition_aux_query = None
        self.phrase_condition_aux_key = None
        self.phrase_condition_aux_value = None
        self.phrase_condition_aux_gate = None
        self.phrase_condition_aux_norm = None
        self.phrase_condition_aux_scale = None
        self.phrase_condition_slot_residual_query = None
        self.phrase_condition_slot_residual_key = None
        self.phrase_condition_slot_residual_value = None
        self.phrase_condition_slot_residual_gate = None
        self.phrase_condition_slot_residual_norm = None
        self.phrase_condition_slot_residual_scale = None
        self.phrase_condition_slot_residual_order_embeddings = None
        self.phrase_condition_slot_residual_type_embeddings = None
        self.phrase_condition_family_embeddings = None
        self.phrase_condition_family_query = None
        self.phrase_condition_family_key = None
        self.phrase_condition_family_value = None
        self.phrase_condition_family_gate = None
        self.phrase_condition_family_norm = None
        self.phrase_condition_family_scale = None
        self.phrase_condition_candidate_query = None
        self.phrase_condition_candidate_key = None
        self.phrase_condition_candidate_value = None
        self.phrase_condition_candidate_gate = None
        self.phrase_condition_candidate_scale = None
        self.phrase_condition_query_bridge_queries = None
        self.phrase_condition_query_bridge_summary_key = None
        self.phrase_condition_query_bridge_summary_value = None
        self.phrase_condition_query_bridge_summary_norm = None
        self.phrase_condition_query_bridge_hidden_query = None
        self.phrase_condition_query_bridge_hidden_key = None
        self.phrase_condition_query_bridge_hidden_value = None
        self.phrase_condition_query_bridge_gate = None
        self.phrase_condition_query_bridge_norm = None
        self.phrase_condition_query_bridge_scale = None
        self.phrase_scale = None
        self.slot_embeddings = None
        self.entity_label_embeddings = None
        self.action_label_embeddings = None
        self.attribute_label_embeddings = None
        self.scene_label_embeddings = None
        self.slot_role_embeddings = None
        self.slot_planner_query = None
        self.slot_planner_key = None
        self.slot_planner_value = None
        self.slot_planner_norm = None
        self.slot_planner_flow_query = None
        self.slot_planner_flow_key = None
        self.slot_planner_flow_value = None
        self.slot_planner_flow_gate = None
        self.slot_planner_flow_update = None
        self.slot_planner_flow_norm = None
        self.slot_presence_head = None
        self.slot_presence_support_proj = None
        self.slot_presence_evidence_proj = None
        self.slot_guidance_target_proj = None
        self.slot_guidance_support_proj = None
        self.slot_guidance_evidence_proj = None
        self.slot_guidance_norm = None
        self.slot_role_anchor_target_proj = None
        self.slot_role_anchor_role_proj = None
        self.slot_role_anchor_norm = None
        if self.phrase_decoder_enable:
            phrase_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=decoder_nhead,
                dim_feedforward=2048,
                dropout=prior_dropout,
                batch_first=True,
            )
            self.phrase_decoder = nn.TransformerDecoder(
                phrase_layer,
                num_layers=max(1, int(phrase_decoder_layers)),
            )
            self.phrase_output_norm = nn.LayerNorm(d_model)
            self.phrase_to_context = nn.Linear(d_model, d_model)
            self.phrase_condition_gate = nn.Linear(d_model, d_model)
            self.phrase_condition_norm = nn.LayerNorm(d_model)
            if self.phrase_condition_slot_aware_enable:
                self.phrase_slot_condition_proj = nn.Linear(self.max_phrase_slots * d_model, d_model)
                self.phrase_slot_condition_presence_proj = nn.Linear(self.max_phrase_slots, d_model)
                self.phrase_slot_condition_norm = nn.LayerNorm(d_model)
            if self.phrase_condition_slot_selective_enable:
                self.phrase_condition_core_proj = nn.Linear(d_model, d_model)
                self.phrase_condition_aux_query = nn.Linear(d_model, d_model)
                self.phrase_condition_aux_key = nn.Linear(d_model, d_model)
                self.phrase_condition_aux_value = nn.Linear(d_model, d_model)
                self.phrase_condition_aux_gate = nn.Linear(d_model * 2, d_model)
                self.phrase_condition_aux_norm = nn.LayerNorm(d_model)
                self.phrase_condition_aux_scale = nn.Parameter(
                    torch.tensor(self.phrase_condition_aux_scale_init, dtype=torch.float32)
                )
            if self.phrase_condition_slot_residual_enable:
                residual_slot_specs = StructuredCaptionDataset.get_phrase_slot_type_specs(
                    max_phrase_slots=self.max_phrase_slots,
                    phrase_slot_schema=self.phrase_slot_schema,
                )
                max_family_id = 0
                for spec in residual_slot_specs:
                    max_family_id = max(max_family_id, int(spec.get("slot_type_family_id", 0)))
                self.phrase_condition_slot_residual_query = nn.Linear(d_model, d_model)
                self.phrase_condition_slot_residual_key = nn.Linear(d_model, d_model)
                self.phrase_condition_slot_residual_value = nn.Linear(d_model, d_model)
                self.phrase_condition_slot_residual_gate = nn.Linear(d_model * 2, d_model)
                self.phrase_condition_slot_residual_norm = nn.LayerNorm(d_model)
                self.phrase_condition_slot_residual_scale = nn.Parameter(
                    torch.tensor(self.phrase_condition_slot_residual_scale_init, dtype=torch.float32)
                )
                self.phrase_condition_slot_residual_order_embeddings = nn.Embedding(self.max_phrase_slots, d_model)
                self.phrase_condition_slot_residual_type_embeddings = nn.Embedding(max_family_id + 1, d_model)
            if self.phrase_condition_family_bridge_enable and self.phrase_condition_family_count > 0:
                self.phrase_condition_family_embeddings = nn.Embedding(self.phrase_condition_family_count, d_model)
                self.phrase_condition_family_query = nn.Linear(d_model, d_model)
                self.phrase_condition_family_key = nn.Linear(d_model, d_model)
                self.phrase_condition_family_value = nn.Linear(d_model, d_model)
                self.phrase_condition_family_gate = nn.Linear(d_model * 2, d_model)
                self.phrase_condition_family_norm = nn.LayerNorm(d_model)
                self.phrase_condition_family_scale = nn.Parameter(
                    torch.tensor(self.phrase_condition_family_bridge_scale_init, dtype=torch.float32)
                )
            if self.phrase_condition_candidate_bias_enable:
                self.phrase_condition_candidate_query = nn.Linear(d_model, d_model)
                self.phrase_condition_candidate_key = nn.Linear(d_model, d_model)
                self.phrase_condition_candidate_value = nn.Linear(d_model, d_model)
                self.phrase_condition_candidate_gate = nn.Linear(d_model * 2, 1)
                self.phrase_condition_candidate_scale = nn.Parameter(
                    torch.tensor(self.phrase_condition_candidate_bias_scale_init, dtype=torch.float32)
                )
            if self.phrase_condition_query_bridge_enable:
                self.phrase_condition_query_bridge_queries = nn.Parameter(
                    torch.randn(self.phrase_condition_query_bridge_num_queries, d_model, dtype=torch.float32) * 0.02
                )
                self.phrase_condition_query_bridge_summary_key = nn.Linear(d_model, d_model)
                self.phrase_condition_query_bridge_summary_value = nn.Linear(d_model, d_model)
                self.phrase_condition_query_bridge_summary_norm = nn.LayerNorm(d_model)
                self.phrase_condition_query_bridge_hidden_query = nn.Linear(d_model, d_model)
                self.phrase_condition_query_bridge_hidden_key = nn.Linear(d_model, d_model)
                self.phrase_condition_query_bridge_hidden_value = nn.Linear(d_model, d_model)
                self.phrase_condition_query_bridge_gate = nn.Linear(d_model * 2, d_model)
                self.phrase_condition_query_bridge_norm = nn.LayerNorm(d_model)
                self.phrase_condition_query_bridge_scale = nn.Parameter(
                    torch.tensor(self.phrase_condition_query_bridge_scale_init, dtype=torch.float32)
                )
            self.phrase_scale = nn.Parameter(torch.tensor(float(phrase_condition_scale), dtype=torch.float32))
            if self.phrase_target_mode == "slot":
                self.slot_embeddings = nn.Embedding(self.max_phrase_slots, d_model)
                if (
                    self.phrase_slot_planner_enable
                    or self.phrase_slot_presence_enable
                    or self.phrase_slot_role_anchor_enable
                ):
                    self.slot_role_embeddings = nn.Embedding(self.max_phrase_slots, d_model)
                if self.phrase_slot_planner_enable or self.phrase_slot_role_anchor_enable:
                    self.entity_label_embeddings = nn.Embedding(self.entity_dim, d_model) if self.entity_dim > 0 else None
                    self.action_label_embeddings = nn.Embedding(self.action_dim, d_model) if self.action_dim > 0 else None
                    self.attribute_label_embeddings = (
                        nn.Embedding(self.attribute_dim, d_model) if self.attribute_dim > 0 else None
                    )
                    self.scene_label_embeddings = nn.Embedding(self.scene_dim, d_model) if self.scene_dim > 0 else None
                    self.slot_planner_query = nn.Linear(d_model, d_model)
                    self.slot_planner_key = nn.Linear(d_model, d_model)
                    self.slot_planner_value = nn.Linear(d_model, d_model)
                    self.slot_planner_norm = nn.LayerNorm(d_model)
                    if self.phrase_slot_planner_flow_enable:
                        self.slot_planner_flow_query = nn.Linear(d_model, d_model)
                        self.slot_planner_flow_key = nn.Linear(d_model, d_model)
                        self.slot_planner_flow_value = nn.Linear(d_model, d_model)
                        self.slot_planner_flow_gate = nn.Linear(d_model * 2, d_model)
                        self.slot_planner_flow_update = nn.Sequential(
                            nn.Linear(d_model, d_model),
                            nn.Tanh(),
                        )
                        self.slot_planner_flow_norm = nn.LayerNorm(d_model)
                if self.phrase_slot_guidance_enable:
                    self.slot_guidance_target_proj = nn.Sequential(
                        nn.Linear(d_model, d_model),
                        nn.Tanh(),
                    )
                    self.slot_guidance_support_proj = nn.Sequential(
                        nn.Linear(3, d_model),
                        nn.Tanh(),
                    )
                    self.slot_guidance_evidence_proj = nn.Sequential(
                        nn.Linear(12, d_model),
                        nn.Tanh(),
                    )
                    self.slot_guidance_norm = nn.LayerNorm(d_model)
                if self.phrase_slot_role_anchor_enable:
                    self.slot_role_anchor_target_proj = nn.Sequential(
                        nn.Linear(d_model, d_model),
                        nn.Tanh(),
                    )
                    self.slot_role_anchor_role_proj = nn.Sequential(
                        nn.Linear(d_model, d_model),
                        nn.Tanh(),
                    )
                    self.slot_role_anchor_norm = nn.LayerNorm(d_model)
                if self.phrase_slot_presence_enable:
                    self.slot_presence_head = nn.Sequential(
                        nn.LayerNorm(d_model),
                        nn.Dropout(prior_dropout),
                        nn.Linear(d_model, 1),
                    )
                    if self.phrase_slot_presence_support_enable:
                        self.slot_presence_support_proj = nn.Sequential(
                            nn.Linear(3, d_model),
                            nn.Tanh(),
                        )
                    if self.phrase_slot_presence_evidence_enable:
                        self.slot_presence_evidence_proj = nn.Sequential(
                            nn.Linear(12, d_model),
                            nn.Tanh(),
                        )

        entity_token_ids_tensor, entity_token_mask_tensor = self._build_label_token_bank_tensors(
            entity_label_token_ids
        )
        action_token_ids_tensor, action_token_mask_tensor = self._build_label_token_bank_tensors(
            action_label_token_ids
        )
        attribute_token_ids_tensor, attribute_token_mask_tensor = self._build_label_token_bank_tensors(
            attribute_label_token_ids
        )
        scene_token_ids_tensor, scene_token_mask_tensor = self._build_label_token_bank_tensors(
            scene_label_token_ids
        )
        self.register_buffer("_entity_label_token_ids", entity_token_ids_tensor, persistent=False)
        self.register_buffer("_entity_label_token_mask", entity_token_mask_tensor, persistent=False)
        self.register_buffer("_action_label_token_ids", action_token_ids_tensor, persistent=False)
        self.register_buffer("_action_label_token_mask", action_token_mask_tensor, persistent=False)
        self.register_buffer("_attribute_label_token_ids", attribute_token_ids_tensor, persistent=False)
        self.register_buffer("_attribute_label_token_mask", attribute_token_mask_tensor, persistent=False)
        self.register_buffer("_scene_label_token_ids", scene_token_ids_tensor, persistent=False)
        self.register_buffer("_scene_label_token_mask", scene_token_mask_tensor, persistent=False)

        family_anchor_token_ids, family_anchor_token_mask, family_anchor_token_weights = (
            self._build_slot_family_anchor_tensors(
                slot_family_anchor_token_ids=slot_family_anchor_token_ids,
                slot_family_anchor_token_weights=slot_family_anchor_token_weights,
            )
        )
        self.register_buffer("_slot_family_anchor_token_ids", family_anchor_token_ids, persistent=False)
        self.register_buffer("_slot_family_anchor_token_mask", family_anchor_token_mask, persistent=False)
        self.register_buffer("_slot_family_anchor_token_weights", family_anchor_token_weights, persistent=False)
        decode_stopword_tensor = self._build_token_id_tensor(decode_stopword_token_ids)
        self.register_buffer("_decode_stopword_token_ids", decode_stopword_tensor, persistent=False)

    @staticmethod
    def _normalize_token_id_sequence(
        token_ids: Optional[Sequence[int]],
        *,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
    ) -> List[int]:
        normalized: List[int] = []
        seen = set()
        special_ids = {int(pad_token_id), int(bos_token_id), int(eos_token_id)}
        for raw_token_id in token_ids or tuple():
            token_id = int(raw_token_id)
            if token_id < 0 or token_id in special_ids or token_id in seen:
                continue
            normalized.append(token_id)
            seen.add(token_id)
        return normalized

    @classmethod
    def _build_label_token_bank_tensors(
        cls,
        token_lists: Optional[Sequence[Sequence[int]]],
        *,
        pad_fill_value: int = 0,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        normalized_rows: List[List[int]] = []
        max_len = 0
        for token_ids in token_lists or tuple():
            normalized = [int(token_id) for token_id in token_ids if int(token_id) >= 0]
            normalized_rows.append(normalized)
            max_len = max(max_len, len(normalized))
        if not normalized_rows or max_len <= 0:
            return None, None

        token_id_tensor = torch.full(
            (len(normalized_rows), max_len),
            int(pad_fill_value),
            dtype=torch.long,
        )
        token_mask_tensor = torch.zeros((len(normalized_rows), max_len), dtype=torch.bool)
        for row_idx, token_ids in enumerate(normalized_rows):
            if not token_ids:
                continue
            token_id_tensor[row_idx, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
            token_mask_tensor[row_idx, : len(token_ids)] = True
        return token_id_tensor, token_mask_tensor

    def _build_slot_family_anchor_tensors(
        self,
        *,
        slot_family_anchor_token_ids: Optional[Dict[str, Sequence[int]]],
        slot_family_anchor_token_weights: Optional[Dict[str, Sequence[float]]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        slot_specs = StructuredCaptionDataset.get_phrase_slot_type_specs(
            max_phrase_slots=self.max_phrase_slots,
            phrase_slot_schema=self.phrase_slot_schema,
        )
        normalized_by_slot: List[List[int]] = []
        weights_by_slot: List[List[float]] = []
        max_len = 0
        for slot_spec in slot_specs[: self.max_phrase_slots]:
            slot_type = str(slot_spec.get("slot_type_family", slot_spec.get("slot_type", ""))).strip().lower()
            normalized = self._normalize_token_id_sequence(
                (slot_family_anchor_token_ids or {}).get(slot_type),
                pad_token_id=self.pad_token_id,
                bos_token_id=self.bos_token_id,
                eos_token_id=self.eos_token_id,
            )
            raw_weights = list((slot_family_anchor_token_weights or {}).get(slot_type, []))
            if len(raw_weights) < len(normalized):
                raw_weights.extend([1.0] * (len(normalized) - len(raw_weights)))
            raw_weights = [max(0.0, float(weight)) for weight in raw_weights[: len(normalized)]]
            weight_sum = float(sum(raw_weights))
            if weight_sum > 0.0:
                raw_weights = [float(weight) / weight_sum for weight in raw_weights]
            normalized_by_slot.append(normalized)
            weights_by_slot.append(raw_weights)
            max_len = max(max_len, len(normalized))
        if max_len <= 0:
            return None, None, None

        token_id_tensor = torch.zeros((self.max_phrase_slots, max_len), dtype=torch.long)
        token_mask_tensor = torch.zeros((self.max_phrase_slots, max_len), dtype=torch.bool)
        token_weight_tensor = torch.zeros((self.max_phrase_slots, max_len), dtype=torch.float32)
        for slot_idx, token_ids in enumerate(normalized_by_slot):
            if not token_ids:
                continue
            token_weights = weights_by_slot[slot_idx] or [1.0 / float(len(token_ids))] * len(token_ids)
            token_id_tensor[slot_idx, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
            token_mask_tensor[slot_idx, : len(token_ids)] = True
            token_weight_tensor[slot_idx, : len(token_ids)] = torch.tensor(token_weights, dtype=torch.float32)
        return token_id_tensor, token_mask_tensor, token_weight_tensor

    @staticmethod
    def _build_token_id_tensor(token_ids: Optional[Sequence[int]]) -> Optional[torch.Tensor]:
        normalized = []
        seen = set()
        for raw_token_id in token_ids or tuple():
            token_id = int(raw_token_id)
            if token_id < 0 or token_id in seen:
                continue
            normalized.append(token_id)
            seen.add(token_id)
        if not normalized:
            return None
        return torch.tensor(normalized, dtype=torch.long)

    def _get_label_token_bank(
        self,
        source_name: str,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        normalized_source_name = str(source_name).strip().lower()
        if normalized_source_name == "entity":
            return self._entity_label_token_ids, self._entity_label_token_mask
        if normalized_source_name == "action":
            return self._action_label_token_ids, self._action_label_token_mask
        if normalized_source_name == "attribute":
            return self._attribute_label_token_ids, self._attribute_label_token_mask
        if normalized_source_name == "scene":
            return self._scene_label_token_ids, self._scene_label_token_mask
        return None, None

    def _resolve_slot_decode_source_weight(
        self,
        *,
        slot_idx: int,
        slot_count: int,
        source_name: str,
        source_names: Optional[Sequence[str]],
        slot_source_weights: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        normalized_source_name = str(source_name).strip().lower()
        if (
            slot_source_weights is not None
            and slot_source_weights.dim() == 3
            and slot_source_weights.size(1) > slot_idx
            and source_names
        ):
            for source_idx, current_source_name in enumerate(source_names):
                if str(current_source_name).strip().lower() == normalized_source_name:
                    return slot_source_weights[:, slot_idx, source_idx].to(device=device, dtype=dtype)

        prior_map = self.get_slot_source_prior_map(self.phrase_slot_schema)
        if prior_map is None:
            return None
        slot_specs = StructuredCaptionDataset.get_phrase_slot_type_specs(
            max_phrase_slots=slot_count,
            phrase_slot_schema=self.phrase_slot_schema,
        )
        if slot_idx >= len(slot_specs):
            return None
        slot_type = str(slot_specs[slot_idx].get("slot_type_family", slot_specs[slot_idx].get("slot_type", "extra")))
        source_prior = prior_map.get(slot_type, prior_map.get("extra", prior_map.get("generic", {})))
        scalar = max(0.0, float(source_prior.get(normalized_source_name, 0.0)))
        if scalar <= 0.0:
            return None
        return torch.full((batch_size,), scalar, device=device, dtype=dtype)

    def _build_slot_decode_anchor_bias(
        self,
        *,
        slot_idx: int,
        slot_count: int,
        batch_size: int,
        vocab_size: int,
        source_names: Optional[Sequence[str]],
        slot_source_weights: Optional[torch.Tensor],
        entity_prior_logits: Optional[torch.Tensor],
        action_prior_logits: Optional[torch.Tensor],
        attribute_prior_logits: Optional[torch.Tensor],
        scene_prior_logits: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not self.phrase_slot_decode_anchor_enable or self.phrase_slot_decode_anchor_scale <= 0.0:
            return None

        bias = None
        for source_name, prior_logits in (
            ("entity", entity_prior_logits),
            ("action", action_prior_logits),
            ("attribute", attribute_prior_logits),
            ("scene", scene_prior_logits),
        ):
            if prior_logits is None or prior_logits.numel() == 0:
                continue
            label_token_ids, label_token_mask = self._get_label_token_bank(source_name)
            if label_token_ids is None or label_token_mask is None:
                continue
            source_weight = self._resolve_slot_decode_source_weight(
                slot_idx=slot_idx,
                slot_count=slot_count,
                source_name=source_name,
                source_names=source_names,
                slot_source_weights=slot_source_weights,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            if source_weight is None or not bool((source_weight > 0).any()):
                continue

            probs = torch.sigmoid(prior_logits.detach())
            topk_k = min(self.phrase_slot_decode_anchor_topk, int(probs.size(-1)), int(label_token_ids.size(0)))
            if topk_k <= 0:
                continue
            top_values, top_indices = probs.topk(topk_k, dim=-1)
            gathered_token_ids = label_token_ids.to(device=device)[top_indices]
            gathered_token_mask = label_token_mask.to(device=device)[top_indices]
            if not bool(gathered_token_mask.any()):
                continue

            token_den = gathered_token_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(dtype=top_values.dtype)
            token_scores = top_values.unsqueeze(-1) / token_den
            token_scores = token_scores * gathered_token_mask.to(dtype=top_values.dtype)
            token_scores = token_scores * source_weight.view(batch_size, 1, 1).to(dtype=top_values.dtype)

            if bias is None:
                bias = torch.zeros((batch_size, vocab_size), device=device, dtype=dtype)
            flat_ids = gathered_token_ids.masked_fill(~gathered_token_mask, 0).reshape(batch_size, -1)
            flat_scores = token_scores.to(dtype=dtype).reshape(batch_size, -1)
            bias.scatter_add_(1, flat_ids, flat_scores)

        if (
            self._slot_family_anchor_token_ids is not None
            and self._slot_family_anchor_token_mask is not None
            and self._slot_family_anchor_token_weights is not None
            and slot_idx < self._slot_family_anchor_token_ids.size(0)
            and self.phrase_slot_decode_anchor_family_scale > 0.0
        ):
            family_ids = self._slot_family_anchor_token_ids[slot_idx].to(device=device)
            family_mask = self._slot_family_anchor_token_mask[slot_idx].to(device=device)
            family_weights = self._slot_family_anchor_token_weights[slot_idx].to(device=device, dtype=dtype)
            if bool(family_mask.any()):
                if bias is None:
                    bias = torch.zeros((batch_size, vocab_size), device=device, dtype=dtype)
                family_ids = family_ids.masked_fill(~family_mask, 0).unsqueeze(0).expand(batch_size, -1)
                family_scores = (
                    self.phrase_slot_decode_anchor_family_scale * family_weights
                ).unsqueeze(0).expand(batch_size, -1)
                family_scores = family_scores * family_mask.unsqueeze(0).to(dtype=dtype)
                bias.scatter_add_(1, family_ids, family_scores)

        if bias is None:
            return None

        row_max = bias.max(dim=-1, keepdim=True).values.clamp_min(1e-6)
        return self.phrase_slot_decode_anchor_scale * (bias / row_max)

    def _apply_slot_decode_anchor_bias(
        self,
        *,
        logits_last: torch.Tensor,
        slot_anchor_bias: Optional[torch.Tensor],
        cur_len: int,
    ) -> torch.Tensor:
        if slot_anchor_bias is None:
            return logits_last

        bias = slot_anchor_bias.to(device=logits_last.device, dtype=logits_last.dtype)
        if cur_len <= max(1, self.phrase_slot_decode_anchor_stopword_steps):
            bias = bias * max(1.0, self.phrase_slot_decode_anchor_early_scale)
        logits_last = logits_last + bias

        if (
            self._decode_stopword_token_ids is not None
            and self.phrase_slot_decode_anchor_stopword_penalty > 0.0
            and cur_len <= max(1, self.phrase_slot_decode_anchor_stopword_steps)
        ):
            stopword_ids = self._decode_stopword_token_ids.to(device=logits_last.device)
            valid_mask = (stopword_ids >= 0) & (stopword_ids < logits_last.size(-1))
            if bool(valid_mask.any()):
                logits_last[:, stopword_ids[valid_mask]] = (
                    logits_last[:, stopword_ids[valid_mask]] - self.phrase_slot_decode_anchor_stopword_penalty
                )
        return logits_last

    def _summarize_slot_anchor_bias(
        self,
        slot_anchor_bias: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if slot_anchor_bias is None or slot_anchor_bias.numel() == 0:
            return None, None
        topk_k = min(self.phrase_slot_decode_anchor_debug_topk, int(slot_anchor_bias.size(-1)))
        if topk_k <= 0:
            return None, None
        values, indices = slot_anchor_bias.topk(topk_k, dim=-1)
        return indices, values

    @staticmethod
    def _masked_mean(video_feats: torch.Tensor, vid_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if vid_mask is None:
            return video_feats.mean(dim=1)
        mask = vid_mask.to(video_feats.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (video_feats * mask).sum(dim=1) / denom

    @staticmethod
    def _module_dtype(module: Optional[nn.Module], fallback_dtype: torch.dtype) -> torch.dtype:
        if module is None:
            return fallback_dtype
        for param in module.parameters():
            return param.dtype
        return fallback_dtype

    @staticmethod
    def _prior_head_logits(
        head: Optional[nn.Module],
        *,
        pooled_video: torch.Tensor,
        video_feats: torch.Tensor,
        vid_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if head is None:
            return None
        if bool(getattr(head, "expects_sequence_inputs", False)):
            return head(video_feats, vid_mask)
        return head(pooled_video)

    def _build_active_slot_mask(
        self,
        *,
        slot_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not self.phrase_slot_active_slot_types:
            return None
        return self._build_phrase_condition_slot_mask(
            slot_count=slot_count,
            device=device,
            dtype=dtype,
            selected_slot_types=self.phrase_slot_active_slot_types,
        )

    @staticmethod
    def _normalize_aux_mask(
        mask: Optional[torch.Tensor],
        feats: torch.Tensor,
    ) -> torch.Tensor:
        if mask is None:
            return feats.detach().abs().sum(dim=-1) > 0
        mask_bool = mask.bool()
        if mask_bool.dim() > 2:
            mask_bool = mask_bool.reshape(mask_bool.size(0), -1)
        if mask_bool.dim() != 2:
            raise ValueError("aux mask must have shape [batch, time].")
        if mask_bool.size(0) != feats.size(0) or mask_bool.size(1) != feats.size(1):
            raise ValueError(
                "aux mask shape mismatch: "
                f"expected {(feats.size(0), feats.size(1))}, got {tuple(mask_bool.shape)}"
            )
        return mask_bool

    def _project_aux_sequence(
        self,
        feats: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        projector: Optional[nn.Module],
        dtype: torch.dtype,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if feats is None or projector is None:
            return None, None, None
        if feats.dim() != 3:
            raise ValueError(f"Expected aux feats shape [batch, time, dim], got {tuple(feats.shape)}")
        proj_dtype = self._module_dtype(projector, feats.dtype)
        projected = projector(feats.to(dtype=proj_dtype))
        if self.aux_visual_token_norm is not None:
            projected = self.aux_visual_token_norm(projected)
        mask_bool = self._normalize_aux_mask(mask, projected)
        projected = projected * mask_bool.unsqueeze(-1).to(projected.dtype)
        pooled = self._masked_mean(projected, mask_bool)
        return projected.to(dtype=dtype), mask_bool, pooled.to(dtype=dtype)

    def _encode_aux_visual_state(
        self,
        *,
        aux_raw_global_feats: Optional[torch.Tensor] = None,
        aux_raw_global_mask: Optional[torch.Tensor] = None,
        aux_patch_feats: Optional[torch.Tensor] = None,
        aux_patch_mask: Optional[torch.Tensor] = None,
        dtype: torch.dtype,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if not self.aux_visual_enable:
            return None

        pooled_states: List[torch.Tensor] = []
        pooled_tokens: List[torch.Tensor] = []
        temporal_tokens: List[torch.Tensor] = []
        temporal_masks: List[torch.Tensor] = []

        if self.aux_raw_global_enable and aux_raw_global_feats is not None:
            raw_tokens, raw_mask, raw_pooled = self._project_aux_sequence(
                aux_raw_global_feats,
                aux_raw_global_mask,
                self.aux_raw_global_proj,
                dtype=dtype,
            )
            if raw_tokens is not None and raw_mask is not None and raw_pooled is not None:
                pooled_states.append(raw_pooled)
                pooled_tokens.append(raw_pooled.unsqueeze(1))
                temporal_tokens.append(raw_tokens)
                temporal_masks.append(raw_mask)

        if self.aux_patch_enable and aux_patch_feats is not None:
            patch_frame_feats = aux_patch_feats
            if patch_frame_feats.dim() == 4:
                patch_frame_feats = patch_frame_feats.mean(dim=2)
            patch_tokens, patch_mask, patch_pooled = self._project_aux_sequence(
                patch_frame_feats,
                aux_patch_mask,
                self.aux_patch_proj,
                dtype=dtype,
            )
            if patch_tokens is not None and patch_mask is not None and patch_pooled is not None:
                pooled_states.append(patch_pooled)
                pooled_tokens.append(patch_pooled.unsqueeze(1))
                temporal_tokens.append(patch_tokens)
                temporal_masks.append(patch_mask)

        if not pooled_states:
            return None

        pooled = torch.stack(pooled_states, dim=0).mean(dim=0)
        pooled_token_tensor = torch.cat(pooled_tokens, dim=1)
        pooled_token_mask = torch.ones(
            pooled_token_tensor.size(0),
            pooled_token_tensor.size(1),
            dtype=torch.bool,
            device=pooled_token_tensor.device,
        )
        temporal = torch.cat(temporal_tokens, dim=1) if temporal_tokens else None
        temporal_mask = torch.cat(temporal_masks, dim=1) if temporal_masks else None

        return {
            "pooled": pooled.to(dtype=dtype),
            "pooled_tokens": pooled_token_tensor.to(dtype=dtype),
            "pooled_token_mask": pooled_token_mask,
            "temporal": temporal.to(dtype=dtype) if temporal is not None else None,
            "temporal_mask": temporal_mask,
        }

    def _encode_progress_state(
        self,
        video_feats: torch.Tensor,
        vid_mask: Optional[torch.Tensor],
        dtype: torch.dtype,
    ) -> Optional[Dict[str, Any]]:
        if not self.phrase_progress_enable:
            return None
        if video_feats.dim() != 3:
            raise ValueError(f"Expected video feats shape [batch, time, dim], got {tuple(video_feats.shape)}")

        mask_bool = self._normalize_aux_mask(vid_mask, video_feats)
        if not bool(mask_bool.any()):
            return None

        progress_dtype = torch.float32
        frame_indices = torch.arange(video_feats.size(1), device=video_feats.device, dtype=progress_dtype)
        frame_indices = frame_indices.unsqueeze(0).expand(video_feats.size(0), -1)
        valid_lengths = mask_bool.sum(dim=1, keepdim=True).to(dtype=progress_dtype).clamp_min(1.0)
        denom = (valid_lengths - 1.0).clamp_min(1.0)
        relative_progress = frame_indices / denom
        relative_progress = torch.where(mask_bool, relative_progress, torch.zeros_like(relative_progress))

        phase_centers = torch.tensor([0.15, 0.50, 0.85], device=video_feats.device, dtype=progress_dtype)
        phase_centers = phase_centers.view(1, 3, 1)
        phase_sigma = 0.22
        raw_weights = torch.exp(-0.5 * ((relative_progress.unsqueeze(1) - phase_centers) / phase_sigma) ** 2)
        raw_weights = raw_weights * mask_bool.unsqueeze(1).to(dtype=progress_dtype)
        norm_weights = raw_weights / raw_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        phase_tokens = torch.einsum("bst,btd->bsd", norm_weights, video_feats.to(dtype=progress_dtype))
        if self.phrase_progress_proj is not None:
            proj_dtype = self._module_dtype(self.phrase_progress_proj, phase_tokens.dtype)
            phase_tokens = self.phrase_progress_proj(phase_tokens.to(dtype=proj_dtype))

        phase_mask = mask_bool.any(dim=1, keepdim=True).expand(-1, 3)
        phase_tokens = phase_tokens * phase_mask.unsqueeze(-1).to(phase_tokens.dtype)
        return {
            "phase_tokens": phase_tokens.to(dtype=dtype),
            "phase_mask": phase_mask,
            "phase_names": ("progress_early", "progress_mid", "progress_late"),
        }

    def _build_phrase_condition_slot_mask(
        self,
        *,
        slot_count: int,
        device: torch.device,
        dtype: torch.dtype,
        selected_slot_types: Sequence[str],
    ) -> Optional[torch.Tensor]:
        normalized_types = [str(slot_type).strip().lower() for slot_type in selected_slot_types if str(slot_type).strip()]
        if not normalized_types:
            return None
        selected_type_set = set(normalized_types)
        slot_specs = StructuredCaptionDataset.get_phrase_slot_type_specs(
            max_phrase_slots=slot_count,
            phrase_slot_schema=self.phrase_slot_schema,
        )
        slot_mask_values = []
        for spec in slot_specs:
            slot_type = str(
                spec.get(
                    "slot_type_family",
                    spec.get("slot_type", "generic"),
                )
            ).strip().lower()
            slot_mask_values.append(1.0 if slot_type in selected_type_set else 0.0)
        if not slot_mask_values or sum(slot_mask_values) <= 0.0:
            return None
        return torch.tensor(slot_mask_values, device=device, dtype=dtype)

    def _build_phrase_condition_slot_family_ids(
        self,
        *,
        slot_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        slot_specs = StructuredCaptionDataset.get_phrase_slot_type_specs(
            max_phrase_slots=slot_count,
            phrase_slot_schema=self.phrase_slot_schema,
        )
        family_ids = [int(spec.get("slot_type_family_id", idx)) for idx, spec in enumerate(slot_specs)]
        return torch.tensor(family_ids, device=device, dtype=torch.long)

    def _pool_phrase_slot_family_summary(
        self,
        phrase_slot_summary: Optional[torch.Tensor],
        phrase_slot_valid: Optional[torch.Tensor],
        selected_slot_types: Sequence[str],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if phrase_slot_summary is None or phrase_slot_summary.dim() != 3:
            return None, None

        batch_size, slot_count, _hidden_dim = phrase_slot_summary.shape
        if phrase_slot_valid is None:
            slot_valid = torch.ones(
                (batch_size, slot_count),
                dtype=torch.bool,
                device=phrase_slot_summary.device,
            )
        else:
            slot_valid = phrase_slot_valid.bool()
        slot_mask = self._build_phrase_condition_slot_mask(
            slot_count=slot_count,
            device=phrase_slot_summary.device,
            dtype=torch.bool,
            selected_slot_types=selected_slot_types,
        )
        if slot_mask is None:
            return None, None
        family_valid = slot_valid & slot_mask.view(1, slot_count)
        if not bool(family_valid.any()):
            return None, family_valid
        family_summary = self._masked_mean(phrase_slot_summary, family_valid)
        family_active = family_valid.any(dim=1)
        family_summary = family_summary * family_active.unsqueeze(-1).to(phrase_slot_summary.dtype)
        return family_summary, family_valid

    def _build_phrase_condition_aux_context(
        self,
        hidden: Optional[torch.Tensor],
        phrase_slot_summary: Optional[torch.Tensor],
        phrase_slot_valid: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if (
            hidden is None
            or phrase_slot_summary is None
            or phrase_slot_summary.dim() != 3
            or not self.phrase_condition_slot_selective_enable
            or self.phrase_condition_aux_query is None
            or self.phrase_condition_aux_key is None
            or self.phrase_condition_aux_value is None
            or self.phrase_condition_aux_gate is None
            or self.phrase_condition_aux_norm is None
        ):
            return None

        batch_size, slot_count, _hidden_dim = phrase_slot_summary.shape
        if phrase_slot_valid is None:
            slot_valid = torch.ones(
                (batch_size, slot_count),
                dtype=torch.bool,
                device=phrase_slot_summary.device,
            )
        else:
            slot_valid = phrase_slot_valid.bool()
        aux_slot_mask = self._build_phrase_condition_slot_mask(
            slot_count=slot_count,
            device=phrase_slot_summary.device,
            dtype=torch.bool,
            selected_slot_types=self.phrase_condition_aux_slot_types,
        )
        if aux_slot_mask is None:
            return None

        aux_valid = slot_valid & aux_slot_mask.view(1, slot_count)
        if not bool(aux_valid.any()):
            return None

        aux_slots = phrase_slot_summary.to(dtype=hidden.dtype) * aux_valid.unsqueeze(-1).to(hidden.dtype)
        query = self.phrase_condition_aux_query(hidden)
        key = self.phrase_condition_aux_key(aux_slots)
        value = self.phrase_condition_aux_value(aux_slots)

        attn_scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(max(1, query.size(-1)))
        attn_scores = attn_scores.masked_fill(~aux_valid.unsqueeze(1), -1e4)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights * aux_valid.unsqueeze(1).to(attn_weights.dtype)
        attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        aux_context = torch.matmul(attn_weights, value)
        aux_global = self._masked_mean(value, aux_valid)
        aux_active = aux_valid.any(dim=1)
        aux_global = aux_global * aux_active.unsqueeze(-1).to(aux_global.dtype)
        aux_context = aux_context + aux_global.unsqueeze(1)
        gate_input = torch.cat([hidden, aux_context], dim=-1)
        gate = torch.sigmoid(self.phrase_condition_aux_gate(gate_input))
        aux_context = self.phrase_condition_aux_norm(aux_context) * gate
        aux_context = aux_context * aux_active.view(-1, 1, 1).to(aux_context.dtype)
        return aux_context

    def _build_phrase_condition_slot_residual_context(
        self,
        hidden: Optional[torch.Tensor],
        phrase_slot_summary: Optional[torch.Tensor],
        phrase_slot_valid: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if (
            hidden is None
            or phrase_slot_summary is None
            or phrase_slot_summary.dim() != 3
            or not self.phrase_condition_slot_residual_enable
            or self.phrase_condition_slot_residual_query is None
            or self.phrase_condition_slot_residual_key is None
            or self.phrase_condition_slot_residual_value is None
            or self.phrase_condition_slot_residual_gate is None
            or self.phrase_condition_slot_residual_norm is None
            or self.phrase_condition_slot_residual_order_embeddings is None
            or self.phrase_condition_slot_residual_type_embeddings is None
        ):
            return None

        batch_size, slot_count, _hidden_dim = phrase_slot_summary.shape
        if phrase_slot_valid is None:
            slot_valid = torch.ones(
                (batch_size, slot_count),
                dtype=torch.bool,
                device=phrase_slot_summary.device,
            )
        else:
            slot_valid = phrase_slot_valid.bool()
        if self.phrase_condition_slot_residual_slot_types:
            slot_mask = self._build_phrase_condition_slot_mask(
                slot_count=slot_count,
                device=phrase_slot_summary.device,
                dtype=torch.bool,
                selected_slot_types=self.phrase_condition_slot_residual_slot_types,
            )
            if slot_mask is None:
                return None
            slot_valid = slot_valid & slot_mask.view(1, slot_count)
        if not bool(slot_valid.any()):
            return None

        order_ids = torch.arange(slot_count, device=hidden.device, dtype=torch.long)
        order_ids = order_ids.clamp_max(max(0, self.max_phrase_slots - 1))
        family_ids = self._build_phrase_condition_slot_family_ids(slot_count=slot_count, device=hidden.device)
        family_ids = family_ids.clamp_max(max(0, self.phrase_condition_slot_residual_type_embeddings.num_embeddings - 1))

        slot_states = phrase_slot_summary.to(device=hidden.device, dtype=hidden.dtype)
        slot_states = slot_states + self.phrase_condition_slot_residual_order_embeddings(order_ids).unsqueeze(0).to(hidden.dtype)
        slot_states = slot_states + self.phrase_condition_slot_residual_type_embeddings(family_ids).unsqueeze(0).to(hidden.dtype)
        slot_states = slot_states * slot_valid.unsqueeze(-1).to(hidden.dtype)

        query = self.phrase_condition_slot_residual_query(hidden)
        key = self.phrase_condition_slot_residual_key(slot_states)
        value = self.phrase_condition_slot_residual_value(slot_states)

        attn_scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(max(1, query.size(-1)))
        attn_scores = attn_scores.masked_fill(~slot_valid.unsqueeze(1), -1e4)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights * slot_valid.unsqueeze(1).to(attn_weights.dtype)
        attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        residual_context = torch.matmul(attn_weights, value)
        residual_global = self._masked_mean(value, slot_valid)
        residual_any = slot_valid.any(dim=1)
        residual_global = residual_global * residual_any.unsqueeze(-1).to(residual_global.dtype)
        residual_context = residual_context + residual_global.unsqueeze(1)
        gate_input = torch.cat([hidden, residual_context], dim=-1)
        gate = torch.sigmoid(self.phrase_condition_slot_residual_gate(gate_input))
        residual_context = self.phrase_condition_slot_residual_norm(residual_context) * gate
        residual_context = residual_context * residual_any.view(-1, 1, 1).to(residual_context.dtype)
        return residual_context

    def _build_phrase_condition_query_bridge_context(
        self,
        hidden: Optional[torch.Tensor],
        phrase_hidden: Optional[torch.Tensor],
        phrase_token_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if (
            hidden is None
            or phrase_hidden is None
            or phrase_hidden.dim() != 3
            or not self.phrase_condition_query_bridge_enable
            or self.phrase_condition_query_bridge_queries is None
            or self.phrase_condition_query_bridge_summary_key is None
            or self.phrase_condition_query_bridge_summary_value is None
            or self.phrase_condition_query_bridge_summary_norm is None
            or self.phrase_condition_query_bridge_hidden_query is None
            or self.phrase_condition_query_bridge_hidden_key is None
            or self.phrase_condition_query_bridge_hidden_value is None
            or self.phrase_condition_query_bridge_gate is None
            or self.phrase_condition_query_bridge_norm is None
        ):
            return None

        if phrase_token_mask is None:
            token_valid = torch.ones(
                phrase_hidden.size(0),
                phrase_hidden.size(1),
                dtype=torch.bool,
                device=phrase_hidden.device,
            )
        else:
            token_valid = phrase_token_mask.bool()
        valid_rows = token_valid.any(dim=1)
        if not bool(valid_rows.any()):
            return None

        query_tokens = self.phrase_condition_query_bridge_queries.unsqueeze(0).expand(
            phrase_hidden.size(0), -1, -1
        ).to(device=phrase_hidden.device, dtype=phrase_hidden.dtype)
        summary_key = self.phrase_condition_query_bridge_summary_key(phrase_hidden)
        summary_value = self.phrase_condition_query_bridge_summary_value(phrase_hidden)
        summary_scores = torch.matmul(query_tokens, summary_key.transpose(1, 2)) / math.sqrt(max(1, summary_key.size(-1)))
        summary_mask = token_valid.unsqueeze(1)
        summary_scores = summary_scores.masked_fill(~summary_mask, -1e4)
        summary_attn = torch.softmax(summary_scores, dim=-1)
        summary_attn = summary_attn * summary_mask.to(summary_attn.dtype)
        summary_attn = summary_attn / summary_attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        bridge_tokens = torch.matmul(summary_attn, summary_value)
        bridge_tokens = self.phrase_condition_query_bridge_summary_norm(bridge_tokens)
        bridge_tokens = bridge_tokens * valid_rows.view(-1, 1, 1).to(bridge_tokens.dtype)

        query = self.phrase_condition_query_bridge_hidden_query(hidden)
        key = self.phrase_condition_query_bridge_hidden_key(bridge_tokens.to(hidden.dtype))
        value = self.phrase_condition_query_bridge_hidden_value(bridge_tokens.to(hidden.dtype))
        bridge_scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(max(1, key.size(-1)))
        bridge_attn = torch.softmax(bridge_scores, dim=-1)
        bridge_context = torch.matmul(bridge_attn, value)
        gate_input = torch.cat([hidden, bridge_context], dim=-1)
        gate = torch.sigmoid(self.phrase_condition_query_bridge_gate(gate_input))
        bridge_context = self.phrase_condition_query_bridge_norm(bridge_context) * gate
        bridge_context = bridge_context * valid_rows.view(-1, 1, 1).to(bridge_context.dtype)
        return bridge_context

    def _build_struct_condition_query_bridge_context(
        self,
        hidden: Optional[torch.Tensor],
        struct_query_bridge_tokens: Optional[torch.Tensor],
        struct_query_bridge_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if (
            hidden is None
            or struct_query_bridge_tokens is None
            or struct_query_bridge_tokens.dim() != 3
            or not self.struct_condition_query_bridge_hidden_enable
            or self.struct_condition_query_bridge_hidden_query is None
            or self.struct_condition_query_bridge_hidden_key is None
            or self.struct_condition_query_bridge_hidden_value is None
            or self.struct_condition_query_bridge_hidden_gate is None
            or self.struct_condition_query_bridge_hidden_norm is None
        ):
            return None

        if struct_query_bridge_mask is None:
            token_valid = torch.ones(
                struct_query_bridge_tokens.size(0),
                struct_query_bridge_tokens.size(1),
                dtype=torch.bool,
                device=struct_query_bridge_tokens.device,
            )
        else:
            token_valid = struct_query_bridge_mask.bool()
        valid_rows = token_valid.any(dim=1)
        if not bool(valid_rows.any()):
            return None

        query = self.struct_condition_query_bridge_hidden_query(hidden)
        key = self.struct_condition_query_bridge_hidden_key(struct_query_bridge_tokens.to(hidden.dtype))
        value = self.struct_condition_query_bridge_hidden_value(struct_query_bridge_tokens.to(hidden.dtype))
        bridge_scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(max(1, key.size(-1)))
        bridge_mask = token_valid.unsqueeze(1)
        bridge_scores = bridge_scores.masked_fill(~bridge_mask, -1e4)
        bridge_attn = torch.softmax(bridge_scores, dim=-1)
        bridge_attn = bridge_attn * bridge_mask.to(bridge_attn.dtype)
        bridge_attn = bridge_attn / bridge_attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        bridge_context = torch.matmul(bridge_attn, value)
        gate_input = torch.cat([hidden, bridge_context], dim=-1)
        gate = torch.sigmoid(self.struct_condition_query_bridge_hidden_gate(gate_input))
        bridge_context = self.struct_condition_query_bridge_hidden_norm(bridge_context) * gate
        bridge_context = bridge_context * valid_rows.view(-1, 1, 1).to(bridge_context.dtype)
        return bridge_context

    def _build_phrase_condition_family_context(
        self,
        hidden: Optional[torch.Tensor],
        phrase_slot_summary: Optional[torch.Tensor],
        phrase_slot_valid: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if (
            hidden is None
            or phrase_slot_summary is None
            or phrase_slot_summary.dim() != 3
            or not self.phrase_condition_family_bridge_enable
            or not self.phrase_condition_family_groups
            or self.phrase_condition_family_embeddings is None
            or self.phrase_condition_family_query is None
            or self.phrase_condition_family_key is None
            or self.phrase_condition_family_value is None
            or self.phrase_condition_family_gate is None
            or self.phrase_condition_family_norm is None
        ):
            return None

        batch_size = phrase_slot_summary.size(0)
        family_tokens: List[torch.Tensor] = []
        family_active_rows: List[torch.Tensor] = []
        for _family_name, slot_types in self.phrase_condition_family_groups:
            family_summary, family_valid = self._pool_phrase_slot_family_summary(
                phrase_slot_summary=phrase_slot_summary,
                phrase_slot_valid=phrase_slot_valid,
                selected_slot_types=slot_types,
            )
            if family_summary is None or family_valid is None:
                family_tokens.append(
                    torch.zeros(
                        (batch_size, hidden.size(-1)),
                        dtype=hidden.dtype,
                        device=hidden.device,
                    )
                )
                family_active_rows.append(
                    torch.zeros(
                        (batch_size,),
                        dtype=torch.bool,
                        device=hidden.device,
                    )
                )
                continue
            family_tokens.append(family_summary.to(device=hidden.device, dtype=hidden.dtype))
            family_active_rows.append(family_valid.any(dim=1).to(device=hidden.device))

        if not family_tokens:
            return None

        family_states = torch.stack(family_tokens, dim=1)
        family_active = torch.stack(family_active_rows, dim=1)
        if not bool(family_active.any()):
            return None

        family_indices = torch.arange(
            family_states.size(1),
            dtype=torch.long,
            device=hidden.device,
        )
        family_states = family_states + self.phrase_condition_family_embeddings(family_indices).unsqueeze(0).to(
            hidden.dtype
        )

        query = self.phrase_condition_family_query(hidden)
        key = self.phrase_condition_family_key(family_states)
        value = self.phrase_condition_family_value(family_states)

        attn_scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(max(1, query.size(-1)))
        attn_scores = attn_scores.masked_fill(~family_active.unsqueeze(1), -1e4)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights * family_active.unsqueeze(1).to(attn_weights.dtype)
        attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        family_context = torch.matmul(attn_weights, value)
        family_global = self._masked_mean(value, family_active)
        family_any = family_active.any(dim=1)
        family_global = family_global * family_any.unsqueeze(-1).to(family_global.dtype)
        family_context = family_context + family_global.unsqueeze(1)
        gate_input = torch.cat([hidden, family_context], dim=-1)
        gate = torch.sigmoid(self.phrase_condition_family_gate(gate_input))
        family_context = self.phrase_condition_family_norm(family_context) * gate
        family_context = family_context * family_any.view(-1, 1, 1).to(family_context.dtype)
        return family_context

    def _collect_phrase_candidate_bank(
        self,
        *,
        token_ids: Optional[torch.Tensor],
        token_hidden: Optional[torch.Tensor],
        token_mask: Optional[torch.Tensor],
        slot_valid: Optional[torch.Tensor] = None,
        selected_slot_types: Sequence[str] = (),
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if token_ids is None or token_hidden is None or token_mask is None:
            return None, None, None, None

        if token_hidden.dim() == 4:
            if token_ids.dim() != 3 or token_mask.dim() != 3:
                raise ValueError("slot candidate bank expects token_ids/token_mask with shape [batch, slots, seq_len].")
            batch_size, slot_count, seq_len = token_ids.shape
            hidden_dim = token_hidden.size(-1)
            aligned_token_ids = token_ids[:, :, : token_hidden.size(2)]
            aligned_token_mask = token_mask[:, :, : token_hidden.size(2)].bool()
            if slot_valid is None:
                active_slots = aligned_token_mask.any(dim=-1)
            else:
                active_slots = slot_valid.bool()
            if selected_slot_types:
                slot_mask = self._build_phrase_condition_slot_mask(
                    slot_count=slot_count,
                    device=token_hidden.device,
                    dtype=torch.bool,
                    selected_slot_types=selected_slot_types,
                )
                if slot_mask is None:
                    return None, None, None, None
                active_slots = active_slots & slot_mask.view(1, slot_count)
            token_valid = aligned_token_mask & active_slots.unsqueeze(-1)
            flat_token_ids = aligned_token_ids.reshape(batch_size, slot_count * token_hidden.size(2))
            flat_token_hidden = token_hidden.reshape(batch_size, slot_count * token_hidden.size(2), hidden_dim)
            flat_token_mask = token_valid.reshape(batch_size, slot_count * token_hidden.size(2))
        else:
            if token_ids.dim() != 2 or token_mask.dim() != 2:
                raise ValueError("flat candidate bank expects token_ids/token_mask with shape [batch, seq_len].")
            flat_token_ids = token_ids[:, : token_hidden.size(1)]
            flat_token_hidden = token_hidden
            flat_token_mask = token_mask[:, : token_hidden.size(1)].bool()

        batch_size = flat_token_ids.size(0)
        device = flat_token_ids.device
        hidden_dim = flat_token_hidden.size(-1)
        topk = self.phrase_condition_candidate_topk
        candidate_ids = torch.full((batch_size, topk), self.pad_token_id, dtype=torch.long, device=device)
        candidate_states = torch.zeros((batch_size, topk, hidden_dim), dtype=flat_token_hidden.dtype, device=device)
        candidate_weights = torch.zeros((batch_size, topk), dtype=flat_token_hidden.dtype, device=device)
        candidate_valid = torch.zeros((batch_size, topk), dtype=torch.bool, device=device)

        for batch_idx in range(batch_size):
            valid_mask = flat_token_mask[batch_idx]
            if not bool(valid_mask.any()):
                continue
            sample_ids = flat_token_ids[batch_idx][valid_mask]
            sample_states = flat_token_hidden[batch_idx][valid_mask]
            sample_valid = (
                sample_ids.ne(self.pad_token_id)
                & sample_ids.ne(self.bos_token_id)
                & sample_ids.ne(self.eos_token_id)
            )
            if not bool(sample_valid.any()):
                continue
            sample_ids = sample_ids[sample_valid]
            sample_states = sample_states[sample_valid]
            unique_ids, inverse_ids, counts = torch.unique(
                sample_ids,
                sorted=False,
                return_inverse=True,
                return_counts=True,
            )
            if unique_ids.numel() == 0:
                continue
            unique_states = torch.zeros(
                (unique_ids.numel(), hidden_dim),
                dtype=sample_states.dtype,
                device=device,
            )
            unique_states.index_add_(0, inverse_ids, sample_states)
            unique_states = unique_states / counts.to(sample_states.dtype).unsqueeze(-1).clamp_min(1.0)
            counts_f = counts.to(sample_states.dtype)
            if unique_ids.numel() > topk:
                top_values, top_indices = torch.topk(counts_f, k=topk, dim=0, largest=True, sorted=True)
                unique_ids = unique_ids[top_indices]
                unique_states = unique_states[top_indices]
                counts_f = top_values
            limit = unique_ids.numel()
            candidate_ids[batch_idx, :limit] = unique_ids
            candidate_states[batch_idx, :limit] = unique_states
            candidate_weights[batch_idx, :limit] = counts_f / counts_f.sum().clamp_min(1.0)
            candidate_valid[batch_idx, :limit] = True

        if not bool(candidate_valid.any()):
            return None, None, None, None
        return candidate_ids, candidate_states, candidate_weights, candidate_valid

    def _apply_phrase_candidate_bias(
        self,
        logits: torch.Tensor,
        hidden: Optional[torch.Tensor],
        *,
        phrase_slot_ids: Optional[torch.Tensor] = None,
        phrase_slot_hidden: Optional[torch.Tensor] = None,
        phrase_slot_token_mask: Optional[torch.Tensor] = None,
        phrase_slot_valid: Optional[torch.Tensor] = None,
        phrase_ids: Optional[torch.Tensor] = None,
        phrase_hidden: Optional[torch.Tensor] = None,
        phrase_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if (
            hidden is None
            or logits is None
            or not self.phrase_condition_candidate_bias_enable
            or self.phrase_condition_candidate_query is None
            or self.phrase_condition_candidate_key is None
            or self.phrase_condition_candidate_value is None
            or self.phrase_condition_candidate_gate is None
            or self.phrase_condition_candidate_scale is None
        ):
            return logits

        if phrase_slot_ids is not None and phrase_slot_hidden is not None and phrase_slot_token_mask is not None:
            candidate_ids, candidate_states, candidate_weights, candidate_valid = self._collect_phrase_candidate_bank(
                token_ids=phrase_slot_ids,
                token_hidden=phrase_slot_hidden,
                token_mask=phrase_slot_token_mask,
                slot_valid=phrase_slot_valid,
                selected_slot_types=self.phrase_condition_candidate_slot_types,
            )
        else:
            candidate_ids, candidate_states, candidate_weights, candidate_valid = self._collect_phrase_candidate_bank(
                token_ids=phrase_ids,
                token_hidden=phrase_hidden,
                token_mask=phrase_token_mask,
            )
        if candidate_ids is None or candidate_states is None or candidate_weights is None or candidate_valid is None:
            return logits

        query = self.phrase_condition_candidate_query(hidden)
        key = self.phrase_condition_candidate_key(candidate_states.to(hidden.dtype))
        value = self.phrase_condition_candidate_value(candidate_states.to(hidden.dtype))
        score_prior = torch.log(candidate_weights.to(hidden.dtype).clamp_min(1e-6)).unsqueeze(1)
        attn_scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(max(1, query.size(-1)))
        attn_scores = attn_scores + score_prior
        attn_scores = attn_scores.masked_fill(~candidate_valid.unsqueeze(1), -1e4)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights * candidate_valid.unsqueeze(1).to(attn_weights.dtype)
        attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        candidate_context = torch.matmul(attn_weights, value)
        candidate_gate = torch.sigmoid(
            self.phrase_condition_candidate_gate(torch.cat([hidden, candidate_context], dim=-1))
        )
        bias_scale = torch.clamp(self.phrase_condition_candidate_scale, min=0.0, max=2.0).to(hidden.dtype)
        bias_source = bias_scale * candidate_gate * attn_weights

        biased_logits = logits.clone()
        batch_size = biased_logits.size(0)
        for batch_idx in range(batch_size):
            valid_ids = candidate_valid[batch_idx]
            if not bool(valid_ids.any()):
                continue
            sample_ids = candidate_ids[batch_idx, valid_ids]
            sample_bias = bias_source[batch_idx, :, valid_ids].to(dtype=biased_logits.dtype)
            biased_logits[batch_idx].index_add_(1, sample_ids, sample_bias)
        return biased_logits

    def _sequence_token_mask(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mask = token_ids.ne(self.pad_token_id)
        if attention_mask is not None:
            mask = mask & attention_mask.bool()
        mask = mask & token_ids.ne(self.bos_token_id) & token_ids.ne(self.eos_token_id)
        return mask

    def _build_struct_condition_state(
        self,
        entity_prior_logits: Optional[torch.Tensor],
        action_prior_logits: Optional[torch.Tensor],
        attribute_prior_logits: Optional[torch.Tensor],
        scene_prior_logits: Optional[torch.Tensor],
        aux_visual_state: Optional[Dict[str, torch.Tensor]],
        dtype: torch.dtype,
    ) -> Dict[str, Optional[torch.Tensor]]:
        cond = None
        source_contexts: List[torch.Tensor] = []
        source_ids: List[int] = []

        if entity_prior_logits is not None and self.entity_to_context is not None:
            ent_prob = torch.sigmoid(entity_prior_logits)
            ent_ctx = self.entity_to_context(ent_prob)
            cond = ent_ctx if cond is None else cond + ent_ctx
            source_contexts.append(ent_ctx)
            source_ids.append(0)

        if action_prior_logits is not None and self.action_to_context is not None:
            act_prob = torch.sigmoid(action_prior_logits)
            act_ctx = self.action_to_context(act_prob)
            cond = act_ctx if cond is None else cond + act_ctx
            source_contexts.append(act_ctx)
            source_ids.append(1)

        if attribute_prior_logits is not None and self.attribute_to_context is not None:
            attr_prob = torch.sigmoid(attribute_prior_logits)
            attr_ctx = self.attribute_to_context(attr_prob)
            if self.attribute_context_scale is not None:
                attr_scale = torch.clamp(self.attribute_context_scale, min=0.0, max=2.0).to(attr_ctx.dtype)
                attr_ctx = attr_scale * attr_ctx
            cond = attr_ctx if cond is None else cond + attr_ctx
            source_contexts.append(attr_ctx)
            source_ids.append(2)

        if scene_prior_logits is not None and self.scene_to_context is not None:
            scene_prob = torch.sigmoid(scene_prior_logits)
            scene_ctx = self.scene_to_context(scene_prob)
            if self.scene_context_scale is not None:
                scene_scale = torch.clamp(self.scene_context_scale, min=0.0, max=2.0).to(scene_ctx.dtype)
                scene_ctx = scene_scale * scene_ctx
            cond = scene_ctx if cond is None else cond + scene_ctx
            source_contexts.append(scene_ctx)
            source_ids.append(3)

        aux_pooled = aux_visual_state.get("pooled") if aux_visual_state is not None else None
        if (
            aux_pooled is not None
            and self.aux_visual_struct_proj is not None
            and self.aux_visual_struct_scale > 0.0
        ):
            aux_ctx = self.aux_visual_struct_proj(
                aux_pooled.to(dtype=self._module_dtype(self.aux_visual_struct_proj, aux_pooled.dtype))
            )
            aux_ctx = float(self.aux_visual_struct_scale) * aux_ctx
            cond = aux_ctx if cond is None else cond + aux_ctx
            source_contexts.append(aux_ctx)
            source_ids.append(4)

        struct_query_bridge_tokens = None
        struct_query_bridge_mask = None
        if (
            source_contexts
            and self.struct_condition_query_bridge_enable
            and self.struct_condition_query_bridge_queries is not None
            and self.struct_condition_query_bridge_source_key is not None
            and self.struct_condition_query_bridge_source_value is not None
            and self.struct_condition_query_bridge_token_norm is not None
            and self.struct_condition_query_bridge_scale is not None
        ):
            source_tokens = torch.stack(source_contexts, dim=1)
            source_valid = torch.ones(
                source_tokens.size(0),
                source_tokens.size(1),
                dtype=torch.bool,
                device=source_tokens.device,
            )
            if self.struct_condition_query_bridge_source_embeddings is not None and source_ids:
                source_id_tensor = torch.tensor(source_ids, dtype=torch.long, device=source_tokens.device)
                source_tokens = source_tokens + self.struct_condition_query_bridge_source_embeddings(source_id_tensor).unsqueeze(0).to(
                    dtype=source_tokens.dtype
                )
            query_tokens = self.struct_condition_query_bridge_queries.unsqueeze(0).expand(
                source_tokens.size(0), -1, -1
            ).to(device=source_tokens.device, dtype=source_tokens.dtype)
            bridge_key = self.struct_condition_query_bridge_source_key(source_tokens)
            bridge_value = self.struct_condition_query_bridge_source_value(source_tokens)
            bridge_scores = torch.matmul(query_tokens, bridge_key.transpose(1, 2)) / math.sqrt(max(1, bridge_key.size(-1)))
            bridge_mask = source_valid.unsqueeze(1)
            bridge_scores = bridge_scores.masked_fill(~bridge_mask, -1e4)
            bridge_attn = torch.softmax(bridge_scores, dim=-1)
            bridge_attn = bridge_attn * bridge_mask.to(bridge_attn.dtype)
            bridge_attn = bridge_attn / bridge_attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            struct_query_bridge_tokens = torch.matmul(bridge_attn, bridge_value)
            struct_query_bridge_tokens = self.struct_condition_query_bridge_token_norm(
                struct_query_bridge_tokens + query_tokens
            )
            struct_query_bridge_mask = torch.ones(
                struct_query_bridge_tokens.size(0),
                struct_query_bridge_tokens.size(1),
                dtype=torch.bool,
                device=struct_query_bridge_tokens.device,
            )
            bridge_summary = self._masked_mean(struct_query_bridge_tokens, struct_query_bridge_mask)
            bridge_scale = torch.clamp(
                self.struct_condition_query_bridge_scale,
                min=0.0,
                max=2.0,
            ).to(bridge_summary.dtype)
            cond = bridge_scale * bridge_summary if cond is None else cond + bridge_scale * bridge_summary

        if cond is None:
            return {
                "struct_condition": None,
                "struct_query_bridge_tokens": struct_query_bridge_tokens,
                "struct_query_bridge_mask": struct_query_bridge_mask,
            }

        gate = torch.sigmoid(self.condition_gate(cond))
        cond = self.condition_dropout(torch.tanh(cond) * gate)
        cond = self.condition_norm(cond)
        return {
            "struct_condition": cond.to(dtype=dtype),
            "struct_query_bridge_tokens": (
                struct_query_bridge_tokens.to(dtype=dtype)
                if struct_query_bridge_tokens is not None
                else None
            ),
            "struct_query_bridge_mask": struct_query_bridge_mask,
        }

    def _build_struct_context(
        self,
        entity_prior_logits: Optional[torch.Tensor],
        action_prior_logits: Optional[torch.Tensor],
        attribute_prior_logits: Optional[torch.Tensor],
        scene_prior_logits: Optional[torch.Tensor],
        aux_visual_state: Optional[Dict[str, torch.Tensor]],
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        return self._build_struct_condition_state(
            entity_prior_logits=entity_prior_logits,
            action_prior_logits=action_prior_logits,
            attribute_prior_logits=attribute_prior_logits,
            scene_prior_logits=scene_prior_logits,
            aux_visual_state=aux_visual_state,
            dtype=dtype,
        ).get("struct_condition")

    @staticmethod
    def _weighted_label_context(
        prior_logits: Optional[torch.Tensor],
        label_embeddings: Optional[nn.Embedding],
    ) -> Optional[torch.Tensor]:
        if prior_logits is None or label_embeddings is None:
            return None
        if prior_logits.numel() == 0 or label_embeddings.weight.numel() == 0:
            return None
        probs = torch.sigmoid(prior_logits)
        return torch.matmul(probs, label_embeddings.weight.to(dtype=probs.dtype))

    @staticmethod
    def _topk_weighted_label_context(
        prior_logits: Optional[torch.Tensor],
        label_embeddings: Optional[nn.Embedding],
        topk: int,
    ) -> Optional[torch.Tensor]:
        if prior_logits is None or label_embeddings is None:
            return None
        if prior_logits.numel() == 0 or label_embeddings.weight.numel() == 0:
            return None
        probs = torch.sigmoid(prior_logits)
        topk_k = min(max(1, int(topk)), int(probs.size(-1)))
        if topk_k <= 0:
            return None
        topk_values, topk_indices = probs.topk(topk_k, dim=-1)
        norm_weights = topk_values / topk_values.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        zero_rows = topk_values.sum(dim=-1, keepdim=True) <= 1e-6
        if bool(zero_rows.any()):
            uniform = torch.full_like(norm_weights, 1.0 / float(topk_k))
            norm_weights = torch.where(zero_rows.expand_as(norm_weights), uniform, norm_weights)
        topk_embeddings = label_embeddings(topk_indices).to(dtype=norm_weights.dtype)
        return (topk_embeddings * norm_weights.unsqueeze(-1)).sum(dim=1)

    def _collect_slot_source_bank(
        self,
        *,
        entity_prior_logits: Optional[torch.Tensor],
        action_prior_logits: Optional[torch.Tensor],
        attribute_prior_logits: Optional[torch.Tensor],
        scene_prior_logits: Optional[torch.Tensor],
        struct_context: Optional[torch.Tensor],
        progress_state: Optional[Dict[str, Any]],
        dtype: torch.dtype,
    ) -> Tuple[Optional[torch.Tensor], Tuple[str, ...]]:
        source_items = []
        for source_name, prior_logits, label_embeddings in (
            ("entity", entity_prior_logits, self.entity_label_embeddings),
            ("action", action_prior_logits, self.action_label_embeddings),
            ("attribute", attribute_prior_logits, self.attribute_label_embeddings),
            ("scene", scene_prior_logits, self.scene_label_embeddings),
        ):
            ctx = self._weighted_label_context(prior_logits, label_embeddings)
            if ctx is not None:
                source_items.append((source_name, ctx.to(dtype=dtype)))

        if struct_context is not None:
            source_items.append(("struct", struct_context.to(dtype=dtype)))
        if progress_state is not None and self.phrase_progress_source_scale > 0.0:
            phase_tokens = progress_state.get("phase_tokens")
            phase_mask = progress_state.get("phase_mask")
            phase_names = tuple(progress_state.get("phase_names", ()))
            if phase_tokens is not None and phase_tokens.dim() == 3 and phase_tokens.size(1) == len(phase_names):
                scaled_phase_tokens = float(self.phrase_progress_source_scale) * phase_tokens.to(dtype=dtype)
                if phase_mask is not None:
                    scaled_phase_tokens = scaled_phase_tokens * phase_mask.unsqueeze(-1).to(scaled_phase_tokens.dtype)
                for phase_idx, phase_name in enumerate(phase_names):
                    source_items.append((str(phase_name), scaled_phase_tokens[:, phase_idx, :]))

        if not source_items:
            return None, tuple()

        source_names = tuple(source_name for source_name, _ in source_items)
        source_bank = torch.stack([ctx for _, ctx in source_items], dim=1)
        return source_bank, source_names

    def _collect_slot_role_anchor_source_bank(
        self,
        *,
        entity_prior_logits: Optional[torch.Tensor],
        action_prior_logits: Optional[torch.Tensor],
        attribute_prior_logits: Optional[torch.Tensor],
        scene_prior_logits: Optional[torch.Tensor],
        dtype: torch.dtype,
    ) -> Tuple[Optional[torch.Tensor], Tuple[str, ...]]:
        if not self.phrase_slot_role_anchor_enable:
            return None, tuple()

        source_items = []
        for source_name, prior_logits, label_embeddings in (
            ("entity", entity_prior_logits, self.entity_label_embeddings),
            ("action", action_prior_logits, self.action_label_embeddings),
            ("attribute", attribute_prior_logits, self.attribute_label_embeddings),
            ("scene", scene_prior_logits, self.scene_label_embeddings),
        ):
            ctx = self._topk_weighted_label_context(
                prior_logits,
                label_embeddings,
                topk=self.phrase_slot_role_anchor_topk,
            )
            if ctx is not None:
                source_items.append((source_name, ctx.to(dtype=dtype)))

        if not source_items:
            return None, tuple()

        source_names = tuple(source_name for source_name, _ in source_items)
        source_bank = torch.stack([ctx for _, ctx in source_items], dim=1)
        return source_bank, source_names

    def _build_slot_source_bias(
        self,
        *,
        slot_count: int,
        source_names: Tuple[str, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        prior_map = self.get_slot_source_prior_map(self.phrase_slot_schema)
        if prior_map is None or not source_names:
            return None

        slot_specs = StructuredCaptionDataset.get_phrase_slot_type_specs(
            max_phrase_slots=slot_count,
            phrase_slot_schema=self.phrase_slot_schema,
        )
        bias_rows = []
        for spec in slot_specs:
            slot_type = str(spec.get("slot_type", "extra"))
            source_prior = prior_map.get(
                slot_type,
                prior_map.get("extra", prior_map.get("generic", {})),
            )
            bias_rows.append([float(source_prior.get(source_name, 0.0)) for source_name in source_names])

        if not bias_rows:
            return None
        return torch.tensor(bias_rows, device=device, dtype=dtype)

    def _build_slot_planner_contexts(
        self,
        entity_prior_logits: Optional[torch.Tensor],
        action_prior_logits: Optional[torch.Tensor],
        attribute_prior_logits: Optional[torch.Tensor],
        scene_prior_logits: Optional[torch.Tensor],
        struct_context: Optional[torch.Tensor],
        progress_state: Optional[Dict[str, Any]],
        slot_count: int,
        dtype: torch.dtype,
    ) -> Tuple[Optional[torch.Tensor], Tuple[str, ...], Optional[torch.Tensor]]:
        if (
            not self.phrase_slot_planner_enable
            or self.slot_role_embeddings is None
            or self.slot_planner_query is None
            or self.slot_planner_key is None
            or self.slot_planner_value is None
            or self.slot_planner_norm is None
        ):
            return None, tuple(), None

        source_bank, source_names = self._collect_slot_source_bank(
            entity_prior_logits=entity_prior_logits,
            action_prior_logits=action_prior_logits,
            attribute_prior_logits=attribute_prior_logits,
            scene_prior_logits=scene_prior_logits,
            struct_context=struct_context,
            progress_state=progress_state,
            dtype=dtype,
        )
        if source_bank is None or not source_names:
            return None, tuple(), None

        batch_size = source_bank.size(0)
        role_ids = torch.arange(slot_count, device=source_bank.device, dtype=torch.long)
        role_emb = self.slot_role_embeddings(role_ids).unsqueeze(0).expand(batch_size, -1, -1).to(dtype=dtype)

        query = self.slot_planner_query(role_emb)
        key = self.slot_planner_key(source_bank)
        value = self.slot_planner_value(source_bank)

        attn_scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(max(1, query.size(-1)))
        source_bias = self._build_slot_source_bias(
            slot_count=slot_count,
            source_names=source_names,
            device=source_bank.device,
            dtype=attn_scores.dtype,
        )
        if source_bias is not None:
            attn_scores = attn_scores + source_bias.unsqueeze(0)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        slot_contexts = torch.matmul(attn_weights, value)
        slot_contexts = self.slot_planner_norm(slot_contexts + role_emb)
        slot_contexts = self._apply_slot_planner_flow(slot_contexts=slot_contexts, dtype=dtype)
        return slot_contexts.to(dtype=dtype), source_names, attn_weights.to(dtype=dtype)

    def _apply_slot_planner_flow(
        self,
        *,
        slot_contexts: Optional[torch.Tensor],
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if (
            slot_contexts is None
            or not self.phrase_slot_planner_flow_enable
            or self.phrase_slot_planner_flow_scale <= 0.0
            or self.slot_planner_flow_query is None
            or self.slot_planner_flow_key is None
            or self.slot_planner_flow_value is None
            or self.slot_planner_flow_gate is None
            or self.slot_planner_flow_update is None
            or self.slot_planner_flow_norm is None
            or slot_contexts.dim() != 3
            or slot_contexts.size(1) <= 1
        ):
            return slot_contexts.to(dtype=dtype) if slot_contexts is not None else None

        batch_size, slot_count, _hidden_dim = slot_contexts.shape
        flow_input = slot_contexts.to(dtype=dtype)
        selected_slot_mask = has_prev = torch.tril(
            torch.ones((slot_count, slot_count), device=slot_contexts.device, dtype=torch.bool),
            diagonal=-1,
        ).any(dim=-1).view(1, slot_count, 1)
        slot_mask = self._build_slot_planner_flow_slot_mask(
            slot_count=slot_count,
            device=slot_contexts.device,
            dtype=dtype,
        )
        if self.phrase_slot_planner_flow_slot_types and slot_mask is None:
            return flow_input
        if slot_mask is not None:
            selected_slot_mask = selected_slot_mask & slot_mask.view(1, slot_count, 1).to(dtype=torch.bool)

        query = self.slot_planner_flow_query(flow_input)
        key = self.slot_planner_flow_key(flow_input)
        value = self.slot_planner_flow_value(flow_input)

        attn_scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(max(1, query.size(-1)))
        causal_prev_mask = torch.tril(
            torch.ones((slot_count, slot_count), device=slot_contexts.device, dtype=torch.bool),
            diagonal=-1,
        )
        attn_scores = attn_scores.masked_fill(
            ~causal_prev_mask.unsqueeze(0),
            torch.finfo(attn_scores.dtype).min,
        )
        attn_weights = torch.softmax(attn_scores, dim=-1)

        has_prev_float = has_prev.to(dtype=attn_weights.dtype)
        attn_weights = attn_weights * has_prev_float
        attn_denominator = attn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        attn_weights = torch.where(has_prev, attn_weights / attn_denominator, torch.zeros_like(attn_weights))

        prev_memory = torch.matmul(attn_weights, value)
        novelty = flow_input - prev_memory
        flow_gate = torch.sigmoid(self.slot_planner_flow_gate(torch.cat([flow_input, prev_memory], dim=-1)))
        selected_slot_float = selected_slot_mask.to(dtype=flow_gate.dtype)
        flow_update = self.slot_planner_flow_update(novelty) * flow_gate * selected_slot_float
        refined_contexts = flow_input + (self.phrase_slot_planner_flow_scale * flow_update)
        refined_contexts = self.slot_planner_flow_norm(refined_contexts)
        refined_contexts = torch.where(
            selected_slot_mask.expand(batch_size, slot_count, refined_contexts.size(-1)),
            refined_contexts,
            flow_input,
        )
        return refined_contexts.to(dtype=dtype)

    def _build_slot_family_mask(
        self,
        *,
        slot_count: int,
        slot_types: Sequence[str],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not slot_types:
            return None

        slot_specs = StructuredCaptionDataset.get_phrase_slot_type_specs(
            max_phrase_slots=slot_count,
            phrase_slot_schema=self.phrase_slot_schema,
        )
        selected_slot_types = set(slot_types)
        slot_mask_values = []
        for spec in slot_specs:
            slot_type = str(
                spec.get(
                    "slot_type_family",
                    spec.get("slot_type", "generic"),
                )
            ).strip().lower()
            slot_mask_values.append(1.0 if slot_type in selected_slot_types else 0.0)

        if not slot_mask_values or sum(slot_mask_values) <= 0.0:
            return None
        return torch.tensor(slot_mask_values, device=device, dtype=dtype)

    def _build_slot_planner_flow_slot_mask(
        self,
        *,
        slot_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        return self._build_slot_family_mask(
            slot_count=slot_count,
            slot_types=self.phrase_slot_planner_flow_slot_types,
            device=device,
            dtype=dtype,
        )

    def _predict_slot_presence(
        self,
        *,
        batch_size: int,
        slot_count: int,
        device: torch.device,
        dtype: torch.dtype,
        struct_context: Optional[torch.Tensor],
        slot_contexts: Optional[torch.Tensor],
        entity_prior_logits: Optional[torch.Tensor],
        action_prior_logits: Optional[torch.Tensor],
        attribute_prior_logits: Optional[torch.Tensor],
        scene_prior_logits: Optional[torch.Tensor],
        source_names: Optional[Sequence[str]],
        slot_source_weights: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if not self.phrase_slot_presence_enable or self.slot_presence_head is None:
            return {}

        presence_context = slot_contexts
        if presence_context is None:
            parts = []
            if struct_context is not None:
                parts.append(struct_context.unsqueeze(1).expand(batch_size, slot_count, -1).to(dtype=dtype))
            if self.slot_role_embeddings is not None:
                role_ids = torch.arange(slot_count, device=device, dtype=torch.long)
                role_emb = self.slot_role_embeddings(role_ids).unsqueeze(0).expand(batch_size, -1, -1).to(dtype=dtype)
                parts.append(role_emb)
            if not parts:
                return {}
            presence_context = parts[0]
            for extra in parts[1:]:
                presence_context = presence_context + extra

        support_context = self._build_slot_presence_support_context(
            slot_count=slot_count,
            source_names=source_names,
            slot_source_weights=slot_source_weights,
            device=device,
            dtype=dtype,
        )
        if support_context is not None:
            if presence_context is None:
                presence_context = support_context
            else:
                presence_context = presence_context + support_context

        evidence_context = self._build_slot_presence_evidence_context(
            slot_count=slot_count,
            source_names=source_names,
            entity_prior_logits=entity_prior_logits,
            action_prior_logits=action_prior_logits,
            attribute_prior_logits=attribute_prior_logits,
            scene_prior_logits=scene_prior_logits,
            device=device,
            dtype=dtype,
        )
        if evidence_context is not None:
            if presence_context is None:
                presence_context = evidence_context
            else:
                presence_context = presence_context + evidence_context

        presence_logits = self.slot_presence_head(presence_context.to(dtype=dtype)).squeeze(-1)
        active_slot_mask = self._build_active_slot_mask(
            slot_count=slot_count,
            device=device,
            dtype=presence_logits.dtype,
        )
        if active_slot_mask is not None:
            active_slot_mask_bool = active_slot_mask.bool().view(1, slot_count)
            presence_logits = presence_logits.masked_fill(~active_slot_mask_bool, -60.0)
        else:
            active_slot_mask_bool = None
        presence_probs = torch.sigmoid(presence_logits)
        presence_thresholds = torch.full(
            (slot_count,),
            self.phrase_slot_presence_threshold,
            device=device,
            dtype=presence_probs.dtype,
        )
        if self._phrase_slot_presence_thresholds_tensor is not None:
            custom_thresholds = self._phrase_slot_presence_thresholds_tensor
            copy_count = min(slot_count, int(custom_thresholds.numel()))
            if copy_count > 0:
                presence_thresholds[:copy_count] = custom_thresholds[:copy_count].to(
                    device=device,
                    dtype=presence_probs.dtype,
                )
        presence_pred = presence_probs >= presence_thresholds.unsqueeze(0)
        if active_slot_mask_bool is not None:
            presence_pred = presence_pred & active_slot_mask_bool
        raw_presence_pred = presence_pred.clone()
        fallback_applied = torch.zeros(batch_size, dtype=torch.bool, device=device)
        fallback_index = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        if presence_pred.size(1) > 0:
            presence_pred = presence_pred.clone()
            empty_rows = ~presence_pred.any(dim=1)
            if empty_rows.any():
                fallback_applied = empty_rows.clone()
                fallback_probs = presence_probs
                if active_slot_mask_bool is not None:
                    fallback_probs = presence_probs.masked_fill(~active_slot_mask_bool, -1.0)
                fallback_slot = fallback_probs.argmax(dim=1)
                fallback_index[empty_rows] = fallback_slot[empty_rows]
                presence_pred[empty_rows] = False
                presence_pred[empty_rows, fallback_slot[empty_rows]] = True
        return {
            "phrase_slot_presence_logits": presence_logits,
            "phrase_slot_presence_probs": presence_probs,
            "phrase_slot_presence_raw_pred": raw_presence_pred,
            "phrase_slot_presence_pred": presence_pred,
            "phrase_slot_presence_thresholds": presence_thresholds.unsqueeze(0).expand(batch_size, -1),
            "phrase_slot_presence_fallback_mask": fallback_applied,
            "phrase_slot_presence_fallback_index": fallback_index,
        }

    def _build_slot_source_target_tensor(
        self,
        *,
        slot_count: int,
        source_names: Sequence[str],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        prior_map = self.get_slot_source_prior_map(self.phrase_slot_schema)
        if prior_map is None or not source_names:
            return None

        slot_specs = StructuredCaptionDataset.get_phrase_slot_type_specs(
            max_phrase_slots=slot_count,
            phrase_slot_schema=self.phrase_slot_schema,
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

    @staticmethod
    def _build_slot_support_features(
        *,
        slot_count: int,
        source_names: Optional[Sequence[str]],
        slot_source_weights: Optional[torch.Tensor],
        target: Optional[torch.Tensor],
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if slot_source_weights is None or not source_names or target is None:
            return None
        if slot_source_weights.dim() != 3:
            return None

        weights = slot_source_weights
        target = target.unsqueeze(0).expand(weights.size(0), -1, -1).to(device=weights.device, dtype=weights.dtype)
        support_mass = (weights * target).sum(dim=-1)
        peak_mass = weights.max(dim=-1).values
        source_count = max(1, int(weights.size(-1)))
        entropy = -(weights.clamp_min(1e-8) * torch.log(weights.clamp_min(1e-8))).sum(dim=-1)
        if source_count > 1:
            entropy = entropy / math.log(float(source_count))
        focus_mass = 1.0 - entropy.clamp(0.0, 1.0)
        return torch.stack([support_mass, peak_mass, focus_mass], dim=-1).to(dtype=dtype)

    def _build_slot_role_anchor_slot_mask(
        self,
        *,
        slot_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        return self._build_slot_family_mask(
            slot_count=slot_count,
            slot_types=self.phrase_slot_role_anchor_slot_types,
            device=device,
            dtype=dtype,
        )

    def _build_slot_presence_support_context(
        self,
        *,
        slot_count: int,
        source_names: Optional[Sequence[str]],
        slot_source_weights: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not self.phrase_slot_presence_support_enable or self.slot_presence_support_proj is None:
            return None
        if slot_source_weights is None or not source_names:
            return None
        if slot_source_weights.dim() != 3:
            return None

        target = self._build_slot_source_target_tensor(
            slot_count=slot_count,
            source_names=source_names,
            device=device,
            dtype=slot_source_weights.dtype,
        )
        support_features = self._build_slot_support_features(
            slot_count=slot_count,
            source_names=source_names,
            slot_source_weights=slot_source_weights.detach(),
            target=target,
            dtype=dtype,
        )
        if support_features is None:
            return None
        support_context = self.slot_presence_support_proj(support_features)
        return self._mask_slot_presence_context(
            support_context,
            slot_count=slot_count,
        )

    @staticmethod
    def _summarize_detached_prior_logits(
        prior_logits: Optional[torch.Tensor],
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if prior_logits is None or prior_logits.numel() == 0:
            return None
        probs = torch.sigmoid(prior_logits.detach())
        peak = probs.max(dim=-1).values
        mean = probs.mean(dim=-1)
        topk_k = min(3, int(probs.size(-1)))
        if topk_k <= 0:
            topk_mean = mean
        else:
            topk_mean = probs.topk(topk_k, dim=-1).values.mean(dim=-1)
        return peak, topk_mean, mean

    def _build_slot_presence_evidence_context(
        self,
        *,
        slot_count: int,
        source_names: Optional[Sequence[str]],
        entity_prior_logits: Optional[torch.Tensor],
        action_prior_logits: Optional[torch.Tensor],
        attribute_prior_logits: Optional[torch.Tensor],
        scene_prior_logits: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        evidence_features = self._build_slot_evidence_features(
            slot_count=slot_count,
            source_names=source_names,
            entity_prior_logits=entity_prior_logits,
            action_prior_logits=action_prior_logits,
            attribute_prior_logits=attribute_prior_logits,
            scene_prior_logits=scene_prior_logits,
            device=device,
            dtype=dtype,
        )
        if (
            not self.phrase_slot_presence_evidence_enable
            or self.slot_presence_evidence_proj is None
            or evidence_features is None
        ):
            return None
        evidence_context = self.slot_presence_evidence_proj(evidence_features)
        return self._mask_slot_presence_context(
            evidence_context,
            slot_count=slot_count,
        )

    def _build_slot_evidence_features(
        self,
        *,
        slot_count: int,
        source_names: Optional[Sequence[str]],
        entity_prior_logits: Optional[torch.Tensor],
        action_prior_logits: Optional[torch.Tensor],
        attribute_prior_logits: Optional[torch.Tensor],
        scene_prior_logits: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:

        source_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for source_name, prior_logits in (
            ("entity", entity_prior_logits),
            ("action", action_prior_logits),
            ("attribute", attribute_prior_logits),
            ("scene", scene_prior_logits),
        ):
            stats = self._summarize_detached_prior_logits(prior_logits)
            if stats is not None:
                source_stats[source_name] = stats
        if not source_stats:
            return None

        evidence_source_names = tuple(
            source_name for source_name in (source_names or tuple(source_stats.keys())) if source_name in source_stats
        )
        if not evidence_source_names:
            evidence_source_names = tuple(source_stats.keys())

        target = self._build_slot_source_target_tensor(
            slot_count=slot_count,
            source_names=evidence_source_names,
            device=device,
            dtype=dtype,
        )
        if target is None:
            return None

        target = target.clamp_min(0.0)
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        peak = torch.stack([source_stats[source_name][0].to(device=device, dtype=dtype) for source_name in evidence_source_names], dim=-1)
        topk_mean = torch.stack(
            [source_stats[source_name][1].to(device=device, dtype=dtype) for source_name in evidence_source_names],
            dim=-1,
        )
        mean = torch.stack([source_stats[source_name][2].to(device=device, dtype=dtype) for source_name in evidence_source_names], dim=-1)

        batch_size = peak.size(0)
        target = target.unsqueeze(0).expand(batch_size, -1, -1)
        peak = peak.unsqueeze(1)
        topk_mean = topk_mean.unsqueeze(1)
        mean = mean.unsqueeze(1)
        zero_feature = torch.zeros((batch_size, slot_count), device=device, dtype=dtype)
        source_index = {source_name: idx for idx, source_name in enumerate(evidence_source_names)}

        target_peak = (target * peak).sum(dim=-1)
        target_topk = (target * topk_mean).sum(dim=-1)
        target_mean = (target * mean).sum(dim=-1)

        def _weighted_source_support(source_name: str) -> torch.Tensor:
            idx = source_index.get(source_name)
            if idx is None:
                return zero_feature
            return target[..., idx] * topk_mean[..., idx]

        def _pair_support(source_a: str, source_b: str) -> torch.Tensor:
            idx_a = source_index.get(source_a)
            idx_b = source_index.get(source_b)
            if idx_a is None or idx_b is None:
                return zero_feature
            pair_weight = torch.sqrt((target[..., idx_a] * target[..., idx_b]).clamp_min(0.0))
            pair_strength = torch.sqrt((topk_mean[..., idx_a] * topk_mean[..., idx_b]).clamp_min(0.0))
            return pair_weight * pair_strength

        evidence_features = torch.stack(
            [
                target_peak,
                target_topk,
                target_mean,
                _weighted_source_support("entity"),
                _weighted_source_support("action"),
                _weighted_source_support("attribute"),
                _weighted_source_support("scene"),
                _pair_support("entity", "action"),
                _pair_support("entity", "attribute"),
                _pair_support("action", "attribute"),
                _pair_support("action", "scene"),
                _pair_support("entity", "scene"),
            ],
            dim=-1,
        ).to(dtype=dtype)
        return evidence_features

    def _build_slot_guidance_context(
        self,
        *,
        slot_count: int,
        source_bank: Optional[torch.Tensor],
        source_names: Optional[Sequence[str]],
        slot_source_weights: Optional[torch.Tensor],
        entity_prior_logits: Optional[torch.Tensor],
        action_prior_logits: Optional[torch.Tensor],
        attribute_prior_logits: Optional[torch.Tensor],
        scene_prior_logits: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if (
            not self.phrase_slot_guidance_enable
            or self.slot_guidance_target_proj is None
            or self.slot_guidance_support_proj is None
            or self.slot_guidance_evidence_proj is None
            or self.slot_guidance_norm is None
        ):
            return None
        if source_bank is None or not source_names:
            return None

        target = self._build_slot_source_target_tensor(
            slot_count=slot_count,
            source_names=source_names,
            device=device,
            dtype=dtype,
        )
        if target is None:
            return None

        target = target.unsqueeze(0).expand(source_bank.size(0), -1, -1)
        guidance_context = torch.matmul(target.to(dtype=source_bank.dtype), source_bank.to(dtype=source_bank.dtype))
        guidance_context = self.slot_guidance_target_proj(guidance_context.to(dtype=dtype))

        support_features = self._build_slot_support_features(
            slot_count=slot_count,
            source_names=source_names,
            slot_source_weights=slot_source_weights,
            target=target[0].to(device=device, dtype=slot_source_weights.dtype if slot_source_weights is not None else dtype),
            dtype=dtype,
        )
        if support_features is not None:
            guidance_context = guidance_context + self.slot_guidance_support_proj(support_features)

        evidence_features = self._build_slot_evidence_features(
            slot_count=slot_count,
            source_names=source_names,
            entity_prior_logits=entity_prior_logits,
            action_prior_logits=action_prior_logits,
            attribute_prior_logits=attribute_prior_logits,
            scene_prior_logits=scene_prior_logits,
            device=device,
            dtype=dtype,
        )
        if evidence_features is not None:
            guidance_context = guidance_context + self.slot_guidance_evidence_proj(evidence_features)
        return self.slot_guidance_norm(guidance_context.to(dtype=dtype))

    def _build_slot_role_anchor_context(
        self,
        *,
        slot_count: int,
        entity_prior_logits: Optional[torch.Tensor],
        action_prior_logits: Optional[torch.Tensor],
        attribute_prior_logits: Optional[torch.Tensor],
        scene_prior_logits: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if (
            not self.phrase_slot_role_anchor_enable
            or self.phrase_slot_role_anchor_scale <= 0.0
            or self.slot_role_embeddings is None
            or self.slot_role_anchor_target_proj is None
            or self.slot_role_anchor_role_proj is None
            or self.slot_role_anchor_norm is None
        ):
            return None

        anchor_source_bank, anchor_source_names = self._collect_slot_role_anchor_source_bank(
            entity_prior_logits=entity_prior_logits,
            action_prior_logits=action_prior_logits,
            attribute_prior_logits=attribute_prior_logits,
            scene_prior_logits=scene_prior_logits,
            dtype=dtype,
        )
        if anchor_source_bank is None or not anchor_source_names:
            return None

        target = self._build_slot_source_target_tensor(
            slot_count=slot_count,
            source_names=anchor_source_names,
            device=device,
            dtype=anchor_source_bank.dtype,
        )
        if target is None:
            return None

        batch_size = anchor_source_bank.size(0)
        target = target.unsqueeze(0).expand(batch_size, -1, -1)
        anchor_context = torch.matmul(target, anchor_source_bank.to(dtype=target.dtype))
        anchor_context = self.slot_role_anchor_target_proj(anchor_context.to(dtype=dtype))

        role_ids = torch.arange(slot_count, device=device, dtype=torch.long)
        role_emb = self.slot_role_embeddings(role_ids).unsqueeze(0).expand(batch_size, -1, -1).to(dtype=dtype)
        anchor_context = anchor_context + self.slot_role_anchor_role_proj(role_emb)
        anchor_context = self.slot_role_anchor_norm(anchor_context.to(dtype=dtype))

        slot_mask = self._build_slot_role_anchor_slot_mask(
            slot_count=slot_count,
            device=device,
            dtype=anchor_context.dtype,
        )
        if self.phrase_slot_role_anchor_slot_types and slot_mask is None:
            return None
        if slot_mask is not None:
            anchor_context = anchor_context * slot_mask.view(1, slot_count, 1)
        return anchor_context * self.phrase_slot_role_anchor_scale

    def _build_slot_presence_context_slot_mask(
        self,
        *,
        slot_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        return self._build_slot_family_mask(
            slot_count=slot_count,
            slot_types=self.phrase_slot_presence_context_slot_types,
            device=device,
            dtype=dtype,
        )

    def _mask_slot_presence_context(
        self,
        slot_context: Optional[torch.Tensor],
        *,
        slot_count: int,
    ) -> Optional[torch.Tensor]:
        if slot_context is None:
            return None
        slot_mask = self._build_slot_presence_context_slot_mask(
            slot_count=slot_count,
            device=slot_context.device,
            dtype=slot_context.dtype,
        )
        if self.phrase_slot_presence_context_slot_types and slot_mask is None:
            return None
        if slot_mask is not None:
            slot_context = slot_context * slot_mask.view(1, slot_count, 1)
        return slot_context

    def _merge_slot_contexts(
        self,
        base_contexts: Optional[torch.Tensor],
        extra_contexts: Optional[torch.Tensor],
        *,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if base_contexts is None:
            return extra_contexts.to(dtype=dtype) if extra_contexts is not None else None
        if extra_contexts is None:
            return base_contexts.to(dtype=dtype)
        merged = base_contexts.to(dtype=dtype) + extra_contexts.to(dtype=dtype)
        merge_norm = self.slot_guidance_norm if self.slot_guidance_norm is not None else self.slot_role_anchor_norm
        if merge_norm is not None:
            merged = merge_norm(merged)
        return merged

    def _build_video_memory(
        self,
        video_feats: torch.Tensor,
        vid_mask: Optional[torch.Tensor],
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.phrase_memory_mode == "temporal":
            memory = video_feats.to(dtype=dtype)
            if vid_mask is None:
                memory_key_padding_mask = torch.zeros(
                    memory.size(0),
                    memory.size(1),
                    dtype=torch.bool,
                    device=memory.device,
                )
            else:
                memory_key_padding_mask = ~vid_mask.bool()
            return memory, memory_key_padding_mask

        memory = self.caption_model.mean_pooling(video_feats, attention_mask=vid_mask).to(dtype=dtype)
        memory_key_padding_mask = torch.zeros(
            memory.size(0),
            memory.size(1),
            dtype=torch.bool,
            device=memory.device,
        )
        return memory, memory_key_padding_mask

    def _build_phrase_memory(
        self,
        video_feats: torch.Tensor,
        vid_mask: Optional[torch.Tensor],
        struct_context: Optional[torch.Tensor],
        struct_query_bridge_tokens: Optional[torch.Tensor],
        struct_query_bridge_mask: Optional[torch.Tensor],
        aux_visual_state: Optional[Dict[str, torch.Tensor]],
        progress_state: Optional[Dict[str, Any]],
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        video_memory, video_memory_key_padding_mask = self._build_video_memory(video_feats, vid_mask, dtype=dtype)
        memory_tokens = [video_memory]
        memory_masks = [video_memory_key_padding_mask]
        if struct_context is not None:
            memory_tokens.append(struct_context.unsqueeze(1).to(dtype=dtype))
            memory_masks.append(
                torch.zeros(
                    struct_context.size(0),
                    1,
                    dtype=torch.bool,
                    device=struct_context.device,
                )
            )
        if (
            struct_query_bridge_tokens is not None
            and self.struct_condition_query_bridge_memory_enable
            and self.struct_condition_query_bridge_memory_scale is not None
        ):
            memory_scale = torch.clamp(
                self.struct_condition_query_bridge_memory_scale,
                min=0.0,
                max=2.0,
            ).to(struct_query_bridge_tokens.dtype)
            bridge_tokens = memory_scale * struct_query_bridge_tokens.to(dtype=dtype)
            memory_tokens.append(bridge_tokens)
            if struct_query_bridge_mask is None:
                bridge_valid = torch.ones(
                    bridge_tokens.size(0),
                    bridge_tokens.size(1),
                    dtype=torch.bool,
                    device=bridge_tokens.device,
                )
            else:
                bridge_valid = struct_query_bridge_mask.bool()
            memory_masks.append(~bridge_valid)
        if (
            aux_visual_state is not None
            and self.aux_visual_memory_proj is not None
            and self.aux_visual_memory_scale > 0.0
        ):
            aux_tokens = None
            aux_mask = None
            if self.phrase_memory_mode == "temporal":
                aux_tokens = aux_visual_state.get("temporal")
                aux_mask = aux_visual_state.get("temporal_mask")
            if aux_tokens is None:
                aux_tokens = aux_visual_state.get("pooled_tokens")
                aux_mask = aux_visual_state.get("pooled_token_mask")
            if aux_tokens is not None:
                proj_dtype = self._module_dtype(self.aux_visual_memory_proj, aux_tokens.dtype)
                aux_tokens = self.aux_visual_memory_proj(aux_tokens.to(dtype=proj_dtype))
                aux_tokens = float(self.aux_visual_memory_scale) * aux_tokens
                memory_tokens.append(aux_tokens.to(dtype=dtype))
                if aux_mask is None:
                    aux_valid = torch.ones(
                        aux_tokens.size(0),
                        aux_tokens.size(1),
                        dtype=torch.bool,
                        device=aux_tokens.device,
                    )
                else:
                    aux_valid = aux_mask.bool()
                memory_masks.append(~aux_valid)
        if progress_state is not None and self.phrase_progress_memory_scale > 0.0:
            phase_tokens = progress_state.get("phase_tokens")
            phase_mask = progress_state.get("phase_mask")
            if phase_tokens is not None:
                progress_tokens = float(self.phrase_progress_memory_scale) * phase_tokens.to(dtype=dtype)
                memory_tokens.append(progress_tokens)
                if phase_mask is None:
                    phase_valid = torch.ones(
                        progress_tokens.size(0),
                        progress_tokens.size(1),
                        dtype=torch.bool,
                        device=progress_tokens.device,
                    )
                else:
                    phase_valid = phase_mask.bool()
                memory_masks.append(~phase_valid)
        memory = torch.cat(memory_tokens, dim=1)
        memory_key_padding_mask = torch.cat(memory_masks, dim=1).to(
            dtype=torch.bool,
            device=memory.device,
        )
        return memory, memory_key_padding_mask

    def _build_decoder_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        tgt_emb = self.caption_model.word_embeddings(input_ids)
        tgt_emb = self.caption_model.pos_embeddings(
            tgt_emb,
            position_ids=pos_ids,
            attention_mask=attention_mask,
        )
        tgt_emb = self.caption_model.norm_input(tgt_emb)
        return tgt_emb

    def _decode_phrase_tokens(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
        phrase_ids: torch.Tensor,
        phrase_mask: Optional[torch.Tensor],
        slot_indices: Optional[torch.Tensor] = None,
        slot_context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.phrase_decoder is None or self.phrase_output_norm is None:
            raise RuntimeError("Phrase decoder requested but not initialized.")

        input_ids = phrase_ids[:, :-1].contiguous()
        attn_mask = phrase_mask[:, :-1].contiguous() if phrase_mask is not None else input_ids.ne(self.pad_token_id)
        seq_len = input_ids.size(1)

        tgt_emb = self._build_decoder_inputs(input_ids, attn_mask)
        if slot_indices is not None:
            if self.slot_embeddings is None:
                raise RuntimeError("Slot-conditioned phrase decoding requested but slot embeddings are not initialized.")
            slot_emb = self.slot_embeddings(slot_indices.to(device=input_ids.device, dtype=torch.long)).to(tgt_emb.dtype)
            tgt_emb = tgt_emb + slot_emb.unsqueeze(1)
        if slot_context is not None:
            tgt_emb = tgt_emb + slot_context.to(dtype=tgt_emb.dtype).unsqueeze(1)
        tgt_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        tgt_key_padding_mask = (attn_mask == 0).bool() if attn_mask is not None else None

        hidden = self.phrase_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        hidden = self.phrase_output_norm(hidden)
        logits = self.caption_model.lm_head(hidden)

        token_mask = self._sequence_token_mask(input_ids, attn_mask)
        phrase_valid = token_mask.any(dim=1)
        phrase_summary = self._masked_mean(hidden, token_mask)
        phrase_summary = phrase_summary * phrase_valid.unsqueeze(-1).to(hidden.dtype)
        return logits, hidden, phrase_summary, token_mask, phrase_valid

    def _decode_phrase_slots(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
        phrase_slot_ids: torch.Tensor,
        phrase_slot_mask: Optional[torch.Tensor],
        slot_contexts: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if phrase_slot_ids.dim() != 3:
            raise ValueError("phrase_slot_ids must have shape [batch, slots, seq_len].")

        batch_size, slot_count, seq_len = phrase_slot_ids.shape
        flat_slot_ids = phrase_slot_ids.reshape(batch_size * slot_count, seq_len)
        flat_slot_mask = (
            phrase_slot_mask.reshape(batch_size * slot_count, seq_len)
            if phrase_slot_mask is not None
            else None
        )
        slot_indices = torch.arange(slot_count, device=phrase_slot_ids.device, dtype=torch.long)
        slot_indices = slot_indices.unsqueeze(0).expand(batch_size, slot_count).reshape(-1)
        slot_memory = memory.repeat_interleave(slot_count, dim=0)
        slot_memory_key_padding_mask = memory_key_padding_mask.repeat_interleave(slot_count, dim=0)
        flat_slot_context = None
        if slot_contexts is not None:
            if slot_contexts.dim() != 3 or slot_contexts.size(0) != batch_size or slot_contexts.size(1) != slot_count:
                raise ValueError("slot_contexts must have shape [batch, slots, hidden_dim].")
            flat_slot_context = slot_contexts.reshape(batch_size * slot_count, slot_contexts.size(-1))

        logits, hidden, _summary, token_mask, _phrase_valid = self._decode_phrase_tokens(
            memory=slot_memory,
            memory_key_padding_mask=slot_memory_key_padding_mask,
            phrase_ids=flat_slot_ids,
            phrase_mask=flat_slot_mask,
            slot_indices=slot_indices,
            slot_context=flat_slot_context,
        )

        hidden_dim = hidden.size(-1)
        vocab_size = logits.size(-1)
        logits = logits.reshape(batch_size, slot_count, seq_len - 1, vocab_size)
        hidden = hidden.reshape(batch_size, slot_count, seq_len - 1, hidden_dim)
        token_mask = token_mask.reshape(batch_size, slot_count, seq_len - 1)

        slot_valid = token_mask.any(dim=-1)
        slot_summary = self._masked_mean(
            hidden.reshape(batch_size * slot_count, seq_len - 1, hidden_dim),
            token_mask.reshape(batch_size * slot_count, seq_len - 1),
        ).reshape(batch_size, slot_count, hidden_dim)
        slot_summary = slot_summary * slot_valid.unsqueeze(-1).to(hidden.dtype)

        phrase_valid = slot_valid.any(dim=1)
        phrase_summary = self._masked_mean(slot_summary, slot_valid)
        phrase_summary = phrase_summary * phrase_valid.unsqueeze(-1).to(hidden.dtype)

        return {
            "phrase_slot_logits": logits,
            "phrase_slot_hidden": hidden,
            "phrase_slot_summary": slot_summary,
            "phrase_slot_token_mask": token_mask,
            "phrase_slot_valid": slot_valid,
            "phrase_decoder_logits": logits.reshape(batch_size, slot_count * (seq_len - 1), vocab_size),
            "phrase_decoder_hidden": hidden.reshape(batch_size, slot_count * (seq_len - 1), hidden_dim),
            "phrase_decoder_summary": phrase_summary,
            "phrase_decoder_token_mask": token_mask.reshape(batch_size, slot_count * (seq_len - 1)),
            "phrase_decoder_valid": phrase_valid,
        }

    def decode_phrase_slot_reference_bank(
        self,
        *,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
        phrase_slot_ref_ids: torch.Tensor,
        phrase_slot_ref_mask: Optional[torch.Tensor],
        slot_contexts: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if phrase_slot_ref_ids.dim() != 4:
            raise ValueError("phrase_slot_ref_ids must have shape [batch, slots, refs, seq_len].")

        batch_size, slot_count, ref_count, seq_len = phrase_slot_ref_ids.shape
        flat_slot_ref_ids = phrase_slot_ref_ids.permute(0, 2, 1, 3).reshape(batch_size * ref_count, slot_count, seq_len)
        flat_slot_ref_mask = None
        if phrase_slot_ref_mask is not None:
            flat_slot_ref_mask = phrase_slot_ref_mask.permute(0, 2, 1, 3).reshape(
                batch_size * ref_count,
                slot_count,
                seq_len,
            )
        flat_slot_contexts = None
        if slot_contexts is not None:
            if slot_contexts.dim() != 3 or slot_contexts.size(0) != batch_size or slot_contexts.size(1) != slot_count:
                raise ValueError("slot_contexts must have shape [batch, slots, hidden_dim].")
            flat_slot_contexts = slot_contexts.repeat_interleave(ref_count, dim=0)

        flat_outputs = self._decode_phrase_slots(
            memory=memory.repeat_interleave(ref_count, dim=0),
            memory_key_padding_mask=memory_key_padding_mask.repeat_interleave(ref_count, dim=0),
            phrase_slot_ids=flat_slot_ref_ids,
            phrase_slot_mask=flat_slot_ref_mask,
            slot_contexts=flat_slot_contexts,
        )
        vocab_size = flat_outputs["phrase_slot_logits"].size(-1)
        hidden_dim = flat_outputs["phrase_slot_hidden"].size(-1)
        seq_out_len = flat_outputs["phrase_slot_logits"].size(-2)
        return {
            "phrase_slot_logits": flat_outputs["phrase_slot_logits"].reshape(
                batch_size,
                ref_count,
                slot_count,
                seq_out_len,
                vocab_size,
            ).permute(0, 2, 1, 3, 4).contiguous(),
            "phrase_slot_hidden": flat_outputs["phrase_slot_hidden"].reshape(
                batch_size,
                ref_count,
                slot_count,
                seq_out_len,
                hidden_dim,
            ).permute(0, 2, 1, 3, 4).contiguous(),
            "phrase_slot_summary": flat_outputs["phrase_slot_summary"].reshape(
                batch_size,
                ref_count,
                slot_count,
                hidden_dim,
            ).permute(0, 2, 1, 3).contiguous(),
            "phrase_slot_token_mask": flat_outputs["phrase_slot_token_mask"].reshape(
                batch_size,
                ref_count,
                slot_count,
                seq_out_len,
            ).permute(0, 2, 1, 3).contiguous(),
            "phrase_slot_valid": flat_outputs["phrase_slot_valid"].reshape(
                batch_size,
                ref_count,
                slot_count,
            ).permute(0, 2, 1).contiguous(),
        }

    def summarize_phrase_slot_reference_bank(
        self,
        *,
        phrase_slot_ref_ids: torch.Tensor,
        phrase_slot_ref_mask: Optional[torch.Tensor],
        phrase_slot_ref_valid: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if phrase_slot_ref_ids.dim() != 4:
            raise ValueError("phrase_slot_ref_ids must have shape [batch, slots, refs, seq_len].")

        ref_token_mask = (
            phrase_slot_ref_mask.bool()
            if phrase_slot_ref_mask is not None
            else phrase_slot_ref_ids.ne(self.pad_token_id)
        )
        ref_token_mask = ref_token_mask & phrase_slot_ref_ids.ne(self.pad_token_id)
        ref_token_mask = ref_token_mask & phrase_slot_ref_ids.ne(self.bos_token_id)
        ref_token_mask = ref_token_mask & phrase_slot_ref_ids.ne(self.eos_token_id)

        ref_embeddings = self.caption_model.word_embeddings(phrase_slot_ref_ids)
        ref_token_mask_f = ref_token_mask.to(ref_embeddings.dtype).unsqueeze(-1)
        ref_token_den = ref_token_mask_f.sum(dim=-2).clamp_min(1.0)
        ref_summary = (ref_embeddings * ref_token_mask_f).sum(dim=-2) / ref_token_den

        if phrase_slot_ref_valid is not None:
            ref_valid = phrase_slot_ref_valid.bool() & ref_token_mask.any(dim=-1)
        else:
            ref_valid = ref_token_mask.any(dim=-1)
        ref_summary = ref_summary * ref_valid.unsqueeze(-1).to(ref_summary.dtype)

        slot_valid = ref_valid.any(dim=-1)
        slot_valid_f = ref_valid.to(ref_summary.dtype).unsqueeze(-1)
        slot_den = slot_valid_f.sum(dim=-2).clamp_min(1.0)
        slot_summary = (ref_summary * slot_valid_f).sum(dim=-2) / slot_den
        slot_summary = slot_summary * slot_valid.unsqueeze(-1).to(slot_summary.dtype)

        phrase_valid = slot_valid.any(dim=-1)
        phrase_summary = self._masked_mean(slot_summary, slot_valid)
        phrase_summary = phrase_summary * phrase_valid.unsqueeze(-1).to(phrase_summary.dtype)

        return {
            "phrase_slot_ref_summary": ref_summary,
            "phrase_slot_ref_valid": ref_valid,
            "phrase_ref_bank_slot_summary": slot_summary,
            "phrase_ref_bank_slot_valid": slot_valid,
            "phrase_ref_bank_phrase_summary": phrase_summary,
            "phrase_ref_bank_phrase_valid": phrase_valid,
        }

    def _build_phrase_condition(
        self,
        phrase_summary: Optional[torch.Tensor],
        phrase_valid: Optional[torch.Tensor],
        dtype: torch.dtype,
        phrase_slot_summary: Optional[torch.Tensor] = None,
        phrase_slot_valid: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if phrase_summary is None or self.phrase_to_context is None:
            return None

        cond = self.phrase_to_context(phrase_summary)
        if self.phrase_condition_slot_selective_enable and self.phrase_condition_core_proj is not None:
            core_summary, _core_valid = self._pool_phrase_slot_family_summary(
                phrase_slot_summary=phrase_slot_summary,
                phrase_slot_valid=phrase_slot_valid,
                selected_slot_types=self.phrase_condition_core_slot_types,
            )
            if core_summary is not None:
                cond = cond + self.phrase_condition_core_proj(core_summary.to(cond.dtype))
        if (
            self.phrase_condition_slot_aware_enable
            and not self.phrase_condition_slot_selective_enable
            and self.phrase_slot_condition_proj is not None
            and phrase_slot_summary is not None
            and phrase_slot_summary.dim() == 3
        ):
            batch_size, slot_count, hidden_dim = phrase_slot_summary.shape
            slot_summary = phrase_slot_summary
            if phrase_slot_valid is None:
                slot_valid = torch.ones(
                    (batch_size, slot_count),
                    dtype=torch.bool,
                    device=phrase_slot_summary.device,
                )
            else:
                slot_valid = phrase_slot_valid.bool()
            slot_summary = slot_summary * slot_valid.unsqueeze(-1).to(slot_summary.dtype)
            if slot_count < self.max_phrase_slots:
                pad_slots = self.max_phrase_slots - slot_count
                pad_summary = torch.zeros(
                    (batch_size, pad_slots, hidden_dim),
                    dtype=slot_summary.dtype,
                    device=slot_summary.device,
                )
                slot_summary = torch.cat([slot_summary, pad_summary], dim=1)
                pad_valid = torch.zeros((batch_size, pad_slots), dtype=torch.bool, device=slot_valid.device)
                slot_valid = torch.cat([slot_valid, pad_valid], dim=1)
            elif slot_count > self.max_phrase_slots:
                slot_summary = slot_summary[:, : self.max_phrase_slots, :]
                slot_valid = slot_valid[:, : self.max_phrase_slots]
            flat_slot_summary = slot_summary.reshape(batch_size, self.max_phrase_slots * hidden_dim)
            slot_cond = self.phrase_slot_condition_proj(flat_slot_summary)
            if self.phrase_slot_condition_presence_proj is not None:
                slot_cond = slot_cond + self.phrase_slot_condition_presence_proj(slot_valid.to(slot_cond.dtype))
            if self.phrase_slot_condition_norm is not None:
                slot_cond = self.phrase_slot_condition_norm(slot_cond)
            cond = cond + slot_cond.to(cond.dtype)
        gate = torch.sigmoid(self.phrase_condition_gate(cond))
        cond = self.condition_dropout(torch.tanh(cond) * gate)
        cond = self.phrase_condition_norm(cond)
        if phrase_valid is not None:
            cond = cond * phrase_valid.unsqueeze(-1).to(cond.dtype)
        return cond.to(dtype=dtype)

    def _compute_priors(
        self,
        video_feats: torch.Tensor,
        vid_mask: Optional[torch.Tensor],
        aux_raw_global_feats: Optional[torch.Tensor] = None,
        aux_raw_global_mask: Optional[torch.Tensor] = None,
        aux_patch_feats: Optional[torch.Tensor] = None,
        aux_patch_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        pooled_video = self._masked_mean(video_feats, vid_mask)
        aux_visual_state = self._encode_aux_visual_state(
            aux_raw_global_feats=aux_raw_global_feats,
            aux_raw_global_mask=aux_raw_global_mask,
            aux_patch_feats=aux_patch_feats,
            aux_patch_mask=aux_patch_mask,
            dtype=pooled_video.dtype,
        )
        aux_pooled = aux_visual_state.get("pooled") if aux_visual_state is not None else None
        if aux_pooled is not None and self.aux_visual_prior_proj is not None and self.aux_visual_prior_scale > 0.0:
            proj_dtype = self._module_dtype(self.aux_visual_prior_proj, aux_pooled.dtype)
            pooled_video = pooled_video + float(self.aux_visual_prior_scale) * self.aux_visual_prior_proj(
                aux_pooled.to(dtype=proj_dtype)
            ).to(dtype=pooled_video.dtype)
        entity_prior_logits = self._prior_head_logits(
            self.entity_prior_head,
            pooled_video=pooled_video,
            video_feats=video_feats,
            vid_mask=vid_mask,
        )
        action_prior_logits = self._prior_head_logits(
            self.action_prior_head,
            pooled_video=pooled_video,
            video_feats=video_feats,
            vid_mask=vid_mask,
        )
        attribute_prior_logits = self._prior_head_logits(
            self.attribute_prior_head,
            pooled_video=pooled_video,
            video_feats=video_feats,
            vid_mask=vid_mask,
        )
        scene_prior_logits = self._prior_head_logits(
            self.scene_prior_head,
            pooled_video=pooled_video,
            video_feats=video_feats,
            vid_mask=vid_mask,
        )
        return pooled_video, entity_prior_logits, action_prior_logits, attribute_prior_logits, scene_prior_logits

    def _generate_flat_phrase_state(
        self,
        video_feats: torch.Tensor,
        vid_mask: Optional[torch.Tensor],
        struct_context: Optional[torch.Tensor],
        struct_query_bridge_tokens: Optional[torch.Tensor] = None,
        struct_query_bridge_mask: Optional[torch.Tensor] = None,
        aux_visual_state: Optional[Dict[str, torch.Tensor]] = None,
        progress_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        if not self.phrase_decoder_enable:
            return {}

        device = video_feats.device
        batch_size = video_feats.size(0)
        memory_dtype = self.caption_model.word_embeddings.weight.dtype
        memory, memory_key_padding_mask = self._build_phrase_memory(
            video_feats=video_feats,
            vid_mask=vid_mask,
            struct_context=struct_context,
            struct_query_bridge_tokens=struct_query_bridge_tokens,
            struct_query_bridge_mask=struct_query_bridge_mask,
            aux_visual_state=aux_visual_state,
            progress_state=progress_state,
            dtype=memory_dtype,
        )

        seqs = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(self.phrase_gen_max_len):
            cur_len = seqs.size(1)
            phrase_ids = torch.full((batch_size, cur_len + 1), self.pad_token_id, dtype=torch.long, device=device)
            phrase_mask = torch.zeros((batch_size, cur_len + 1), dtype=torch.bool, device=device)
            phrase_ids[:, :cur_len] = seqs
            phrase_mask[:, :cur_len] = seqs.ne(self.pad_token_id)

            logits, _hidden, _summary, _token_mask, _phrase_valid = self._decode_phrase_tokens(
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                phrase_ids=phrase_ids,
                phrase_mask=phrase_mask,
            )
            next_ids = torch.argmax(logits[:, -1, :], dim=-1)

            early_eos = (cur_len == 1) & (next_ids == self.eos_token_id)
            if early_eos.any():
                logits_last = logits[:, -1, :].clone()
                logits_last[early_eos, self.eos_token_id] = torch.finfo(logits_last.dtype).min
                next_ids = torch.where(early_eos, torch.argmax(logits_last, dim=-1), next_ids)

            append_ids = torch.where(finished, torch.full_like(next_ids, self.pad_token_id), next_ids)
            seqs = torch.cat([seqs, append_ids.unsqueeze(-1)], dim=-1)
            finished = finished | (next_ids == self.eos_token_id)
            if bool(finished.all()):
                break

        unfinished = ~finished
        if unfinished.any():
            eos_pad = torch.where(
                unfinished,
                torch.full_like(finished.long(), self.eos_token_id),
                torch.full_like(finished.long(), self.pad_token_id),
            )
            seqs = torch.cat([seqs, eos_pad.unsqueeze(-1)], dim=-1)

        final_len = seqs.size(1)
        phrase_ids = torch.full((batch_size, final_len + 1), self.pad_token_id, dtype=torch.long, device=device)
        phrase_mask = torch.zeros((batch_size, final_len + 1), dtype=torch.bool, device=device)
        phrase_ids[:, :final_len] = seqs
        phrase_mask[:, :final_len] = seqs.ne(self.pad_token_id)

        phrase_logits, phrase_hidden, phrase_summary, token_mask, phrase_valid = self._decode_phrase_tokens(
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
            phrase_ids=phrase_ids,
            phrase_mask=phrase_mask,
        )
        return {
            "phrase_decoder_ids": phrase_ids,
            "phrase_decoder_mask": phrase_mask,
            "phrase_decoder_logits": phrase_logits,
            "phrase_decoder_hidden": phrase_hidden,
            "phrase_decoder_summary": phrase_summary,
            "phrase_decoder_token_mask": token_mask,
            "phrase_decoder_valid": phrase_valid,
        }

    def _generate_slot_phrase_state(
        self,
        video_feats: torch.Tensor,
        vid_mask: Optional[torch.Tensor],
        struct_context: Optional[torch.Tensor],
        struct_query_bridge_tokens: Optional[torch.Tensor] = None,
        struct_query_bridge_mask: Optional[torch.Tensor] = None,
        aux_visual_state: Optional[Dict[str, torch.Tensor]] = None,
        progress_state: Optional[Dict[str, Any]] = None,
        entity_prior_logits: Optional[torch.Tensor] = None,
        action_prior_logits: Optional[torch.Tensor] = None,
        attribute_prior_logits: Optional[torch.Tensor] = None,
        scene_prior_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if not self.phrase_decoder_enable:
            return {}
        if self.slot_embeddings is None:
            raise RuntimeError("Slot phrase generation requested but slot embeddings are not initialized.")

        device = video_feats.device
        batch_size = video_feats.size(0)
        slot_count = self.max_phrase_slots
        memory_dtype = self.caption_model.word_embeddings.weight.dtype
        memory, memory_key_padding_mask = self._build_phrase_memory(
            video_feats=video_feats,
            vid_mask=vid_mask,
            struct_context=struct_context,
            struct_query_bridge_tokens=struct_query_bridge_tokens,
            struct_query_bridge_mask=struct_query_bridge_mask,
            aux_visual_state=aux_visual_state,
            progress_state=progress_state,
            dtype=memory_dtype,
        )
        slot_source_bank, slot_source_names = self._collect_slot_source_bank(
            entity_prior_logits=entity_prior_logits,
            action_prior_logits=action_prior_logits,
            attribute_prior_logits=attribute_prior_logits,
            scene_prior_logits=scene_prior_logits,
            struct_context=struct_context,
            progress_state=progress_state,
            dtype=memory_dtype,
        )
        slot_contexts, planner_source_names, slot_source_weights = self._build_slot_planner_contexts(
            entity_prior_logits=entity_prior_logits,
            action_prior_logits=action_prior_logits,
            attribute_prior_logits=attribute_prior_logits,
            scene_prior_logits=scene_prior_logits,
            struct_context=struct_context,
            progress_state=progress_state,
            slot_count=slot_count,
            dtype=memory_dtype,
        )
        if planner_source_names:
            slot_source_names = planner_source_names
        slot_guidance_context = self._build_slot_guidance_context(
            slot_count=slot_count,
            source_bank=slot_source_bank,
            source_names=slot_source_names,
            slot_source_weights=slot_source_weights,
            entity_prior_logits=entity_prior_logits,
            action_prior_logits=action_prior_logits,
            attribute_prior_logits=attribute_prior_logits,
            scene_prior_logits=scene_prior_logits,
            device=device,
            dtype=memory_dtype,
        )
        slot_contexts = self._merge_slot_contexts(slot_contexts, slot_guidance_context, dtype=memory_dtype)
        slot_role_anchor_context = self._build_slot_role_anchor_context(
            slot_count=slot_count,
            entity_prior_logits=entity_prior_logits,
            action_prior_logits=action_prior_logits,
            attribute_prior_logits=attribute_prior_logits,
            scene_prior_logits=scene_prior_logits,
            device=device,
            dtype=memory_dtype,
        )
        slot_contexts = self._merge_slot_contexts(slot_contexts, slot_role_anchor_context, dtype=memory_dtype)
        active_slot_mask = self._build_active_slot_mask(
            slot_count=slot_count,
            device=device,
            dtype=memory_dtype,
        )
        if active_slot_mask is not None and slot_contexts is not None:
            slot_contexts = slot_contexts * active_slot_mask.view(1, slot_count, 1)
        slot_presence_outputs = self._predict_slot_presence(
            batch_size=batch_size,
            slot_count=slot_count,
            device=device,
            dtype=memory_dtype,
            struct_context=struct_context,
            slot_contexts=slot_contexts,
            entity_prior_logits=entity_prior_logits,
            action_prior_logits=action_prior_logits,
            attribute_prior_logits=attribute_prior_logits,
            scene_prior_logits=scene_prior_logits,
            source_names=slot_source_names,
            slot_source_weights=slot_source_weights,
        )
        slot_presence_pred = slot_presence_outputs.get("phrase_slot_presence_pred")
        if slot_presence_pred is None and active_slot_mask is not None:
            slot_presence_pred = active_slot_mask.bool().view(1, slot_count).expand(batch_size, -1)
            slot_presence_outputs["phrase_slot_presence_pred"] = slot_presence_pred
            slot_presence_outputs["phrase_slot_presence_probs"] = slot_presence_pred.to(dtype=memory_dtype)

        slot_phrase_ids = []
        slot_phrase_masks = []
        slot_anchor_token_ids = []
        slot_anchor_token_scores = []
        max_steps = max(1, self.phrase_slot_max_len - 1)

        for slot_idx in range(slot_count):
            seqs = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=device)
            if slot_presence_pred is not None:
                slot_active = slot_presence_pred[:, slot_idx].bool()
                finished = ~slot_active
            else:
                slot_active = torch.ones(batch_size, dtype=torch.bool, device=device)
                finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            slot_indices = torch.full((batch_size,), slot_idx, dtype=torch.long, device=device)
            slot_context = slot_contexts[:, slot_idx, :] if slot_contexts is not None else None
            slot_anchor_bias = self._build_slot_decode_anchor_bias(
                slot_idx=slot_idx,
                slot_count=slot_count,
                batch_size=batch_size,
                vocab_size=self.caption_model.lm_head.out_features,
                source_names=slot_source_names,
                slot_source_weights=slot_source_weights,
                entity_prior_logits=entity_prior_logits,
                action_prior_logits=action_prior_logits,
                attribute_prior_logits=attribute_prior_logits,
                scene_prior_logits=scene_prior_logits,
                device=device,
                dtype=memory_dtype,
            )
            anchor_ids, anchor_scores = self._summarize_slot_anchor_bias(slot_anchor_bias)
            slot_anchor_token_ids.append(anchor_ids)
            slot_anchor_token_scores.append(anchor_scores)

            for _ in range(max_steps):
                if bool(finished.all()):
                    break
                cur_len = seqs.size(1)
                phrase_ids = torch.full((batch_size, cur_len + 1), self.pad_token_id, dtype=torch.long, device=device)
                phrase_mask = torch.zeros((batch_size, cur_len + 1), dtype=torch.bool, device=device)
                phrase_ids[:, :cur_len] = seqs
                phrase_mask[:, :cur_len] = seqs.ne(self.pad_token_id)

                logits, _hidden, _summary, _token_mask, _phrase_valid = self._decode_phrase_tokens(
                    memory=memory,
                    memory_key_padding_mask=memory_key_padding_mask,
                    phrase_ids=phrase_ids,
                    phrase_mask=phrase_mask,
                    slot_indices=slot_indices,
                    slot_context=slot_context,
                )
                logits_last = self._apply_slot_decode_anchor_bias(
                    logits_last=logits[:, -1, :],
                    slot_anchor_bias=slot_anchor_bias,
                    cur_len=cur_len,
                )
                next_ids = torch.argmax(logits_last, dim=-1)

                if cur_len == 1:
                    if slot_presence_pred is not None:
                        early_eos = slot_active & (next_ids == self.eos_token_id)
                    else:
                        early_eos = (next_ids == self.eos_token_id) if slot_idx == 0 else torch.zeros_like(finished)
                    if early_eos.any():
                        logits_last = logits_last.clone()
                        logits_last[early_eos, self.eos_token_id] = torch.finfo(logits_last.dtype).min
                        next_ids = torch.where(early_eos, torch.argmax(logits_last, dim=-1), next_ids)

                append_ids = torch.where(finished, torch.full_like(next_ids, self.pad_token_id), next_ids)
                seqs = torch.cat([seqs, append_ids.unsqueeze(-1)], dim=-1)
                finished = finished | (next_ids == self.eos_token_id)
                if bool(finished.all()):
                    break

            unfinished = ~finished
            if unfinished.any():
                eos_pad = torch.where(
                    unfinished,
                    torch.full_like(finished.long(), self.eos_token_id),
                    torch.full_like(finished.long(), self.pad_token_id),
                )
                seqs = torch.cat([seqs, eos_pad.unsqueeze(-1)], dim=-1)

            final_len = seqs.size(1)
            phrase_ids = torch.full((batch_size, final_len + 1), self.pad_token_id, dtype=torch.long, device=device)
            phrase_mask = torch.zeros((batch_size, final_len + 1), dtype=torch.bool, device=device)
            phrase_ids[:, :final_len] = seqs
            phrase_mask[:, :final_len] = seqs.ne(self.pad_token_id)
            slot_phrase_ids.append(phrase_ids)
            slot_phrase_masks.append(phrase_mask)

        max_slot_seq_len = max(slot_ids.size(1) for slot_ids in slot_phrase_ids)
        phrase_slot_ids = torch.full(
            (batch_size, slot_count, max_slot_seq_len),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        phrase_slot_mask = torch.zeros((batch_size, slot_count, max_slot_seq_len), dtype=torch.bool, device=device)
        for slot_idx, (slot_ids, slot_mask) in enumerate(zip(slot_phrase_ids, slot_phrase_masks)):
            cur_len = slot_ids.size(1)
            phrase_slot_ids[:, slot_idx, :cur_len] = slot_ids
            phrase_slot_mask[:, slot_idx, :cur_len] = slot_mask

        slot_outputs = self._decode_phrase_slots(
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
            phrase_slot_ids=phrase_slot_ids,
            phrase_slot_mask=phrase_slot_mask,
            slot_contexts=slot_contexts,
        )
        slot_outputs["phrase_slot_ids"] = phrase_slot_ids
        slot_outputs["phrase_slot_mask"] = phrase_slot_mask
        slot_outputs["phrase_slot_source_names"] = list(slot_source_names)
        if slot_source_weights is not None:
            slot_outputs["phrase_slot_source_weights"] = slot_source_weights
        if slot_anchor_token_ids and all(item is not None for item in slot_anchor_token_ids):
            slot_outputs["phrase_slot_anchor_token_ids"] = torch.stack(slot_anchor_token_ids, dim=1)
        if slot_anchor_token_scores and all(item is not None for item in slot_anchor_token_scores):
            slot_outputs["phrase_slot_anchor_token_scores"] = torch.stack(slot_anchor_token_scores, dim=1)
        slot_outputs.update(slot_presence_outputs)
        return slot_outputs

    def generate_phrase_state(
        self,
        video_feats: torch.Tensor,
        vid_mask: Optional[torch.Tensor],
        struct_context: Optional[torch.Tensor],
        struct_query_bridge_tokens: Optional[torch.Tensor] = None,
        struct_query_bridge_mask: Optional[torch.Tensor] = None,
        aux_visual_state: Optional[Dict[str, torch.Tensor]] = None,
        progress_state: Optional[Dict[str, Any]] = None,
        entity_prior_logits: Optional[torch.Tensor] = None,
        action_prior_logits: Optional[torch.Tensor] = None,
        attribute_prior_logits: Optional[torch.Tensor] = None,
        scene_prior_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.phrase_target_mode == "slot":
            return self._generate_slot_phrase_state(
                video_feats,
                vid_mask,
                struct_context,
                struct_query_bridge_tokens=struct_query_bridge_tokens,
                struct_query_bridge_mask=struct_query_bridge_mask,
                aux_visual_state=aux_visual_state,
                progress_state=progress_state,
                entity_prior_logits=entity_prior_logits,
                action_prior_logits=action_prior_logits,
                attribute_prior_logits=attribute_prior_logits,
                scene_prior_logits=scene_prior_logits,
            )
        return self._generate_flat_phrase_state(
            video_feats,
            vid_mask,
            struct_context,
            struct_query_bridge_tokens=struct_query_bridge_tokens,
            struct_query_bridge_mask=struct_query_bridge_mask,
            aux_visual_state=aux_visual_state,
            progress_state=progress_state,
        )

    def prepare_generation_state(
        self,
        video_feats: torch.Tensor,
        vid_mask: Optional[torch.Tensor],
        aux_raw_global_feats: Optional[torch.Tensor] = None,
        aux_raw_global_mask: Optional[torch.Tensor] = None,
        aux_patch_feats: Optional[torch.Tensor] = None,
        aux_patch_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        _, entity_prior_logits, action_prior_logits, attribute_prior_logits, scene_prior_logits = self._compute_priors(
            video_feats,
            vid_mask,
            aux_raw_global_feats=aux_raw_global_feats,
            aux_raw_global_mask=aux_raw_global_mask,
            aux_patch_feats=aux_patch_feats,
            aux_patch_mask=aux_patch_mask,
        )
        aux_visual_state = self._encode_aux_visual_state(
            aux_raw_global_feats=aux_raw_global_feats,
            aux_raw_global_mask=aux_raw_global_mask,
            aux_patch_feats=aux_patch_feats,
            aux_patch_mask=aux_patch_mask,
            dtype=self.caption_model.word_embeddings.weight.dtype,
        )
        progress_state = self._encode_progress_state(
            video_feats=video_feats,
            vid_mask=vid_mask,
            dtype=self.caption_model.word_embeddings.weight.dtype,
        )

        cond_dtype = self.caption_model.word_embeddings.weight.dtype
        struct_condition_state = self._build_struct_condition_state(
            entity_prior_logits=entity_prior_logits,
            action_prior_logits=action_prior_logits,
            attribute_prior_logits=attribute_prior_logits,
            scene_prior_logits=scene_prior_logits,
            aux_visual_state=aux_visual_state,
            dtype=cond_dtype,
        )
        struct_context = struct_condition_state.get("struct_condition")
        struct_query_bridge_tokens = struct_condition_state.get("struct_query_bridge_tokens")
        struct_query_bridge_mask = struct_condition_state.get("struct_query_bridge_mask")

        state: Dict[str, Any] = {
            "entity_prior_logits": entity_prior_logits,
            "action_prior_logits": action_prior_logits,
            "attribute_prior_logits": attribute_prior_logits,
            "scene_prior_logits": scene_prior_logits,
            "struct_condition": struct_context,
        }
        if struct_query_bridge_tokens is not None:
            state["struct_query_bridge_tokens"] = struct_query_bridge_tokens
        if struct_query_bridge_mask is not None:
            state["struct_query_bridge_mask"] = struct_query_bridge_mask
        if self.phrase_decoder_enable:
            phrase_state = self.generate_phrase_state(
                video_feats=video_feats,
                vid_mask=vid_mask,
                struct_context=struct_context,
                struct_query_bridge_tokens=struct_query_bridge_tokens,
                struct_query_bridge_mask=struct_query_bridge_mask,
                aux_visual_state=aux_visual_state,
                progress_state=progress_state,
                entity_prior_logits=entity_prior_logits,
                action_prior_logits=action_prior_logits,
                attribute_prior_logits=attribute_prior_logits,
                scene_prior_logits=scene_prior_logits,
            )
            state.update(phrase_state)
            if self.phrase_condition_enable:
                state["phrase_condition"] = self._build_phrase_condition(
                    phrase_summary=phrase_state.get("phrase_decoder_summary"),
                    phrase_valid=phrase_state.get("phrase_decoder_valid"),
                    phrase_slot_summary=phrase_state.get("phrase_slot_summary"),
                    phrase_slot_valid=phrase_state.get("phrase_slot_valid"),
                    dtype=cond_dtype,
                )
        return state

    def expand_generation_state(self, generation_state: Dict[str, Any], batch_size: int) -> Dict[str, Any]:
        expanded: Dict[str, Any] = {}
        for key, value in generation_state.items():
            if torch.is_tensor(value) and value.dim() > 0 and value.size(0) == 1 and batch_size > 1:
                expanded[key] = value.expand((batch_size,) + tuple(value.shape[1:]))
            else:
                expanded[key] = value
        return expanded

    def select_generation_state(self, generation_state: Dict[str, Any], index: int) -> Dict[str, Any]:
        selected: Dict[str, Any] = {}
        for key, value in generation_state.items():
            if torch.is_tensor(value) and value.dim() > 0 and value.size(0) > index:
                selected[key] = value[index : index + 1]
            else:
                selected[key] = value
        return selected

    def forward(
        self,
        video_feats: torch.Tensor,
        vid_mask: torch.Tensor,
        captions: torch.Tensor,
        caption_mask: Optional[torch.Tensor] = None,
        *,
        phrase_ids: Optional[torch.Tensor] = None,
        phrase_mask: Optional[torch.Tensor] = None,
        phrase_slot_ids: Optional[torch.Tensor] = None,
        phrase_slot_mask: Optional[torch.Tensor] = None,
        phrase_slot_ref_ids: Optional[torch.Tensor] = None,
        phrase_slot_ref_mask: Optional[torch.Tensor] = None,
        phrase_slot_ref_valid: Optional[torch.Tensor] = None,
        aux_raw_global_feats: Optional[torch.Tensor] = None,
        aux_raw_global_mask: Optional[torch.Tensor] = None,
        aux_patch_feats: Optional[torch.Tensor] = None,
        aux_patch_mask: Optional[torch.Tensor] = None,
        generation_state: Optional[Dict[str, Any]] = None,
        return_aux: bool = False,
        return_hidden: bool = False,
    ):
        need_hidden = bool(return_hidden or return_aux or self.struct_condition or self.phrase_condition_enable)
        outputs = self.caption_model(
            video_feats,
            vid_mask,
            captions,
            caption_mask,
            return_hidden=need_hidden,
        )

        if need_hidden:
            base_logits, hidden = outputs
        else:
            base_logits = outputs
            hidden = None

        phrase_logits = None
        phrase_hidden = None
        phrase_summary = None
        phrase_token_mask = None
        phrase_valid = None
        phrase_slot_logits = None
        phrase_slot_hidden = None
        phrase_slot_summary = None
        phrase_slot_token_mask = None
        phrase_slot_valid = None
        phrase_slot_presence_logits = None
        phrase_slot_presence_probs = None
        phrase_slot_presence_raw_pred = None
        phrase_slot_presence_pred = None
        phrase_slot_presence_thresholds = None
        phrase_slot_presence_fallback_mask = None
        phrase_slot_presence_fallback_index = None
        phrase_slot_source_weights = None
        phrase_slot_source_names = None
        phrase_slot_anchor_token_ids = None
        phrase_slot_anchor_token_scores = None
        phrase_teacher_memory = None
        phrase_teacher_memory_key_padding_mask = None
        phrase_teacher_slot_contexts = None
        phrase_ref_bank_slot_summary = None
        phrase_ref_bank_slot_valid = None
        phrase_ref_bank_phrase_summary = None
        phrase_ref_bank_phrase_valid = None
        current_phrase_ids = phrase_ids
        current_phrase_slot_ids = phrase_slot_ids
        struct_context = None
        struct_query_bridge_tokens = None
        struct_query_bridge_mask = None
        pred_phrase_summary = None
        pred_phrase_valid = None
        pred_phrase_logits = None
        pred_phrase_ids = None
        pred_phrase_hidden = None
        pred_phrase_token_mask = None
        pred_phrase_slot_logits = None
        pred_phrase_slot_summary = None
        pred_phrase_slot_valid = None
        pred_phrase_slot_ids = None
        pred_phrase_slot_mask = None
        pred_phrase_slot_hidden = None
        pred_phrase_slot_token_mask = None
        aux_visual_state = None
        progress_state = None

        if generation_state is None:
            _, entity_prior_logits, action_prior_logits, attribute_prior_logits, scene_prior_logits = self._compute_priors(
                video_feats,
                vid_mask,
                aux_raw_global_feats=aux_raw_global_feats,
                aux_raw_global_mask=aux_raw_global_mask,
                aux_patch_feats=aux_patch_feats,
                aux_patch_mask=aux_patch_mask,
            )
            aux_visual_state = self._encode_aux_visual_state(
                aux_raw_global_feats=aux_raw_global_feats,
                aux_raw_global_mask=aux_raw_global_mask,
                aux_patch_feats=aux_patch_feats,
                aux_patch_mask=aux_patch_mask,
                dtype=self.caption_model.word_embeddings.weight.dtype,
            )
            progress_state = self._encode_progress_state(
                video_feats=video_feats,
                vid_mask=vid_mask,
                dtype=self.caption_model.word_embeddings.weight.dtype,
            )
            cond_dtype = hidden.dtype if hidden is not None else base_logits.dtype
            struct_condition_state = self._build_struct_condition_state(
                entity_prior_logits=entity_prior_logits,
                action_prior_logits=action_prior_logits,
                attribute_prior_logits=attribute_prior_logits,
                scene_prior_logits=scene_prior_logits,
                aux_visual_state=aux_visual_state,
                dtype=cond_dtype,
            )
            struct_context = struct_condition_state.get("struct_condition")
            struct_query_bridge_tokens = struct_condition_state.get("struct_query_bridge_tokens")
            struct_query_bridge_mask = struct_condition_state.get("struct_query_bridge_mask")
            phrase_condition = None
            if self.phrase_decoder_enable and phrase_ids is not None:
                memory, memory_key_padding_mask = self._build_phrase_memory(
                    video_feats=video_feats,
                    vid_mask=vid_mask,
                    struct_context=struct_context,
                    struct_query_bridge_tokens=struct_query_bridge_tokens,
                    struct_query_bridge_mask=struct_query_bridge_mask,
                    aux_visual_state=aux_visual_state,
                    progress_state=progress_state,
                    dtype=self.caption_model.word_embeddings.weight.dtype,
                )
                if self.phrase_target_mode == "slot" and phrase_slot_ids is not None:
                    slot_source_bank, phrase_slot_source_names = self._collect_slot_source_bank(
                        entity_prior_logits=entity_prior_logits,
                        action_prior_logits=action_prior_logits,
                        attribute_prior_logits=attribute_prior_logits,
                        scene_prior_logits=scene_prior_logits,
                        struct_context=struct_context,
                        progress_state=progress_state,
                        dtype=self.caption_model.word_embeddings.weight.dtype,
                    )
                    slot_contexts, planner_source_names, phrase_slot_source_weights = self._build_slot_planner_contexts(
                        entity_prior_logits=entity_prior_logits,
                        action_prior_logits=action_prior_logits,
                        attribute_prior_logits=attribute_prior_logits,
                        scene_prior_logits=scene_prior_logits,
                        struct_context=struct_context,
                        progress_state=progress_state,
                        slot_count=phrase_slot_ids.size(1),
                        dtype=self.caption_model.word_embeddings.weight.dtype,
                    )
                    if planner_source_names:
                        phrase_slot_source_names = planner_source_names
                    slot_guidance_context = self._build_slot_guidance_context(
                        slot_count=phrase_slot_ids.size(1),
                        source_bank=slot_source_bank,
                        source_names=phrase_slot_source_names,
                        slot_source_weights=phrase_slot_source_weights,
                        entity_prior_logits=entity_prior_logits,
                        action_prior_logits=action_prior_logits,
                        attribute_prior_logits=attribute_prior_logits,
                        scene_prior_logits=scene_prior_logits,
                        device=phrase_slot_ids.device,
                        dtype=self.caption_model.word_embeddings.weight.dtype,
                    )
                    slot_contexts = self._merge_slot_contexts(
                        slot_contexts,
                        slot_guidance_context,
                        dtype=self.caption_model.word_embeddings.weight.dtype,
                    )
                    slot_role_anchor_context = self._build_slot_role_anchor_context(
                        slot_count=phrase_slot_ids.size(1),
                        entity_prior_logits=entity_prior_logits,
                        action_prior_logits=action_prior_logits,
                        attribute_prior_logits=attribute_prior_logits,
                        scene_prior_logits=scene_prior_logits,
                        device=phrase_slot_ids.device,
                        dtype=self.caption_model.word_embeddings.weight.dtype,
                    )
                    slot_contexts = self._merge_slot_contexts(
                        slot_contexts,
                        slot_role_anchor_context,
                        dtype=self.caption_model.word_embeddings.weight.dtype,
                    )
                    active_slot_mask = self._build_active_slot_mask(
                        slot_count=phrase_slot_ids.size(1),
                        device=phrase_slot_ids.device,
                        dtype=self.caption_model.word_embeddings.weight.dtype,
                    )
                    if active_slot_mask is not None and slot_contexts is not None:
                        slot_contexts = slot_contexts * active_slot_mask.view(1, phrase_slot_ids.size(1), 1)
                    slot_presence_outputs = self._predict_slot_presence(
                        batch_size=phrase_slot_ids.size(0),
                        slot_count=phrase_slot_ids.size(1),
                        device=phrase_slot_ids.device,
                        dtype=self.caption_model.word_embeddings.weight.dtype,
                        struct_context=struct_context,
                        slot_contexts=slot_contexts,
                        entity_prior_logits=entity_prior_logits,
                        action_prior_logits=action_prior_logits,
                        attribute_prior_logits=attribute_prior_logits,
                        scene_prior_logits=scene_prior_logits,
                        source_names=phrase_slot_source_names,
                        slot_source_weights=phrase_slot_source_weights,
                    )
                    slot_outputs = self._decode_phrase_slots(
                        memory=memory,
                        memory_key_padding_mask=memory_key_padding_mask,
                        phrase_slot_ids=phrase_slot_ids,
                        phrase_slot_mask=phrase_slot_mask,
                        slot_contexts=slot_contexts,
                    )
                    phrase_teacher_memory = memory
                    phrase_teacher_memory_key_padding_mask = memory_key_padding_mask
                    phrase_teacher_slot_contexts = slot_contexts
                    phrase_logits = slot_outputs.get("phrase_decoder_logits")
                    phrase_hidden = slot_outputs.get("phrase_decoder_hidden")
                    phrase_summary = slot_outputs.get("phrase_decoder_summary")
                    phrase_token_mask = slot_outputs.get("phrase_decoder_token_mask")
                    phrase_valid = slot_outputs.get("phrase_decoder_valid")
                    phrase_slot_logits = slot_outputs.get("phrase_slot_logits")
                    phrase_slot_hidden = slot_outputs.get("phrase_slot_hidden")
                    phrase_slot_summary = slot_outputs.get("phrase_slot_summary")
                    phrase_slot_token_mask = slot_outputs.get("phrase_slot_token_mask")
                    phrase_slot_valid = slot_outputs.get("phrase_slot_valid")
                    phrase_slot_presence_logits = slot_presence_outputs.get("phrase_slot_presence_logits")
                    phrase_slot_presence_probs = slot_presence_outputs.get("phrase_slot_presence_probs")
                    phrase_slot_presence_raw_pred = slot_presence_outputs.get("phrase_slot_presence_raw_pred")
                    phrase_slot_presence_pred = slot_presence_outputs.get("phrase_slot_presence_pred")
                    phrase_slot_presence_thresholds = slot_presence_outputs.get("phrase_slot_presence_thresholds")
                    phrase_slot_presence_fallback_mask = slot_presence_outputs.get("phrase_slot_presence_fallback_mask")
                    phrase_slot_presence_fallback_index = slot_presence_outputs.get("phrase_slot_presence_fallback_index")
                    phrase_slot_anchor_token_ids = slot_outputs.get("phrase_slot_anchor_token_ids")
                    phrase_slot_anchor_token_scores = slot_outputs.get("phrase_slot_anchor_token_scores")
                    if (
                        phrase_slot_ref_ids is not None
                        and phrase_slot_ref_valid is not None
                        and bool(phrase_slot_ref_valid.any())
                    ):
                        phrase_ref_bank_state = self.summarize_phrase_slot_reference_bank(
                            phrase_slot_ref_ids=phrase_slot_ref_ids,
                            phrase_slot_ref_mask=phrase_slot_ref_mask,
                            phrase_slot_ref_valid=phrase_slot_ref_valid,
                        )
                        phrase_ref_bank_slot_summary = phrase_ref_bank_state.get("phrase_ref_bank_slot_summary")
                        phrase_ref_bank_slot_valid = phrase_ref_bank_state.get("phrase_ref_bank_slot_valid")
                        phrase_ref_bank_phrase_summary = phrase_ref_bank_state.get("phrase_ref_bank_phrase_summary")
                        phrase_ref_bank_phrase_valid = phrase_ref_bank_state.get("phrase_ref_bank_phrase_valid")
                else:
                    phrase_logits, phrase_hidden, phrase_summary, phrase_token_mask, phrase_valid = self._decode_phrase_tokens(
                        memory=memory,
                        memory_key_padding_mask=memory_key_padding_mask,
                        phrase_ids=phrase_ids,
                        phrase_mask=phrase_mask,
                    )
                if self.phrase_condition_enable and not self.phrase_condition_train_use_predicted:
                    condition_phrase_summary = phrase_summary
                    condition_phrase_valid = phrase_valid
                    condition_phrase_slot_summary = phrase_slot_summary
                    condition_phrase_slot_valid = phrase_slot_valid
                    if (
                        self.phrase_condition_teacher_source == "ref_bank"
                        and phrase_ref_bank_phrase_summary is not None
                        and phrase_ref_bank_phrase_valid is not None
                    ):
                        condition_phrase_summary = phrase_ref_bank_phrase_summary
                        condition_phrase_valid = phrase_ref_bank_phrase_valid
                        condition_phrase_slot_summary = phrase_ref_bank_slot_summary
                        condition_phrase_slot_valid = phrase_ref_bank_slot_valid
                    phrase_condition = self._build_phrase_condition(
                        phrase_summary=condition_phrase_summary,
                        phrase_valid=condition_phrase_valid,
                        phrase_slot_summary=condition_phrase_slot_summary,
                        phrase_slot_valid=condition_phrase_slot_valid,
                        dtype=cond_dtype,
                    )
            if self.phrase_condition_enable and self.phrase_condition_train_use_predicted:
                predicted_phrase_state = self.generate_phrase_state(
                    video_feats=video_feats,
                    vid_mask=vid_mask,
                    struct_context=struct_context,
                    struct_query_bridge_tokens=struct_query_bridge_tokens,
                    struct_query_bridge_mask=struct_query_bridge_mask,
                    aux_visual_state=aux_visual_state,
                    progress_state=progress_state,
                    entity_prior_logits=entity_prior_logits,
                    action_prior_logits=action_prior_logits,
                    attribute_prior_logits=attribute_prior_logits,
                    scene_prior_logits=scene_prior_logits,
                )
                pred_phrase_logits = predicted_phrase_state.get("phrase_decoder_logits")
                pred_phrase_summary = predicted_phrase_state.get("phrase_decoder_summary")
                pred_phrase_valid = predicted_phrase_state.get("phrase_decoder_valid")
                pred_phrase_ids = predicted_phrase_state.get("phrase_decoder_ids")
                pred_phrase_hidden = predicted_phrase_state.get("phrase_decoder_hidden")
                pred_phrase_token_mask = predicted_phrase_state.get("phrase_decoder_token_mask")
                pred_phrase_slot_logits = predicted_phrase_state.get("phrase_slot_logits")
                pred_phrase_slot_summary = predicted_phrase_state.get("phrase_slot_summary")
                pred_phrase_slot_valid = predicted_phrase_state.get("phrase_slot_valid")
                pred_phrase_slot_ids = predicted_phrase_state.get("phrase_slot_ids")
                pred_phrase_slot_mask = predicted_phrase_state.get("phrase_slot_mask")
                pred_phrase_slot_hidden = predicted_phrase_state.get("phrase_slot_hidden")
                pred_phrase_slot_token_mask = predicted_phrase_state.get("phrase_slot_token_mask")
                phrase_condition = self._build_phrase_condition(
                    phrase_summary=pred_phrase_summary,
                    phrase_valid=pred_phrase_valid,
                    phrase_slot_summary=pred_phrase_slot_summary,
                    phrase_slot_valid=pred_phrase_slot_valid,
                    dtype=cond_dtype,
                )
                if phrase_condition is not None and self.phrase_condition_pred_detach:
                    phrase_condition = phrase_condition.detach()
        else:
            entity_prior_logits = generation_state.get("entity_prior_logits")
            action_prior_logits = generation_state.get("action_prior_logits")
            attribute_prior_logits = generation_state.get("attribute_prior_logits")
            scene_prior_logits = generation_state.get("scene_prior_logits")
            struct_context = generation_state.get("struct_condition")
            struct_query_bridge_tokens = generation_state.get("struct_query_bridge_tokens")
            struct_query_bridge_mask = generation_state.get("struct_query_bridge_mask")
            phrase_condition = generation_state.get("phrase_condition")
            current_phrase_ids = generation_state.get("phrase_decoder_ids")
            phrase_logits = generation_state.get("phrase_decoder_logits")
            phrase_hidden = generation_state.get("phrase_decoder_hidden")
            phrase_summary = generation_state.get("phrase_decoder_summary")
            phrase_token_mask = generation_state.get("phrase_decoder_token_mask")
            phrase_valid = generation_state.get("phrase_decoder_valid")
            current_phrase_slot_ids = generation_state.get("phrase_slot_ids")
            phrase_slot_logits = generation_state.get("phrase_slot_logits")
            phrase_slot_hidden = generation_state.get("phrase_slot_hidden")
            phrase_slot_summary = generation_state.get("phrase_slot_summary")
            phrase_slot_token_mask = generation_state.get("phrase_slot_token_mask")
            phrase_slot_valid = generation_state.get("phrase_slot_valid")
            phrase_slot_presence_logits = generation_state.get("phrase_slot_presence_logits")
            phrase_slot_presence_probs = generation_state.get("phrase_slot_presence_probs")
            phrase_slot_presence_raw_pred = generation_state.get("phrase_slot_presence_raw_pred")
            phrase_slot_presence_pred = generation_state.get("phrase_slot_presence_pred")
            phrase_slot_presence_thresholds = generation_state.get("phrase_slot_presence_thresholds")
            phrase_slot_presence_fallback_mask = generation_state.get("phrase_slot_presence_fallback_mask")
            phrase_slot_presence_fallback_index = generation_state.get("phrase_slot_presence_fallback_index")
            phrase_slot_source_weights = generation_state.get("phrase_slot_source_weights")
            phrase_slot_source_names = generation_state.get("phrase_slot_source_names")
            phrase_slot_anchor_token_ids = generation_state.get("phrase_slot_anchor_token_ids")
            phrase_slot_anchor_token_scores = generation_state.get("phrase_slot_anchor_token_scores")

        logits = base_logits
        if hidden is not None:
            condition_phrase_ids = current_phrase_ids
            condition_phrase_hidden = phrase_hidden
            condition_phrase_token_mask = phrase_token_mask
            condition_phrase_slot_ids = current_phrase_slot_ids
            condition_phrase_slot_hidden = phrase_slot_hidden
            condition_phrase_slot_token_mask = phrase_slot_token_mask
            condition_phrase_slot_summary = phrase_slot_summary
            condition_phrase_slot_valid = phrase_slot_valid
            if generation_state is None and self.phrase_condition_train_use_predicted:
                if pred_phrase_ids is not None:
                    condition_phrase_ids = pred_phrase_ids
                if pred_phrase_hidden is not None:
                    condition_phrase_hidden = (
                        pred_phrase_hidden.detach() if self.phrase_condition_pred_detach else pred_phrase_hidden
                    )
                if pred_phrase_token_mask is not None:
                    condition_phrase_token_mask = pred_phrase_token_mask
                if pred_phrase_slot_ids is not None:
                    condition_phrase_slot_ids = pred_phrase_slot_ids
                if pred_phrase_slot_hidden is not None:
                    condition_phrase_slot_hidden = (
                        pred_phrase_slot_hidden.detach()
                        if self.phrase_condition_pred_detach
                        else pred_phrase_slot_hidden
                    )
                if pred_phrase_slot_token_mask is not None:
                    condition_phrase_slot_token_mask = pred_phrase_slot_token_mask
                if pred_phrase_slot_summary is not None:
                    condition_phrase_slot_summary = (
                        pred_phrase_slot_summary.detach()
                        if self.phrase_condition_pred_detach
                        else pred_phrase_slot_summary
                    )
                if pred_phrase_slot_valid is not None:
                    condition_phrase_slot_valid = pred_phrase_slot_valid
            condition_applied = False
            if self.struct_condition and struct_context is not None:
                scale = torch.clamp(self.struct_scale, min=0.0, max=2.0).to(hidden.dtype)
                hidden = hidden + scale * struct_context.unsqueeze(1).to(hidden.dtype)
                condition_applied = True
            if (
                self.struct_condition_query_bridge_hidden_enable
                and self.struct_condition_query_bridge_hidden_scale is not None
            ):
                struct_query_bridge_context = self._build_struct_condition_query_bridge_context(
                    hidden=hidden,
                    struct_query_bridge_tokens=struct_query_bridge_tokens,
                    struct_query_bridge_mask=struct_query_bridge_mask,
                )
                if struct_query_bridge_context is not None:
                    struct_query_hidden_scale = torch.clamp(
                        self.struct_condition_query_bridge_hidden_scale,
                        min=0.0,
                        max=2.0,
                    ).to(hidden.dtype)
                    hidden = hidden + struct_query_hidden_scale * struct_query_bridge_context.to(hidden.dtype)
                    condition_applied = True
            if self.phrase_condition_enable and phrase_condition is not None and self.phrase_scale is not None:
                scale = torch.clamp(self.phrase_scale, min=0.0, max=2.0).to(hidden.dtype)
                hidden = hidden + scale * phrase_condition.unsqueeze(1).to(hidden.dtype)
                condition_applied = True
            if self.phrase_condition_family_bridge_enable and self.phrase_condition_family_scale is not None:
                family_context = self._build_phrase_condition_family_context(
                    hidden=hidden,
                    phrase_slot_summary=phrase_slot_summary,
                    phrase_slot_valid=phrase_slot_valid,
                )
                if family_context is not None:
                    family_scale = torch.clamp(self.phrase_condition_family_scale, min=0.0, max=2.0).to(hidden.dtype)
                    hidden = hidden + family_scale * family_context.to(hidden.dtype)
                    condition_applied = True
            if self.phrase_condition_slot_selective_enable and self.phrase_condition_aux_scale is not None:
                aux_context = self._build_phrase_condition_aux_context(
                    hidden=hidden,
                    phrase_slot_summary=phrase_slot_summary,
                    phrase_slot_valid=phrase_slot_valid,
                )
                if aux_context is not None:
                    aux_scale = torch.clamp(self.phrase_condition_aux_scale, min=0.0, max=2.0).to(hidden.dtype)
                    hidden = hidden + aux_scale * aux_context.to(hidden.dtype)
                    condition_applied = True
            if self.phrase_condition_slot_residual_enable and self.phrase_condition_slot_residual_scale is not None:
                slot_residual_context = self._build_phrase_condition_slot_residual_context(
                    hidden=hidden,
                    phrase_slot_summary=condition_phrase_slot_summary,
                    phrase_slot_valid=condition_phrase_slot_valid,
                )
                if slot_residual_context is not None:
                    slot_residual_scale = torch.clamp(
                        self.phrase_condition_slot_residual_scale,
                        min=0.0,
                        max=2.0,
                    ).to(hidden.dtype)
                    hidden = hidden + slot_residual_scale * slot_residual_context.to(hidden.dtype)
                    condition_applied = True
            if self.phrase_condition_query_bridge_enable and self.phrase_condition_query_bridge_scale is not None:
                query_bridge_context = self._build_phrase_condition_query_bridge_context(
                    hidden=hidden,
                    phrase_hidden=condition_phrase_hidden,
                    phrase_token_mask=condition_phrase_token_mask,
                )
                if query_bridge_context is not None:
                    query_bridge_scale = torch.clamp(
                        self.phrase_condition_query_bridge_scale,
                        min=0.0,
                        max=2.0,
                    ).to(hidden.dtype)
                    hidden = hidden + query_bridge_scale * query_bridge_context.to(hidden.dtype)
                    condition_applied = True
            if condition_applied:
                hidden = self.condition_post_norm(hidden)
                logits = self.caption_model.lm_head(hidden)
            if self.phrase_condition_candidate_bias_enable:
                logits = self._apply_phrase_candidate_bias(
                    logits=logits,
                    hidden=hidden,
                    phrase_slot_ids=condition_phrase_slot_ids,
                    phrase_slot_hidden=condition_phrase_slot_hidden,
                    phrase_slot_token_mask=condition_phrase_slot_token_mask,
                    phrase_slot_valid=condition_phrase_slot_valid,
                    phrase_ids=condition_phrase_ids,
                    phrase_hidden=condition_phrase_hidden,
                    phrase_token_mask=condition_phrase_token_mask,
                )

        if not return_aux:
            if return_hidden:
                return logits, hidden
            return logits

        aux: Dict[str, Any] = {}
        if entity_prior_logits is not None:
            aux["entity_prior_logits"] = entity_prior_logits
        if action_prior_logits is not None:
            aux["action_prior_logits"] = action_prior_logits
        if attribute_prior_logits is not None:
            aux["attribute_prior_logits"] = attribute_prior_logits
        if scene_prior_logits is not None:
            aux["scene_prior_logits"] = scene_prior_logits
        if struct_context is not None:
            aux["struct_condition"] = struct_context
        if struct_query_bridge_tokens is not None:
            aux["struct_query_bridge_tokens"] = struct_query_bridge_tokens
        if struct_query_bridge_mask is not None:
            aux["struct_query_bridge_mask"] = struct_query_bridge_mask
        if phrase_logits is not None:
            aux["phrase_decoder_logits"] = phrase_logits
        if phrase_hidden is not None:
            aux["phrase_decoder_hidden"] = phrase_hidden
        if phrase_summary is not None:
            aux["phrase_decoder_summary"] = phrase_summary
        if phrase_token_mask is not None:
            aux["phrase_decoder_token_mask"] = phrase_token_mask
        if phrase_valid is not None:
            aux["phrase_decoder_valid"] = phrase_valid
        if phrase_slot_logits is not None:
            aux["phrase_slot_logits"] = phrase_slot_logits
        if phrase_slot_hidden is not None:
            aux["phrase_slot_hidden"] = phrase_slot_hidden
        if phrase_slot_summary is not None:
            aux["phrase_slot_summary"] = phrase_slot_summary
        if phrase_slot_token_mask is not None:
            aux["phrase_slot_token_mask"] = phrase_slot_token_mask
        if phrase_slot_valid is not None:
            aux["phrase_slot_valid"] = phrase_slot_valid
        if phrase_slot_presence_logits is not None:
            aux["phrase_slot_presence_logits"] = phrase_slot_presence_logits
        if phrase_slot_presence_probs is not None:
            aux["phrase_slot_presence_probs"] = phrase_slot_presence_probs
        if phrase_slot_presence_raw_pred is not None:
            aux["phrase_slot_presence_raw_pred"] = phrase_slot_presence_raw_pred
        if phrase_slot_presence_pred is not None:
            aux["phrase_slot_presence_pred"] = phrase_slot_presence_pred
        if phrase_slot_presence_thresholds is not None:
            aux["phrase_slot_presence_thresholds"] = phrase_slot_presence_thresholds
        if phrase_slot_presence_fallback_mask is not None:
            aux["phrase_slot_presence_fallback_mask"] = phrase_slot_presence_fallback_mask
        if phrase_slot_presence_fallback_index is not None:
            aux["phrase_slot_presence_fallback_index"] = phrase_slot_presence_fallback_index
        if phrase_slot_source_names:
            aux["phrase_slot_source_names"] = list(phrase_slot_source_names)
        if phrase_slot_source_weights is not None:
            aux["phrase_slot_source_weights"] = phrase_slot_source_weights
        if phrase_slot_anchor_token_ids is not None:
            aux["phrase_slot_anchor_token_ids"] = phrase_slot_anchor_token_ids
        if phrase_slot_anchor_token_scores is not None:
            aux["phrase_slot_anchor_token_scores"] = phrase_slot_anchor_token_scores
        if phrase_teacher_memory is not None:
            aux["phrase_teacher_memory"] = phrase_teacher_memory
        if phrase_teacher_memory_key_padding_mask is not None:
            aux["phrase_teacher_memory_key_padding_mask"] = phrase_teacher_memory_key_padding_mask
        if phrase_teacher_slot_contexts is not None:
            aux["phrase_teacher_slot_contexts"] = phrase_teacher_slot_contexts
        if phrase_ref_bank_slot_summary is not None:
            aux["phrase_ref_bank_slot_summary"] = phrase_ref_bank_slot_summary
        if phrase_ref_bank_slot_valid is not None:
            aux["phrase_ref_bank_slot_valid"] = phrase_ref_bank_slot_valid
        if phrase_ref_bank_phrase_summary is not None:
            aux["phrase_ref_bank_phrase_summary"] = phrase_ref_bank_phrase_summary
        if phrase_ref_bank_phrase_valid is not None:
            aux["phrase_ref_bank_phrase_valid"] = phrase_ref_bank_phrase_valid
        aux["phrase_condition_teacher_source"] = self.phrase_condition_teacher_source
        if pred_phrase_summary is not None:
            aux["pred_phrase_decoder_summary"] = pred_phrase_summary
        if pred_phrase_valid is not None:
            aux["pred_phrase_decoder_valid"] = pred_phrase_valid
        if pred_phrase_logits is not None:
            aux["pred_phrase_decoder_logits"] = pred_phrase_logits
        if pred_phrase_ids is not None:
            aux["pred_phrase_decoder_ids"] = pred_phrase_ids
        if pred_phrase_slot_summary is not None:
            aux["pred_phrase_slot_summary"] = pred_phrase_slot_summary
        if pred_phrase_slot_valid is not None:
            aux["pred_phrase_slot_valid"] = pred_phrase_slot_valid
        if pred_phrase_slot_logits is not None:
            aux["pred_phrase_slot_logits"] = pred_phrase_slot_logits
        if pred_phrase_slot_ids is not None:
            aux["pred_phrase_slot_ids"] = pred_phrase_slot_ids
        if pred_phrase_slot_mask is not None:
            aux["pred_phrase_slot_mask"] = pred_phrase_slot_mask
        if generation_state is not None:
            for key in (
                "phrase_decoder_ids",
                "phrase_decoder_mask",
                "phrase_slot_ids",
                "phrase_slot_mask",
                "phrase_slot_source_names",
                "phrase_slot_source_weights",
                "phrase_slot_anchor_token_ids",
                "phrase_slot_anchor_token_scores",
                "phrase_slot_presence_raw_pred",
                "phrase_slot_presence_fallback_mask",
                "phrase_slot_presence_fallback_index",
                "phrase_slot_presence_thresholds",
                "phrase_condition",
                "struct_query_bridge_tokens",
                "struct_query_bridge_mask",
            ):
                if key in generation_state and generation_state[key] is not None:
                    aux[key] = generation_state[key]

        if return_hidden:
            return logits, hidden, aux
        return logits, aux
