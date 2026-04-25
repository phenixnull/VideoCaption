# -*- coding: utf-8 -*-
"""Structured-caption dataset wrapper.

Wraps the baseline caption dataset and appends:
- video-level entity/action/attribute/scene multi-hot labels
- caption-level anchor-local entity/action/attribute/scene multi-hot labels
- known masks for optional branches (attribute/scene)
- caption-level phrase-unit token ids for alignment loss
"""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class StructuredCaptionDataset(Dataset):
    _TYPED_SLOT_SPECS = (
        {
            "slot_type_id": 0,
            "slot_type": "subject_action",
            "slot_description": "subject with active predicate focus",
        },
        {
            "slot_type_id": 1,
            "slot_type": "object_passive",
            "slot_description": "object with passive predicate focus",
        },
        {
            "slot_type_id": 2,
            "slot_type": "relation_scene",
            "slot_description": "relation or scene phrase focus",
        },
        {
            "slot_type_id": 3,
            "slot_type": "attribute_misc",
            "slot_description": "attribute or residual phrase focus",
        },
    )
    _TYPED_SLOT_FAMILY_ORDER = tuple(spec["slot_type"] for spec in _TYPED_SLOT_SPECS)
    _TYPED_RICH_SLOT_SPECS = (
        {
            "slot_type_id": 0,
            "slot_type": "subject_action",
            "slot_description": "subject with active predicate focus",
        },
        {
            "slot_type_id": 1,
            "slot_type": "object_passive",
            "slot_description": "object with passive predicate focus",
        },
        {
            "slot_type_id": 2,
            "slot_type": "subject_entity",
            "slot_description": "subject-side entity phrase focus",
        },
        {
            "slot_type_id": 3,
            "slot_type": "object_entity",
            "slot_description": "object-side entity phrase focus",
        },
        {
            "slot_type_id": 4,
            "slot_type": "relation_detail",
            "slot_description": "relation, instrument, or modifier detail focus",
        },
        {
            "slot_type_id": 5,
            "slot_type": "scene_context",
            "slot_description": "scene or background phrase focus",
        },
    )
    _TYPED_RICH_SLOT_FAMILY_ORDER = tuple(spec["slot_type"] for spec in _TYPED_RICH_SLOT_SPECS)
    _TYPED_RICH_SEMANTIC_SLOT_SPECS = (
        {
            "slot_type_id": 0,
            "slot_type": "subject_action",
            "slot_description": "subject with active predicate focus",
        },
        {
            "slot_type_id": 1,
            "slot_type": "object_passive",
            "slot_description": "object with passive predicate focus",
        },
        {
            "slot_type_id": 2,
            "slot_type": "subject_entity",
            "slot_description": "subject-side entity phrase focus",
        },
        {
            "slot_type_id": 3,
            "slot_type": "object_entity",
            "slot_description": "object-side entity phrase focus",
        },
        {
            "slot_type_id": 4,
            "slot_type": "relation_detail",
            "slot_description": "interaction or relational detail focus",
        },
        {
            "slot_type_id": 5,
            "slot_type": "instrument_detail",
            "slot_description": "tool, carrier, or instrument detail focus",
        },
        {
            "slot_type_id": 6,
            "slot_type": "entity_modifier",
            "slot_description": "entity-bound modifier phrase focus",
        },
        {
            "slot_type_id": 7,
            "slot_type": "scene_context",
            "slot_description": "scene or background phrase focus",
        },
    )
    _TYPED_RICH_SEMANTIC_SLOT_FAMILY_ORDER = tuple(spec["slot_type"] for spec in _TYPED_RICH_SEMANTIC_SLOT_SPECS)
    _TYPED_RICH_ROLEAWARE_SLOT_SPECS = (
        {
            "slot_type_id": 0,
            "slot_type": "subject_action",
            "slot_description": "subject with active predicate focus",
        },
        {
            "slot_type_id": 1,
            "slot_type": "object_passive",
            "slot_description": "object with passive predicate focus",
        },
        {
            "slot_type_id": 2,
            "slot_type": "subject_entity",
            "slot_description": "subject-side entity core phrase focus",
        },
        {
            "slot_type_id": 3,
            "slot_type": "object_entity",
            "slot_description": "object-side entity core phrase focus",
        },
        {
            "slot_type_id": 4,
            "slot_type": "subject_modifier",
            "slot_description": "subject-side modified noun phrase focus",
        },
        {
            "slot_type_id": 5,
            "slot_type": "object_modifier",
            "slot_description": "object-side modified noun phrase focus",
        },
        {
            "slot_type_id": 6,
            "slot_type": "relation_detail",
            "slot_description": "interaction or relational detail focus",
        },
        {
            "slot_type_id": 7,
            "slot_type": "instrument_detail",
            "slot_description": "tool, carrier, or instrument detail focus",
        },
        {
            "slot_type_id": 8,
            "slot_type": "scene_context",
            "slot_description": "scene or background phrase focus",
        },
    )
    _TYPED_RICH_ROLEAWARE_SLOT_FAMILY_ORDER = tuple(spec["slot_type"] for spec in _TYPED_RICH_ROLEAWARE_SLOT_SPECS)
    _TYPED_RICH_ROLEAWARE_EXTRA_SLOT_PRIORITY = (
        "subject_modifier",
        "object_modifier",
        "relation_detail",
        "scene_context",
        "instrument_detail",
        "subject_entity",
        "object_entity",
        "subject_action",
        "object_passive",
    )
    _TYPED_RICH_ROLEAWARE_SCENEPLUS_EXTRA_SLOT_PRIORITY = (
        "scene_context",
        "scene_context",
        "relation_detail",
        "instrument_detail",
        "subject_modifier",
        "object_modifier",
        "subject_entity",
        "object_entity",
        "subject_action",
        "object_passive",
    )
    _FAMILY4_COMPACT_SLOT_SPECS = (
        {
            "slot_type_id": 0,
            "slot_type": "subject_entity",
            "slot_description": "primary subject-family phrase focus",
        },
        {
            "slot_type_id": 1,
            "slot_type": "object_entity",
            "slot_description": "primary object-family phrase focus",
        },
        {
            "slot_type_id": 2,
            "slot_type": "subject_action",
            "slot_description": "primary action-family phrase focus",
        },
        {
            "slot_type_id": 3,
            "slot_type": "scene_context",
            "slot_description": "primary scene-family phrase focus",
        },
    )
    _FAMILY4_COMPACT_SLOT_FAMILY_ORDER = tuple(spec["slot_type"] for spec in _FAMILY4_COMPACT_SLOT_SPECS)
    _ANCHORED_SOV_SCENE_SLOT_SPECS = (
        {
            "slot_type_id": 0,
            "slot_type": "subject_action",
            "slot_description": "subject-bound action phrase focus",
        },
        {
            "slot_type_id": 1,
            "slot_type": "object_entity",
            "slot_description": "object noun phrase focus",
        },
        {
            "slot_type_id": 2,
            "slot_type": "scene_context",
            "slot_description": "scene or supplemental context phrase focus",
        },
        {
            "slot_type_id": 3,
            "slot_type": "relation_detail",
            "slot_description": "action-object or relation detail focus",
        },
        {
            "slot_type_id": 4,
            "slot_type": "subject_entity",
            "slot_description": "subject noun phrase focus",
        },
    )
    _ANCHORED_SOV_SCENE_SLOT_FAMILY_ORDER = tuple(spec["slot_type"] for spec in _ANCHORED_SOV_SCENE_SLOT_SPECS)
    _COPULA_TOKENS = {"is", "are", "was", "were"}
    _RELATION_HEAD_TOKENS = {
        "in",
        "on",
        "at",
        "near",
        "with",
        "into",
        "from",
        "inside",
        "outside",
        "under",
        "over",
        "behind",
        "beside",
        "by",
        "around",
        "across",
        "through",
        "towards",
        "along",
        "alongside",
        "amid",
        "amidst",
        "between",
        "before",
        "after",
        "during",
        "while",
    }
    _INSTRUMENT_RELATION_HEAD_TOKENS = {"with", "using", "via", "by"}
    _INSTRUMENT_NOUN_HINTS = {
        "bat",
        "bike",
        "bicycle",
        "board",
        "boat",
        "bowl",
        "broom",
        "brush",
        "camera",
        "car",
        "cup",
        "fork",
        "glass",
        "glove",
        "guitar",
        "gun",
        "hammer",
        "hand",
        "hands",
        "helmet",
        "hose",
        "instrument",
        "knife",
        "ladder",
        "machine",
        "microphone",
        "motorcycle",
        "pan",
        "phone",
        "pick",
        "plate",
        "pot",
        "racket",
        "rope",
        "shovel",
        "sieve",
        "skateboard",
        "spatula",
        "spoon",
        "stick",
        "sword",
        "tool",
        "tray",
        "vehicle",
        "wheel",
    }
    _SCENE_RELATION_HEAD_TOKENS = {
        "in",
        "on",
        "at",
        "near",
        "inside",
        "outside",
        "under",
        "over",
        "behind",
        "beside",
        "by",
        "around",
        "across",
        "through",
        "along",
        "alongside",
        "between",
    }
    _SCENE_NOUN_HINTS = {
        "background",
        "beach",
        "bench",
        "bridge",
        "building",
        "campus",
        "city",
        "court",
        "field",
        "floor",
        "forest",
        "garden",
        "grass",
        "ground",
        "gym",
        "hall",
        "hill",
        "home",
        "house",
        "indoors",
        "indoor",
        "kitchen",
        "lake",
        "mountain",
        "ocean",
        "outdoors",
        "outdoor",
        "park",
        "pool",
        "river",
        "road",
        "room",
        "sand",
        "sea",
        "shore",
        "sidewalk",
        "sky",
        "snow",
        "stage",
        "stadium",
        "street",
        "table",
        "track",
        "trail",
        "tree",
        "water",
        "woods",
        "yard",
    }
    _SUBJECT_STOPWORDS = {
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "another",
        "other",
        "some",
        "someone",
        "somebody",
        "something",
        "young",
        "old",
        "little",
        "small",
        "big",
        "huge",
        "dead",
        "dried",
        "fresh",
        "tiny",
        "large",
        "his",
        "her",
        "their",
        "our",
        "my",
        "your",
        "its",
        "one",
        "two",
        "three",
        "four",
        "five",
        "many",
        "several",
    }
    _ENTITY_CORE_PREFIX_TOKENS = {
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "his",
        "her",
        "their",
        "our",
        "my",
        "your",
        "its",
    }
    _HUMAN_HEAD_TOKENS = {
        "man",
        "woman",
        "boy",
        "girl",
        "person",
        "people",
        "someone",
        "somebody",
        "guy",
        "lady",
        "child",
        "kid",
        "friend",
        "baby",
        "mother",
        "father",
    }
    _LOW_INFO_ENTITY_HEAD_TOKENS = _HUMAN_HEAD_TOKENS | {
        "piece",
        "pieces",
        "part",
        "parts",
        "item",
        "items",
        "thing",
        "things",
        "something",
        "stuff",
        "one",
        "ones",
    }
    _PASSIVE_PREDICATE_BLACKLIST = {
        "a",
        "an",
        "the",
        "this",
        "that",
        "very",
        "too",
        "my",
        "your",
        "his",
        "her",
        "their",
        "its",
    }
    _PASSIVE_HINTS = {
        "played",
        "ridden",
        "driven",
        "cut",
        "sliced",
        "peeled",
        "chopped",
        "diced",
        "stirred",
        "poured",
        "filled",
        "mixed",
        "cooked",
        "opened",
        "closed",
        "used",
        "held",
        "shown",
        "called",
        "made",
        "put",
        "taken",
        "shot",
        "walked",
        "run",
        "jumped",
        "cleaned",
        "washed",
        "prepared",
        "wrapped",
        "spread",
        "sprinkled",
        "grated",
        "fed",
        "licked",
        "drawn",
        "written",
        "read",
        "typed",
        "served",
        "bitten",
        "attacked",
        "watched",
        "kissed",
        "described",
        "named",
    }
    _HUMAN_INVALID_PASSIVE_HINTS = {
        "played",
        "ridden",
        "driven",
        "cut",
        "sliced",
        "peeled",
        "chopped",
        "diced",
        "stirred",
        "poured",
        "filled",
        "mixed",
        "cooked",
        "opened",
        "closed",
        "used",
        "typed",
        "walked",
        "run",
        "jumped",
        "cleaned",
        "washed",
        "grated",
        "spread",
        "sprinkled",
    }
    _ACTIVE_FINITE_HINTS = {
        "break",
        "breaks",
        "play",
        "plays",
        "ride",
        "rides",
        "drive",
        "drives",
        "stab",
        "stabs",
        "cut",
        "cuts",
        "slice",
        "slices",
        "peel",
        "peels",
        "chop",
        "chops",
        "cook",
        "cooks",
        "sing",
        "sings",
        "talk",
        "talks",
        "walk",
        "walks",
        "run",
        "runs",
        "dance",
        "dances",
        "laugh",
        "laughs",
        "shoot",
        "shoots",
        "throw",
        "throws",
        "eat",
        "eats",
        "ate",
        "drink",
        "drinks",
        "brush",
        "brushes",
        "rub",
        "rubs",
        "show",
        "shows",
        "showed",
        "put",
        "puts",
        "make",
        "makes",
        "do",
        "does",
        "did",
        "prepare",
        "prepares",
        "practice",
        "practices",
        "swing",
        "swings",
        "water",
        "waters",
        "demonstrate",
        "demonstrates",
        "demonstrating",
    }

    @classmethod
    def get_phrase_slot_type_specs(
        cls,
        max_phrase_slots: int,
        phrase_slot_schema: str = "raw",
    ) -> List[Dict[str, Any]]:
        slot_count = max(1, int(max_phrase_slots))
        schema = str(phrase_slot_schema).strip().lower() or "raw"
        extra_slot_priority: Optional[Tuple[str, ...]] = None
        allow_extra_repeats = False
        if schema == "typed":
            slot_family_specs = cls._TYPED_SLOT_SPECS
        elif schema == "typed_rich":
            slot_family_specs = cls._TYPED_RICH_SLOT_SPECS
        elif schema == "typed_rich_semantic":
            slot_family_specs = cls._TYPED_RICH_SEMANTIC_SLOT_SPECS
        elif schema == "typed_rich_roleaware":
            slot_family_specs = cls._TYPED_RICH_ROLEAWARE_SLOT_SPECS
            extra_slot_priority = cls._TYPED_RICH_ROLEAWARE_EXTRA_SLOT_PRIORITY
        elif schema == "typed_rich_roleaware_sceneplus":
            slot_family_specs = cls._TYPED_RICH_ROLEAWARE_SLOT_SPECS
            extra_slot_priority = cls._TYPED_RICH_ROLEAWARE_SCENEPLUS_EXTRA_SLOT_PRIORITY
            allow_extra_repeats = True
        elif schema == "family4_compact":
            slot_family_specs = cls._FAMILY4_COMPACT_SLOT_SPECS
        elif schema == "anchored_sov_scene":
            slot_family_specs = cls._ANCHORED_SOV_SCENE_SLOT_SPECS
        else:
            slot_family_specs = tuple()

        specs: List[Dict[str, Any]] = []
        if slot_family_specs:
            family_spec_map = {
                str(spec.get("slot_type", f"slot_{idx}")): dict(spec)
                for idx, spec in enumerate(slot_family_specs)
            }
            family_type_order = [str(spec.get("slot_type", f"slot_{idx}")) for idx, spec in enumerate(slot_family_specs)]
            normalized_extra_priority: List[str] = []
            seen_extra_families = set()
            for slot_type in extra_slot_priority or tuple(family_type_order):
                normalized_slot_type = str(slot_type).strip()
                if (
                    not normalized_slot_type
                    or normalized_slot_type not in family_spec_map
                    or (normalized_slot_type in seen_extra_families and not allow_extra_repeats)
                ):
                    continue
                normalized_extra_priority.append(normalized_slot_type)
                seen_extra_families.add(normalized_slot_type)
            if not normalized_extra_priority:
                normalized_extra_priority = list(family_type_order)

            family_repeat_counts = {
                slot_type: 0
                for slot_type in family_type_order
            }
            slot_family_sequence = list(family_type_order[:slot_count])
            extra_slot_idx = 0
            while len(slot_family_sequence) < slot_count:
                slot_family_sequence.append(
                    normalized_extra_priority[extra_slot_idx % len(normalized_extra_priority)]
                )
                extra_slot_idx += 1

            for slot_idx, slot_type in enumerate(slot_family_sequence):
                base_spec = dict(family_spec_map[slot_type])
                repeat_idx = family_repeat_counts.get(slot_type, 0)
                family_repeat_counts[slot_type] = repeat_idx + 1
                slot_label = slot_type if repeat_idx == 0 else f"{slot_type}_{repeat_idx + 1}"
                family_idx = int(base_spec.get("slot_type_id", slot_idx))
                base_spec.update(
                    {
                        "slot_type_family_id": family_idx,
                        "slot_type_family": slot_type,
                        "slot_repeat_index": repeat_idx,
                        "slot_label": slot_label,
                        "slot_id": slot_idx,
                        "slot_name": f"slot_{slot_idx}",
                        "slot_schema": schema,
                    }
                )
                specs.append(base_spec)
        else:
            for slot_idx in range(slot_count):
                base_spec = {
                    "slot_type_id": slot_idx,
                    "slot_type": "generic",
                    "slot_description": "generic phrase slot",
                    "slot_type_family_id": slot_idx,
                    "slot_type_family": "generic",
                    "slot_repeat_index": 0,
                    "slot_label": f"generic_{slot_idx + 1}",
                }
                base_spec.update(
                    {
                        "slot_id": slot_idx,
                        "slot_name": f"slot_{slot_idx}",
                        "slot_schema": schema,
                    }
                )
                specs.append(base_spec)
        return specs

    @staticmethod
    def parse_slot_type_list(slot_types: str) -> Tuple[str, ...]:
        parsed: List[str] = []
        if slot_types:
            for raw_slot_type in str(slot_types).split(","):
                slot_type = str(raw_slot_type).strip().lower()
                if slot_type and slot_type not in parsed:
                    parsed.append(slot_type)
        return tuple(parsed)

    def __init__(
        self,
        base_dataset: Dataset,
        structured_gt_path: str,
        phrase_max_len: int = 77,
        phrase_fallback_to_caption: bool = True,
        phrase_target_mode: str = "flat",
        max_phrase_slots: int = 4,
        phrase_slot_max_len: int = 24,
        phrase_slot_schema: str = "raw",
        phrase_include_attr_units: bool = False,
        phrase_include_scene_units: bool = False,
        phrase_include_video_phrase_units: bool = False,
        phrase_include_video_attr_units: bool = False,
        phrase_include_video_scene_units: bool = False,
        phrase_video_phrase_min_support: int = 2,
        phrase_video_phrase_max_units: int = 4,
        phrase_slot_active_slot_types: str = "",
        phrase_slot_multiref_enable: bool = False,
        phrase_slot_multiref_max_refs: int = 0,
        phrase_slot_family_sample_mode: str = "first",
        phrase_slot_family_sample_seed: int = 42,
        phrase_slot_family_expand_mode: str = "none",
    ):
        self.base_dataset = base_dataset
        self.structured_gt_path = Path(structured_gt_path)
        self.phrase_max_len = int(phrase_max_len)
        self.phrase_fallback_to_caption = bool(phrase_fallback_to_caption)
        self.phrase_target_mode = str(phrase_target_mode).strip().lower() or "flat"
        if self.phrase_target_mode not in {"flat", "slot"}:
            raise ValueError(f"Unsupported phrase_target_mode: {phrase_target_mode}")
        self.max_phrase_slots = max(1, int(max_phrase_slots))
        self.phrase_slot_max_len = max(4, int(phrase_slot_max_len))
        self.phrase_slot_schema = str(phrase_slot_schema).strip().lower() or "raw"
        if self.phrase_slot_schema not in {
            "raw",
            "typed",
            "typed_rich",
            "typed_rich_semantic",
            "typed_rich_roleaware",
            "typed_rich_roleaware_sceneplus",
            "family4_compact",
            "anchored_sov_scene",
        }:
            raise ValueError(f"Unsupported phrase_slot_schema: {phrase_slot_schema}")
        self.phrase_include_attr_units = bool(phrase_include_attr_units)
        self.phrase_include_scene_units = bool(phrase_include_scene_units)
        self.phrase_include_video_phrase_units = bool(phrase_include_video_phrase_units)
        self.phrase_include_video_attr_units = bool(phrase_include_video_attr_units)
        self.phrase_include_video_scene_units = bool(phrase_include_video_scene_units)
        self.phrase_video_phrase_min_support = max(1, int(phrase_video_phrase_min_support))
        self.phrase_video_phrase_max_units = max(0, int(phrase_video_phrase_max_units))
        self.phrase_slot_type_specs = self.get_phrase_slot_type_specs(
            max_phrase_slots=self.max_phrase_slots,
            phrase_slot_schema=self.phrase_slot_schema,
        )
        self.phrase_slot_active_slot_types = self.parse_slot_type_list(phrase_slot_active_slot_types)
        self.phrase_slot_multiref_enable = bool(phrase_slot_multiref_enable)
        self.phrase_slot_multiref_max_refs = int(phrase_slot_multiref_max_refs)
        family_sample_mode = str(phrase_slot_family_sample_mode).strip().lower() or "first"
        if family_sample_mode not in {"first", "seeded_hash", "epoch_seeded_hash"}:
            raise ValueError(
                "phrase_slot_family_sample_mode must be one of {'first', 'seeded_hash', 'epoch_seeded_hash'}, "
                f"got {phrase_slot_family_sample_mode}"
            )
        self.phrase_slot_family_sample_mode = family_sample_mode
        self.phrase_slot_family_sample_seed = int(phrase_slot_family_sample_seed)
        family_expand_mode = str(phrase_slot_family_expand_mode).strip().lower() or "none"
        if family_expand_mode not in {"none", "parallel"}:
            raise ValueError(
                "phrase_slot_family_expand_mode must be one of {'none', 'parallel'}, "
                f"got {phrase_slot_family_expand_mode}"
            )
        if family_expand_mode != "none" and not (
            self.phrase_target_mode == "slot"
            and self.phrase_slot_schema in {"family4_compact", "typed_rich_roleaware", "anchored_sov_scene"}
        ):
            raise ValueError(
                "phrase_slot_family_expand_mode only supports phrase_target_mode='slot' "
                "with phrase_slot_schema in {'family4_compact', 'typed_rich_roleaware', 'anchored_sov_scene'}."
            )
        self.phrase_slot_family_expand_mode = family_expand_mode

        if not self.structured_gt_path.exists():
            raise FileNotFoundError(f"Structured GT file not found: {self.structured_gt_path}")

        with self.structured_gt_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        self.structured_videos: Dict[str, dict] = payload.get("videos", {})
        self.entity_vocab: List[str] = payload.get("entity_vocab", [])
        self.action_vocab: List[str] = payload.get("action_vocab", [])
        self.attribute_vocab: List[str] = payload.get("attribute_vocab", [])
        self.scene_vocab: List[str] = payload.get("scene_vocab", [])
        self.scene_vocab_lookup = {
            self._normalize_phrase_unit(item)
            for item in self.scene_vocab
            if isinstance(item, str) and item.strip()
        }
        self.scene_token_lookup = {
            token
            for item in self.scene_vocab_lookup
            for token in item.split()
            if token
        }

        self.entity_to_idx = {w: i for i, w in enumerate(self.entity_vocab)}
        self.action_to_idx = {w: i for i, w in enumerate(self.action_vocab)}
        self.attribute_to_idx = {w: i for i, w in enumerate(self.attribute_vocab)}
        self.scene_to_idx = {w: i for i, w in enumerate(self.scene_vocab)}

        if not hasattr(self.base_dataset, "tokenizer"):
            raise AttributeError("base_dataset must expose `tokenizer` for phrase tokenization")
        self.tokenizer = self.base_dataset.tokenizer
        self.phrase_slot_multiref_enable = bool(
            self.phrase_slot_multiref_enable
            and self.phrase_target_mode == "slot"
        )
        if self.phrase_slot_multiref_enable:
            self.phrase_slot_multiref_max_refs = self._resolve_slot_multiref_max_refs(
                requested_max_refs=self.phrase_slot_multiref_max_refs,
            )
        else:
            self.phrase_slot_multiref_max_refs = 0

        self._caption_map_cache: Dict[str, Dict[str, dict]] = {}
        self._video_phrase_bank_cache: Dict[str, List[str]] = {}
        self._phrase_slot_reweight_stats_cache: Dict[Tuple[float, float, float], Dict[str, Any]] = {}
        self.current_epoch = 0
        self.base_sample_count = len(self.base_dataset)
        self.expanded_sample_count = self.base_sample_count
        self.expanded_sample_extra = 0
        self._sample_expansion_index: Optional[List[Tuple[int, int]]] = None
        if self.phrase_slot_family_expand_mode == "parallel":
            expansion_index, expansion_extra = self._build_parallel_family_expansion_index()
            if expansion_extra > 0:
                self._sample_expansion_index = expansion_index
                self.expanded_sample_count = len(expansion_index)
                self.expanded_sample_extra = expansion_extra

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = max(0, int(epoch))

    def _iter_base_caption_keys(self):
        data_list = getattr(self.base_dataset, "data_list", None)
        if isinstance(data_list, list):
            for base_idx, item in enumerate(data_list):
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    yield int(base_idx), str(item[0]), str(item[2])
                    continue
                base = self.base_dataset[base_idx]
                if len(base) < 7:
                    raise ValueError("base_dataset item must have at least 7 fields")
                yield int(base_idx), str(base[5]), str(base[6])
            return

        for base_idx in range(len(self.base_dataset)):
            base = self.base_dataset[base_idx]
            if len(base) < 7:
                raise ValueError("base_dataset item must have at least 7 fields")
            yield int(base_idx), str(base[5]), str(base[6])

    def _resolve_parallel_family_bundle_count(self, cap_info: Optional[dict]) -> int:
        if not isinstance(cap_info, dict):
            return 1
        if self.phrase_slot_schema == "typed_rich_roleaware":
            prepared_outputs = self._prepare_phrase_targets(
                phrase_units=cap_info.get("phrase_units", []),
                caption_fallback=str(cap_info.get("caption", "")),
                caption_text=str(cap_info.get("caption", "")),
                cap_info=cap_info,
                sample_key=f"parallel::{cap_info.get('sen_id', '')}",
                attr_units=cap_info.get("attr_units", []),
                scene_units=cap_info.get("scene_units", []),
                video_phrase_units=[],
                video_attributes=[],
                video_scenes=[],
            )
            _phrase_units, _slot_units, slot_reference_units = self._unpack_phrase_target_outputs(
                prepared_outputs
            )
            bundle_counts = [
                len(normalized_reference_units)
                for normalized_reference_units in (
                    self._normalize_phrase_units(reference_units) for reference_units in slot_reference_units
                )
                if normalized_reference_units
            ]
            return max(bundle_counts, default=1)
        family_counts = []
        for family_key in (
            "family_subject_phrases",
            "family_object_phrases",
            "family_action_phrases",
            "family_scene_phrases",
            "family_relation_phrases",
        ):
            family_items = self._normalize_phrase_units(cap_info.get(family_key))
            if family_items:
                family_counts.append(len(family_items))
        return max(family_counts, default=1)

    def _build_parallel_family_expansion_index(self) -> Tuple[List[Tuple[int, int]], int]:
        expansion_index: List[Tuple[int, int]] = []
        expansion_extra = 0
        for base_idx, vid, sen_id in self._iter_base_caption_keys():
            cap_info = self._video_caption_map(vid).get(sen_id)
            bundle_count = self._resolve_parallel_family_bundle_count(cap_info)
            expansion_extra += max(0, bundle_count - 1)
            for bundle_idx in range(bundle_count):
                expansion_index.append((base_idx, bundle_idx))
        return expansion_index, expansion_extra

    def _select_family_phrase_target_by_bundle(self, family_items: List[str], *, bundle_idx: int) -> str:
        normalized_items = self._normalize_phrase_units(family_items)
        if not normalized_items:
            return ""
        resolved_idx = min(max(0, int(bundle_idx)), len(normalized_items) - 1)
        return normalized_items[resolved_idx]

    def _select_family_phrase_target(
        self,
        family_items: List[str],
        *,
        slot_type: str,
        sample_key: str,
    ) -> str:
        normalized_items = self._normalize_phrase_units(family_items)
        if not normalized_items:
            return ""
        sample_mode = str(getattr(self, "phrase_slot_family_sample_mode", "first")).strip().lower() or "first"
        if sample_mode == "first" or len(normalized_items) == 1:
            return normalized_items[0]

        seed_value = int(getattr(self, "phrase_slot_family_sample_seed", 42))
        epoch_value = int(getattr(self, "current_epoch", 0))
        digest_key = f"{seed_value}|{sample_key}|{slot_type}"
        if sample_mode == "epoch_seeded_hash":
            digest_key = f"{digest_key}|epoch={epoch_value}"
        digest = hashlib.sha256(
            digest_key.encode("utf-8")
        ).digest()
        selected_index = int.from_bytes(digest[:8], byteorder="big", signed=False) % len(normalized_items)
        return normalized_items[selected_index]

    def _resolve_slot_multiref_max_refs(self, requested_max_refs: int) -> int:
        requested_max_refs = int(requested_max_refs)
        if requested_max_refs > 0:
            return requested_max_refs

        max_refs = 1
        for vid, video_info in self.structured_videos.items():
            attributes = video_info.get("attributes", None)
            scenes = video_info.get("scenes", None)
            for cap_info in video_info.get("captions", []):
                if not isinstance(cap_info, dict):
                    continue
                prepared_outputs = self._prepare_phrase_targets(
                    phrase_units=cap_info.get("phrase_units", []),
                    caption_fallback=str(cap_info.get("caption", "")),
                    caption_text=str(cap_info.get("caption", "")),
                    cap_info=cap_info,
                    sample_key=f"{vid}::{cap_info.get('sen_id', '')}",
                    attr_units=cap_info.get("attr_units", []),
                    scene_units=cap_info.get("scene_units", []),
                    video_phrase_units=[],
                    video_attributes=attributes if isinstance(attributes, list) else None,
                    video_scenes=scenes if isinstance(scenes, list) else None,
                )
                _phrase_units, _slot_units, slot_reference_units = self._unpack_phrase_target_outputs(
                    prepared_outputs
                )
                for reference_units in slot_reference_units[: self.max_phrase_slots]:
                    normalized_reference_units = self._normalize_phrase_units(reference_units)
                    if normalized_reference_units:
                        max_refs = max(max_refs, len(normalized_reference_units))
        return max_refs

    def _resolve_family4_compact_multiref_max_refs(self, requested_max_refs: int) -> int:
        return self._resolve_slot_multiref_max_refs(requested_max_refs)

    @property
    def entity_dim(self) -> int:
        return len(self.entity_vocab)

    @property
    def action_dim(self) -> int:
        return len(self.action_vocab)

    @property
    def attribute_dim(self) -> int:
        return len(self.attribute_vocab)

    @property
    def scene_dim(self) -> int:
        return len(self.scene_vocab)

    def __len__(self) -> int:
        if self._sample_expansion_index is not None:
            return len(self._sample_expansion_index)
        return len(self.base_dataset)

    def _video_caption_map(self, vid: str) -> Dict[str, dict]:
        if vid in self._caption_map_cache:
            return self._caption_map_cache[vid]
        video_info = self.structured_videos.get(vid, {})
        captions = video_info.get("captions", [])
        cap_map = {}
        for item in captions:
            if not isinstance(item, dict):
                continue
            sen_id = str(item.get("sen_id", ""))
            if not sen_id:
                continue
            cap_map[sen_id] = item
        self._caption_map_cache[vid] = cap_map
        return cap_map

    def _caption_branch_units(self, cap_info: Optional[dict], *branch_names: str) -> List[str]:
        if not isinstance(cap_info, dict):
            return []
        sources: List[List[str]] = []
        for branch_name in branch_names:
            branch_value = cap_info.get(branch_name, [])
            if isinstance(branch_value, list):
                sources.append(branch_value)
        return self._merge_phrase_unit_sources(*sources)

    def _caption_stage1_items(self, cap_info: Optional[dict], branch_name: str) -> List[str]:
        if not isinstance(cap_info, dict):
            return []
        stage1 = cap_info.get("stage1")
        if not isinstance(stage1, dict):
            return []

        collected: List[str] = []
        for item in stage1.get(branch_name) or []:
            if isinstance(item, dict):
                text = item.get("canonical_text", "") or item.get("surface_text", "")
            else:
                text = item
            normalized = self._normalize_phrase_unit(text)
            if normalized:
                collected.append(normalized)
        return self._dedupe_preserve_order(collected)

    def _normalize_scene_label(self, scene_text: str) -> str:
        normalized = self._normalize_phrase_unit(scene_text)
        if not normalized:
            return ""
        if normalized in self.scene_to_idx:
            return normalized

        tokens = normalized.split()
        for span_len in range(len(tokens), 0, -1):
            for start in range(0, len(tokens) - span_len + 1):
                candidate = " ".join(tokens[start : start + span_len])
                if candidate in self.scene_to_idx:
                    return candidate
        return ""

    def _caption_entity_items(self, cap_info: Optional[dict]) -> List[str]:
        items = self._caption_branch_units(cap_info, "subject_entities", "object_entities")
        if not items:
            items = self._caption_stage1_items(cap_info, "entities")
        return [item for item in items if item in self.entity_to_idx]

    def _caption_action_items(self, cap_info: Optional[dict]) -> List[str]:
        items = self._caption_stage1_items(cap_info, "action_support")
        return [item for item in items if item in self.action_to_idx]

    def _caption_attribute_items(self, cap_info: Optional[dict]) -> List[str]:
        items = self._caption_branch_units(
            cap_info,
            "subject_attributes",
            "object_attributes",
            "attr_units",
        )
        return [item for item in items if item in self.attribute_to_idx]

    def _caption_scene_items(self, cap_info: Optional[dict]) -> List[str]:
        scene_items = self._caption_branch_units(cap_info, "scene_phrases", "scene_units")
        if not scene_items:
            scene_items = self._caption_stage1_items(cap_info, "scenes")

        labels: List[str] = []
        seen = set()
        for item in scene_items:
            label = self._normalize_scene_label(item)
            if not label or label in seen:
                continue
            seen.add(label)
            labels.append(label)
        return labels

    def _multi_hot(self, items: List[str], vocab_map: Dict[str, int], dim: int) -> torch.Tensor:
        vec = torch.zeros(dim, dtype=torch.float32)
        if dim == 0:
            return vec
        for item in items:
            idx = vocab_map.get(item)
            if idx is not None:
                vec[idx] = 1.0
        return vec

    def _encode_phrase_text(self, phrase_text: str, max_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer.encode_plus(
            phrase_text,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        token_ids = encoded["input_ids"].squeeze(0)
        token_mask = encoded["attention_mask"].squeeze(0)
        return token_ids, token_mask

    @staticmethod
    def _normalize_phrase_unit(phrase_text: str) -> str:
        return " ".join(str(phrase_text).strip().lower().split())

    @staticmethod
    def _dedupe_preserve_order(phrase_units: List[str]) -> List[str]:
        deduped: List[str] = []
        seen = set()
        for phrase in phrase_units:
            if phrase in seen:
                continue
            seen.add(phrase)
            deduped.append(phrase)
        return deduped

    def _normalize_phrase_units(self, phrase_units: Optional[List[str]]) -> List[str]:
        if not isinstance(phrase_units, list):
            return []
        return self._dedupe_preserve_order(
            [
                self._normalize_phrase_unit(phrase_text)
                for phrase_text in phrase_units
                if isinstance(phrase_text, str) and phrase_text.strip()
            ]
        )

    def _merge_phrase_unit_sources(self, *phrase_sources: Optional[List[str]]) -> List[str]:
        merged: List[str] = []
        for source in phrase_sources:
            merged.extend(self._normalize_phrase_units(source))
        return self._dedupe_preserve_order(merged)

    def _caption_stage2_phrase_units(self, cap_info: Optional[dict]) -> List[str]:
        if not isinstance(cap_info, dict):
            return []
        stage2 = cap_info.get("stage2")
        if isinstance(stage2, dict):
            phrases: List[str] = []
            primary_phrase = stage2.get("primary_phrase")
            if isinstance(primary_phrase, dict):
                phrases.append(primary_phrase.get("text", ""))
            for item in stage2.get("supporting_phrases") or []:
                if isinstance(item, dict):
                    phrases.append(item.get("text", ""))
            return self._normalize_phrase_units(phrases)
        return self._normalize_phrase_units(cap_info.get("phrase_units", []))

    def _caption_family_slot_items(self, cap_info: Optional[dict], slot_type: str) -> List[str]:
        normalized_slot_type = str(slot_type).strip().lower()
        if normalized_slot_type == "subject_entity":
            items = self._caption_branch_units(
                cap_info,
                "family_subject_phrases",
                "subject_phrase_candidates",
                "subject_entities",
            )
            if not items:
                items = self._caption_stage1_items(cap_info, "entities")
            return items
        if normalized_slot_type == "object_entity":
            items = self._caption_branch_units(
                cap_info,
                "family_object_phrases",
                "object_phrase_candidates",
                "object_entities",
            )
            if not items:
                items = self._caption_stage1_items(cap_info, "entities")
            return items
        if normalized_slot_type == "subject_action":
            items = self._caption_branch_units(
                cap_info,
                "family_action_phrases",
                "subject_action_phrases",
                "object_passive_phrases",
            )
            if not items:
                items = self._caption_stage1_items(cap_info, "action_support")
            return items
        if normalized_slot_type == "scene_context":
            items = self._caption_branch_units(
                cap_info,
                "family_scene_phrases",
                "scene_phrases",
                "scene_units",
            )
            if not items:
                items = self._caption_stage1_items(cap_info, "scenes")
            return items
        if normalized_slot_type == "relation_detail":
            return self._caption_branch_units(
                cap_info,
                "family_relation_phrases",
                "relation_phrase_candidates",
                "relation_phrases",
                "instrument_phrases",
            )
        return []

    def _video_phrase_bank(self, vid: str) -> List[str]:
        if not bool(getattr(self, "phrase_include_video_phrase_units", False)):
            return []

        cache = getattr(self, "_video_phrase_bank_cache", None)
        if cache is None:
            cache = {}
            self._video_phrase_bank_cache = cache
        if vid in cache:
            return cache[vid]

        min_support = max(1, int(getattr(self, "phrase_video_phrase_min_support", 2)))
        max_units = max(0, int(getattr(self, "phrase_video_phrase_max_units", 4)))

        support_counter: Counter = Counter()
        first_seen: Dict[str, int] = {}
        for cap_info in self._video_caption_map(vid).values():
            for phrase_text in self._caption_stage2_phrase_units(cap_info):
                support_counter[phrase_text] += 1
                if phrase_text not in first_seen:
                    first_seen[phrase_text] = len(first_seen)

        ranked_units = [
            phrase_text
            for phrase_text, support in sorted(
                support_counter.items(),
                key=lambda item: (-int(item[1]), first_seen.get(item[0], 10**9), item[0]),
            )
            if int(support) >= min_support
        ]
        if max_units > 0:
            ranked_units = ranked_units[:max_units]
        cache[vid] = ranked_units
        return ranked_units

    def _derive_caption_phrase_units(self, caption_text: Optional[str]) -> List[str]:
        normalized = self._normalize_phrase_unit(caption_text or "")
        if not normalized:
            return []

        candidates: List[str] = []
        seen = set()

        def append_candidate(phrase_text: Optional[str]) -> None:
            candidate = self._normalize_phrase_unit(phrase_text or "")
            if not candidate:
                return
            self._append_bucket_phrase(candidates, candidate, seen)

        append_candidate(normalized)

        clause_subject, clause_action, clause_object, clause_detail = self._split_noncopular_clause(normalized)
        if clause_subject and clause_action:
            append_candidate(clause_action)
            append_candidate(clause_subject)
            append_candidate(clause_object)
            append_candidate(clause_detail)

        subject_text, predicate_text = self._split_copula_predicate(normalized)
        if subject_text is not None and predicate_text is not None:
            subject_phrase = self._extract_subject_phrase(subject_text)
            append_candidate(subject_phrase)
            relation_head, tail_tokens, _ = self._split_relation_head_tail(normalized)
            if relation_head is not None and tail_tokens:
                append_candidate(" ".join([relation_head] + tail_tokens))

        for entity_phrase in [clause_subject, clause_object, normalized]:
            core_entity, modifier_phrase, detail_phrase = self._split_entity_core_and_modifier(entity_phrase)
            append_candidate(core_entity)
            append_candidate(modifier_phrase)
            append_candidate(detail_phrase)

        return candidates

    @staticmethod
    def _append_bucket_phrase(slot_bucket: List[str], phrase_text: str, seen_phrases: set) -> None:
        if not phrase_text or phrase_text in seen_phrases:
            return
        seen_phrases.add(phrase_text)
        slot_bucket.append(phrase_text)

    def _split_copula_predicate(self, phrase_text: str) -> Tuple[Optional[str], Optional[str]]:
        tokens = phrase_text.split()
        for idx, token in enumerate(tokens):
            if token in self._COPULA_TOKENS and 0 < idx < len(tokens) - 1:
                return " ".join(tokens[:idx]), " ".join(tokens[idx + 1 :])
        return None, None

    def _extract_subject_head(self, subject_text: Optional[str]) -> str:
        if not subject_text:
            return ""
        tokens = [tok.strip(" ,.;:!?") for tok in subject_text.split()]
        tokens = [tok for tok in tokens if tok]
        if not tokens:
            return ""
        for token in reversed(tokens):
            if token not in self._SUBJECT_STOPWORDS:
                return token
        return tokens[-1]

    def _looks_like_active_predicate(self, predicate_head: str) -> bool:
        if not predicate_head:
            return False
        if predicate_head in self._RELATION_HEAD_TOKENS:
            return False
        return predicate_head.endswith("ing") or predicate_head in self._ACTIVE_FINITE_HINTS

    def _looks_like_passive_predicate(self, predicate_head: str) -> bool:
        if not predicate_head:
            return False
        if predicate_head in self._PASSIVE_PREDICATE_BLACKLIST:
            return False
        if predicate_head in self._RELATION_HEAD_TOKENS:
            return False
        if predicate_head in self._PASSIVE_HINTS:
            return True
        return predicate_head.endswith("ed") or predicate_head.endswith("en")

    def _is_relation_phrase(self, phrase_text: str) -> bool:
        tokens = phrase_text.split()
        if not tokens:
            return False
        if tokens[0] in self._RELATION_HEAD_TOKENS:
            return True
        subject_text, predicate_text = self._split_copula_predicate(phrase_text)
        if subject_text is not None and predicate_text is not None:
            predicate_head = predicate_text.split()[0] if predicate_text.split() else ""
            if predicate_head in self._RELATION_HEAD_TOKENS:
                return True
        return any(token in self._RELATION_HEAD_TOKENS for token in tokens[1:])

    def _split_relation_head_tail(
        self,
        phrase_text: str,
    ) -> Tuple[Optional[str], List[str], Optional[str]]:
        normalized = self._normalize_phrase_unit(phrase_text)
        if not normalized:
            return None, [], None

        subject_text, predicate_text = self._split_copula_predicate(normalized)
        if subject_text is not None and predicate_text is not None:
            predicate_tokens = predicate_text.split()
            if predicate_tokens and predicate_tokens[0] in self._RELATION_HEAD_TOKENS:
                return predicate_tokens[0], predicate_tokens[1:], self._extract_subject_phrase(subject_text)

        tokens = normalized.split()
        for idx, token in enumerate(tokens):
            if token in self._RELATION_HEAD_TOKENS:
                subject_prefix = " ".join(tokens[:idx]).strip() if idx > 0 else None
                return token, tokens[idx + 1 :], subject_prefix or None
        return None, [], None

    def _looks_like_clause_verb_token(self, token: str) -> bool:
        normalized = self._normalize_phrase_unit(token)
        if not normalized or normalized in self._RELATION_HEAD_TOKENS:
            return False
        if self._looks_like_active_predicate(normalized):
            return True
        if normalized in self._PASSIVE_HINTS:
            return True
        return normalized.endswith("ed") or normalized.endswith("en")

    def _looks_like_verb_led_fragment(self, phrase_text: Optional[str]) -> bool:
        normalized = self._normalize_phrase_unit(phrase_text or "")
        if not normalized:
            return False
        tokens = normalized.split()
        if not tokens:
            return False
        head = tokens[0]
        if self._looks_like_active_predicate(head):
            return True
        if not self._looks_like_clause_verb_token(head) or len(tokens) < 2:
            return False
        return tokens[1] in self._ENTITY_CORE_PREFIX_TOKENS or tokens[1] == "to"

    def _looks_like_entity_phrase_basic(self, phrase_text: str) -> bool:
        normalized = self._normalize_phrase_unit(phrase_text)
        if not normalized or self._is_relation_phrase(normalized):
            return False
        subject_text, predicate_text = self._split_copula_predicate(normalized)
        if subject_text is not None and predicate_text is not None:
            predicate_head = predicate_text.split()[0] if predicate_text.split() else ""
            if self._looks_like_active_predicate(predicate_head) or self._looks_like_passive_predicate(predicate_head):
                return False
            return True
        tokens = normalized.split()
        if not tokens:
            return False
        if all(token in self._ENTITY_CORE_PREFIX_TOKENS for token in tokens):
            return False
        tail = tokens[-1]
        if self._looks_like_active_predicate(tail) or self._looks_like_passive_predicate(tail):
            return False
        return True

    def _looks_like_entity_or_attached_detail_phrase(self, phrase_text: Optional[str]) -> bool:
        normalized = self._normalize_phrase_unit(phrase_text or "")
        if not normalized or self._looks_like_verb_led_fragment(normalized):
            return False
        if self._looks_like_entity_phrase_basic(normalized):
            return True
        base_entity, detail_phrase = self._split_entity_core_and_detail(normalized)
        return bool(
            detail_phrase
            and base_entity != normalized
            and self._looks_like_entity_phrase_basic(base_entity)
        )

    def _has_noncopular_clause_signature(self, phrase_text: Optional[str]) -> bool:
        normalized = self._normalize_phrase_unit(phrase_text or "")
        if not normalized or self._looks_like_verb_led_fragment(normalized):
            return False
        subject_text, predicate_text = self._split_copula_predicate(normalized)
        if subject_text is not None or predicate_text is not None:
            return False
        tokens = normalized.split()
        if len(tokens) < 3:
            return False
        for idx in range(1, len(tokens) - 1):
            if not self._looks_like_clause_verb_token(tokens[idx]):
                continue
            subject_phrase = self._normalize_phrase_unit(" ".join(tokens[:idx]))
            if self._looks_like_entity_or_attached_detail_phrase(subject_phrase):
                return True
        return False

    def _split_noncopular_clause(
        self,
        phrase_text: Optional[str],
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        normalized = self._normalize_phrase_unit(phrase_text or "")
        if not normalized or self._looks_like_verb_led_fragment(normalized):
            return None, None, None, None

        subject_text, predicate_text = self._split_copula_predicate(normalized)
        if subject_text is not None or predicate_text is not None:
            return None, None, None, None

        tokens = normalized.split()
        if len(tokens) < 2:
            return None, None, None, None

        for idx in range(1, len(tokens)):
            verb_token = tokens[idx]
            if not self._looks_like_clause_verb_token(verb_token):
                continue

            subject_phrase = self._normalize_phrase_unit(" ".join(tokens[:idx]))
            if not self._looks_like_entity_or_attached_detail_phrase(subject_phrase):
                continue

            remainder = tokens[idx + 1 :]
            relation_idx = None
            for offset, token in enumerate(remainder):
                if token in self._RELATION_HEAD_TOKENS:
                    relation_idx = offset
                    break

            if relation_idx is None:
                middle_tokens = remainder
                detail_phrase = None
            else:
                middle_tokens = remainder[:relation_idx]
                detail_phrase = self._normalize_phrase_unit(" ".join(remainder[relation_idx:])) or None

            predicate_tail_tokens: List[str] = []
            object_phrase = None
            if middle_tokens:
                middle_phrase = self._normalize_phrase_unit(" ".join(middle_tokens))
                if middle_tokens[0] == "to" or self._looks_like_clause_verb_token(middle_tokens[0]):
                    predicate_tail_tokens = middle_tokens
                elif self._looks_like_entity_or_attached_detail_phrase(middle_phrase):
                    object_phrase = middle_phrase
                else:
                    predicate_tail_tokens = middle_tokens

            action_phrase = self._normalize_phrase_unit(
                " ".join(tokens[: idx + 1] + predicate_tail_tokens)
            )
            if not action_phrase:
                continue
            return subject_phrase, action_phrase, object_phrase, detail_phrase

        return None, None, None, None

    def _split_entity_core_and_detail(self, phrase_text: Optional[str]) -> Tuple[str, Optional[str]]:
        normalized = self._normalize_phrase_unit(phrase_text or "")
        if not normalized:
            return "", None

        relation_head, tail_tokens, subject_prefix = self._split_relation_head_tail(normalized)
        if relation_head is None or not tail_tokens or not subject_prefix:
            return normalized, None

        base_entity = self._normalize_phrase_unit(subject_prefix)
        detail_phrase = self._normalize_phrase_unit(" ".join([relation_head] + tail_tokens))
        if not base_entity or not detail_phrase:
            return normalized, None
        return base_entity, detail_phrase

    def _extract_entity_core_phrase(self, phrase_text: Optional[str]) -> str:
        normalized = self._normalize_phrase_unit(phrase_text or "")
        if not normalized:
            return ""

        tokens = normalized.split()
        if not tokens:
            return ""
        if len(tokens) == 1:
            return normalized

        head_idx = -1
        for idx in range(len(tokens) - 1, -1, -1):
            if tokens[idx] not in self._SUBJECT_STOPWORDS:
                head_idx = idx
                break
        if head_idx < 0:
            return normalized

        if head_idx == 0:
            return tokens[0]

        prefix_tokens: List[str] = []
        if tokens[0] in self._ENTITY_CORE_PREFIX_TOKENS:
            if tokens[0] in {"a", "an"}:
                article = "an" if tokens[head_idx][:1] in {"a", "e", "i", "o", "u"} else "a"
                prefix_tokens.append(article)
            else:
                prefix_tokens.append(tokens[0])

        core_tokens = prefix_tokens + [tokens[head_idx]]
        core_phrase = self._normalize_phrase_unit(" ".join(core_tokens))
        return core_phrase or normalized

    def _extract_content_tokens(self, phrase_text: Optional[str]) -> List[str]:
        normalized = self._normalize_phrase_unit(phrase_text or "")
        if not normalized:
            return []
        return [token for token in normalized.split() if token not in self._SUBJECT_STOPWORDS]

    def _extract_phrase_head_token(self, phrase_text: Optional[str]) -> Optional[str]:
        content_tokens = self._extract_content_tokens(phrase_text)
        if not content_tokens:
            return None
        return content_tokens[-1]

    def _looks_like_low_information_entity_head(self, token: Optional[str]) -> bool:
        normalized = self._normalize_phrase_unit(token or "")
        return bool(normalized) and normalized in self._LOW_INFO_ENTITY_HEAD_TOKENS

    def _is_degenerate_entity_modifier_phrase(
        self,
        phrase_text: Optional[str],
        *,
        core_entity: Optional[str] = None,
    ) -> bool:
        normalized = self._normalize_phrase_unit(phrase_text or "")
        if not normalized:
            return False

        tokens = normalized.split()
        content_tokens = self._extract_content_tokens(normalized)
        if len(content_tokens) >= 2 and len(set(content_tokens)) == 1:
            return True

        tail_token = tokens[-1] if tokens else ""
        if tail_token and (
            self._looks_like_active_predicate(tail_token)
            or self._looks_like_passive_predicate(tail_token)
        ):
            return True

        core_head = self._extract_phrase_head_token(core_entity or "")
        if core_head and len(content_tokens) >= 2:
            if content_tokens[-1] == core_head and content_tokens[-2] == core_head:
                return True

        if "of" not in tokens:
            return False

        of_idx = tokens.index("of")
        left_phrase = self._normalize_phrase_unit(" ".join(tokens[:of_idx]))
        right_phrase = self._normalize_phrase_unit(" ".join(tokens[of_idx + 1 :]))
        left_head = self._extract_phrase_head_token(left_phrase)
        right_head = self._extract_phrase_head_token(right_phrase)
        if left_head and right_head and left_head == right_head:
            return True
        if core_head and right_head == core_head and (not left_head or left_head == core_head):
            return True
        if of_idx == 0 and self._looks_like_low_information_entity_head(right_head):
            return True
        return False

    def _is_low_information_relation_detail(
        self,
        phrase_text: Optional[str],
        *,
        base_entity: Optional[str] = None,
    ) -> bool:
        normalized = self._normalize_phrase_unit(phrase_text or "")
        if not normalized:
            return False

        tokens = normalized.split()
        if not tokens:
            return False

        relation_head = tokens[0]
        if relation_head in self._RELATION_HEAD_TOKENS:
            _, tail_tokens, _ = self._split_relation_head_tail(normalized)
        elif relation_head == "of":
            tail_tokens = tokens[1:]
        else:
            return False

        if not tail_tokens:
            return True

        tail_phrase = self._normalize_phrase_unit(" ".join(tail_tokens))
        tail_head = self._extract_phrase_head_token(tail_phrase)
        base_head = self._extract_phrase_head_token(base_entity or "")
        if base_head and tail_head and base_head == tail_head:
            return True
        if relation_head == "of" and self._looks_like_low_information_entity_head(tail_head):
            return True
        return False

    def _split_entity_core_and_modifier(
        self,
        phrase_text: Optional[str],
    ) -> Tuple[str, Optional[str], Optional[str]]:
        normalized = self._normalize_phrase_unit(phrase_text or "")
        if not normalized:
            return "", None, None
        if self._looks_like_verb_led_fragment(normalized):
            return normalized, None, None
        clause_subject, clause_action, _, _ = self._split_noncopular_clause(normalized)
        if clause_subject and clause_action:
            return normalized, None, None

        modifier_phrase: Optional[str] = None
        detail_phrase: Optional[str] = None

        relation_head, tail_tokens, subject_prefix = self._split_relation_head_tail(normalized)
        base_phrase = normalized
        if relation_head is not None and (not tail_tokens or not subject_prefix):
            return normalized, None, None
        if relation_head is not None and tail_tokens and subject_prefix:
            base_phrase = self._normalize_phrase_unit(subject_prefix)
            detail_phrase = self._normalize_phrase_unit(" ".join([relation_head] + tail_tokens))
            modifier_phrase = normalized

        core_entity = self._extract_entity_core_phrase(base_phrase)
        if core_entity and core_entity != base_phrase:
            modifier_phrase = modifier_phrase or base_phrase

        if not core_entity:
            core_entity = base_phrase or normalized

        if modifier_phrase:
            modifier_phrase = self._normalize_phrase_unit(modifier_phrase)
            if modifier_phrase == core_entity:
                modifier_phrase = None
            elif self._is_degenerate_entity_modifier_phrase(modifier_phrase, core_entity=core_entity):
                modifier_phrase = None
        if detail_phrase:
            detail_phrase = self._normalize_phrase_unit(detail_phrase)
            if detail_phrase == core_entity:
                detail_phrase = None
            elif self._is_low_information_relation_detail(detail_phrase, base_entity=core_entity):
                detail_phrase = None

        return core_entity, modifier_phrase, detail_phrase

    def _sanitize_passive_phrase(self, phrase_text: str) -> Optional[str]:
        subject_text, predicate_text = self._split_copula_predicate(phrase_text)
        if subject_text is None or predicate_text is None:
            return phrase_text
        predicate_tokens = predicate_text.split()
        if not predicate_tokens:
            return None
        predicate_head = predicate_tokens[0]
        if predicate_head in self._PASSIVE_PREDICATE_BLACKLIST:
            return None
        subject_head = self._extract_subject_head(subject_text)
        if (
            subject_head in self._HUMAN_HEAD_TOKENS
            and predicate_head in self._HUMAN_INVALID_PASSIVE_HINTS
        ):
            return None
        return phrase_text

    def _extract_subject_phrase(self, phrase_text: Optional[str]) -> str:
        return self._normalize_phrase_unit(phrase_text or "")

    def _prefers_subject_entity_slot(self, phrase_text: str) -> bool:
        head = self._extract_subject_head(phrase_text)
        return head in self._HUMAN_HEAD_TOKENS

    def _looks_like_scene_phrase(self, phrase_text: str) -> bool:
        normalized = self._normalize_phrase_unit(phrase_text)
        if not normalized:
            return False
        if normalized in getattr(self, "scene_vocab_lookup", set()):
            return True
        tokens = normalized.split()
        scene_tokens = getattr(self, "scene_token_lookup", set())
        return any(token in scene_tokens for token in tokens)

    def _tail_has_scene_hint(self, tokens: List[str]) -> bool:
        if not tokens:
            return False
        normalized_tail = self._normalize_phrase_unit(" ".join(tokens))
        if not normalized_tail:
            return False
        if normalized_tail in getattr(self, "scene_vocab_lookup", set()):
            return True
        scene_tokens = getattr(self, "scene_token_lookup", set())
        return any(token in scene_tokens or token in self._SCENE_NOUN_HINTS for token in normalized_tail.split())

    def _looks_like_clean_scene_context_phrase(self, phrase_text: str) -> bool:
        normalized = self._normalize_phrase_unit(phrase_text)
        if not normalized:
            return False
        if normalized in getattr(self, "scene_vocab_lookup", set()):
            return True

        tokens = normalized.split()
        if not tokens:
            return False

        if (
            not self._is_relation_phrase(normalized)
            and self._tail_has_scene_hint(tokens)
            and not any(
                self._looks_like_active_predicate(token) or self._looks_like_passive_predicate(token)
                for token in tokens
            )
        ):
            return True

        relation_head, tail_tokens, _ = self._split_relation_head_tail(normalized)
        if relation_head is None or relation_head not in self._SCENE_RELATION_HEAD_TOKENS:
            return False
        if not tail_tokens or not self._tail_has_scene_hint(tail_tokens):
            return False

        tail_head = tail_tokens[0]
        if self._looks_like_active_predicate(tail_head) or self._looks_like_passive_predicate(tail_head):
            return False
        if any(token in self._HUMAN_HEAD_TOKENS for token in tail_tokens):
            return False
        return True

    def _looks_like_instrument_phrase(self, phrase_text: str) -> bool:
        normalized = self._normalize_phrase_unit(phrase_text)
        if not normalized:
            return False

        relation_head, tail_tokens, _ = self._split_relation_head_tail(normalized)
        if relation_head is None or relation_head not in self._INSTRUMENT_RELATION_HEAD_TOKENS:
            return False
        if not tail_tokens or self._tail_has_scene_hint(tail_tokens):
            return False

        tail_token_set = set(tail_tokens)
        if relation_head == "using":
            return True
        if relation_head == "via":
            return True
        if relation_head == "by":
            return "hand" in tail_token_set or "hands" in tail_token_set or bool(
                tail_token_set & self._INSTRUMENT_NOUN_HINTS
            )
        if any(token in self._HUMAN_HEAD_TOKENS for token in tail_tokens):
            return False
        if tail_token_set & self._INSTRUMENT_NOUN_HINTS:
            return True
        return len(tail_tokens) <= 4

    def _looks_like_entity_phrase(self, phrase_text: str) -> bool:
        normalized = self._normalize_phrase_unit(phrase_text)
        if (
            not normalized
            or self._looks_like_verb_led_fragment(normalized)
            or self._has_noncopular_clause_signature(normalized)
        ):
            return False
        return self._looks_like_entity_phrase_basic(normalized)

    def _classify_phrase_unit(self, phrase_text: str) -> int:
        subject_text, predicate_text = self._split_copula_predicate(phrase_text)
        if subject_text is not None and predicate_text is not None:
            predicate_head = predicate_text.split()[0] if predicate_text.split() else ""
            if predicate_head in self._RELATION_HEAD_TOKENS:
                return 2
            if self._looks_like_active_predicate(predicate_head):
                return 0
            if self._looks_like_passive_predicate(predicate_head):
                return 1

        tokens = phrase_text.split()
        if tokens:
            last_token = tokens[-1]
            if self._looks_like_active_predicate(last_token):
                return 0
        if self._is_relation_phrase(phrase_text):
            return 2
        return 3

    def _build_typed_phrase_targets(
        self,
        phrase_units: List[str],
        attr_units: Optional[List[str]] = None,
        scene_units: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        slot_buckets: Dict[int, List[str]] = {0: [], 1: [], 2: [], 3: []}
        seen_phrases = set()

        for phrase_text in phrase_units:
            slot_idx = self._classify_phrase_unit(phrase_text)
            if slot_idx == 1:
                phrase_text = self._sanitize_passive_phrase(phrase_text)
                if not phrase_text:
                    continue
            self._append_bucket_phrase(slot_buckets[slot_idx], phrase_text, seen_phrases)

        if self.phrase_include_scene_units:
            for phrase_text in scene_units or []:
                self._append_bucket_phrase(slot_buckets[2], phrase_text, seen_phrases)

        if self.phrase_include_attr_units:
            for phrase_text in attr_units or []:
                self._append_bucket_phrase(slot_buckets[3], phrase_text, seen_phrases)

        slot_units = [""] * self.max_phrase_slots
        ordered_units: List[str] = []
        bucket_offsets = {0: 0, 1: 0, 2: 0, 3: 0}
        slot_type_to_bucket = {
            "subject_action": 0,
            "object_passive": 1,
            "relation_scene": 2,
            "attribute_misc": 3,
        }

        for slot_spec in self.phrase_slot_type_specs:
            slot_idx = int(slot_spec.get("slot_id", 0))
            slot_type = str(slot_spec.get("slot_type_family", slot_spec.get("slot_type", "attribute_misc")))
            bucket_idx = slot_type_to_bucket.get(slot_type, 3)
            bucket = slot_buckets[bucket_idx]
            bucket_offset = bucket_offsets[bucket_idx]
            if bucket_offset >= len(bucket):
                continue
            phrase_text = bucket[bucket_offset]
            bucket_offsets[bucket_idx] = bucket_offset + 1
            slot_units[slot_idx] = phrase_text
            ordered_units.append(phrase_text)

        for slot_type in self._TYPED_SLOT_FAMILY_ORDER:
            bucket_idx = slot_type_to_bucket[slot_type]
            bucket = slot_buckets[bucket_idx]
            bucket_offset = bucket_offsets[bucket_idx]
            if bucket_offset < len(bucket):
                ordered_units.extend(bucket[bucket_offset:])

        return ordered_units, slot_units

    def _build_typed_rich_phrase_targets(
        self,
        phrase_units: List[str],
        attr_units: Optional[List[str]] = None,
        scene_units: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        slot_buckets: Dict[str, List[str]] = {
            slot_type: []
            for slot_type in self._TYPED_RICH_SLOT_FAMILY_ORDER
        }
        seen_phrases = set()

        def append_phrase(slot_type: str, phrase_text: Optional[str]) -> None:
            normalized = self._normalize_phrase_unit(phrase_text or "")
            if not normalized:
                return
            self._append_bucket_phrase(slot_buckets[slot_type], normalized, seen_phrases)

        def route_entity_phrase(phrase_text: Optional[str], preferred_slot: Optional[str] = None) -> None:
            normalized = self._normalize_phrase_unit(phrase_text or "")
            if not normalized or not self._looks_like_entity_phrase(normalized):
                return
            slot_type = preferred_slot
            if slot_type is None:
                slot_type = "subject_entity" if self._prefers_subject_entity_slot(normalized) else "object_entity"
            append_phrase(slot_type, normalized)

        def route_detail_or_scene_phrase(phrase_text: Optional[str]) -> None:
            normalized = self._normalize_phrase_unit(phrase_text or "")
            if not normalized:
                return
            if self._looks_like_clean_scene_context_phrase(normalized):
                append_phrase("scene_context", normalized)
            else:
                append_phrase("relation_detail", normalized)

        for phrase_text in phrase_units:
            subject_text, predicate_text = self._split_copula_predicate(phrase_text)
            predicate_head = predicate_text.split()[0] if predicate_text and predicate_text.split() else ""
            subject_phrase = self._extract_subject_phrase(subject_text)

            if subject_text is not None and predicate_text is not None:
                if predicate_head in self._RELATION_HEAD_TOKENS:
                    route_detail_or_scene_phrase(phrase_text)
                    route_entity_phrase(subject_phrase, preferred_slot="subject_entity")
                    continue
                if self._looks_like_active_predicate(predicate_head):
                    append_phrase("subject_action", phrase_text)
                    route_entity_phrase(subject_phrase, preferred_slot="subject_entity")
                    continue
                if self._looks_like_passive_predicate(predicate_head):
                    sanitized_phrase = self._sanitize_passive_phrase(phrase_text)
                    if not sanitized_phrase:
                        continue
                    append_phrase("object_passive", sanitized_phrase)
                    route_entity_phrase(subject_phrase, preferred_slot="object_entity")
                    continue
                route_entity_phrase(subject_phrase, preferred_slot="subject_entity")
                route_detail_or_scene_phrase(phrase_text)
                continue

            if self._looks_like_clean_scene_context_phrase(phrase_text):
                append_phrase("scene_context", phrase_text)
                continue
            if self._is_relation_phrase(phrase_text):
                append_phrase("relation_detail", phrase_text)
                continue
            if self._looks_like_entity_phrase(phrase_text):
                route_entity_phrase(phrase_text)
                continue

            tokens = phrase_text.split()
            if tokens and self._looks_like_active_predicate(tokens[-1]):
                append_phrase("subject_action", phrase_text)
            else:
                append_phrase("relation_detail", phrase_text)

        for phrase_text in attr_units or []:
            if self._looks_like_clean_scene_context_phrase(phrase_text):
                append_phrase("scene_context", phrase_text)
            elif self._is_relation_phrase(phrase_text):
                append_phrase("relation_detail", phrase_text)
            elif self._looks_like_entity_phrase(phrase_text):
                route_entity_phrase(phrase_text)
            else:
                append_phrase("relation_detail", phrase_text)

        for phrase_text in scene_units or []:
            if self._looks_like_clean_scene_context_phrase(phrase_text):
                append_phrase("scene_context", phrase_text)
            elif self._looks_like_entity_phrase(phrase_text):
                route_entity_phrase(phrase_text)
            elif self._is_relation_phrase(phrase_text):
                append_phrase("relation_detail", phrase_text)
            else:
                subject_phrase = self._split_relation_head_tail(phrase_text)[2]
                if subject_phrase:
                    route_entity_phrase(subject_phrase, preferred_slot="subject_entity")
                append_phrase("relation_detail", phrase_text)

        slot_units = [""] * self.max_phrase_slots
        ordered_units: List[str] = []
        bucket_offsets = {
            slot_type: 0
            for slot_type in self._TYPED_RICH_SLOT_FAMILY_ORDER
        }

        for slot_spec in self.phrase_slot_type_specs:
            slot_idx = int(slot_spec.get("slot_id", 0))
            slot_type = str(slot_spec.get("slot_type_family", slot_spec.get("slot_type", "relation_detail")))
            bucket = slot_buckets.get(slot_type, [])
            bucket_offset = bucket_offsets.get(slot_type, 0)
            if bucket_offset >= len(bucket):
                continue
            phrase_text = bucket[bucket_offset]
            bucket_offsets[slot_type] = bucket_offset + 1
            slot_units[slot_idx] = phrase_text
            ordered_units.append(phrase_text)

        for slot_type in self._TYPED_RICH_SLOT_FAMILY_ORDER:
            bucket = slot_buckets.get(slot_type, [])
            bucket_offset = bucket_offsets.get(slot_type, 0)
            if bucket_offset < len(bucket):
                ordered_units.extend(bucket[bucket_offset:])

        return ordered_units, slot_units

    def _build_typed_rich_semantic_phrase_targets(
        self,
        phrase_units: List[str],
        attr_units: Optional[List[str]] = None,
        scene_units: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        slot_buckets: Dict[str, List[str]] = {
            slot_type: []
            for slot_type in self._TYPED_RICH_SEMANTIC_SLOT_FAMILY_ORDER
        }
        seen_phrases = set()

        def append_phrase(slot_type: str, phrase_text: Optional[str]) -> None:
            normalized = self._normalize_phrase_unit(phrase_text or "")
            if not normalized:
                return
            self._append_bucket_phrase(slot_buckets[slot_type], normalized, seen_phrases)

        def route_detail_phrase(phrase_text: Optional[str], *, prefer_modifier: bool = False) -> None:
            normalized = self._normalize_phrase_unit(phrase_text or "")
            if not normalized:
                return
            if self._looks_like_clean_scene_context_phrase(normalized):
                append_phrase("scene_context", normalized)
                return
            if prefer_modifier:
                append_phrase("entity_modifier", normalized)
                return
            if self._looks_like_instrument_phrase(normalized):
                append_phrase("instrument_detail", normalized)
                return
            if self._is_relation_phrase(normalized):
                append_phrase("relation_detail", normalized)
                return
            append_phrase("entity_modifier", normalized)

        def maybe_route_entity_relation_phrase(phrase_text: Optional[str]) -> bool:
            normalized = self._normalize_phrase_unit(phrase_text or "")
            if not normalized:
                return False

            base_entity, detail_phrase = self._split_entity_core_and_detail(normalized)
            if not detail_phrase or base_entity == normalized:
                return False
            if not self._looks_like_entity_phrase(base_entity):
                return False

            preferred_slot = (
                "subject_entity"
                if self._prefers_subject_entity_slot(base_entity)
                else "object_entity"
            )
            route_entity_phrase(
                base_entity,
                preferred_slot=preferred_slot,
                allow_detail_split=False,
            )
            route_detail_phrase(detail_phrase, prefer_modifier=True)
            return True

        def route_entity_phrase(
            phrase_text: Optional[str],
            preferred_slot: Optional[str] = None,
            *,
            allow_detail_split: bool = True,
        ) -> None:
            normalized = self._normalize_phrase_unit(phrase_text or "")
            if not normalized:
                return

            base_entity = normalized
            detail_phrase = None
            if allow_detail_split:
                base_entity, detail_phrase = self._split_entity_core_and_detail(normalized)

            if self._looks_like_entity_phrase(base_entity):
                slot_type = preferred_slot
                if slot_type is None:
                    slot_type = "subject_entity" if self._prefers_subject_entity_slot(base_entity) else "object_entity"
                append_phrase(slot_type, base_entity)
            elif self._looks_like_entity_phrase(normalized):
                slot_type = preferred_slot
                if slot_type is None:
                    slot_type = "subject_entity" if self._prefers_subject_entity_slot(normalized) else "object_entity"
                append_phrase(slot_type, normalized)

            if detail_phrase:
                route_detail_phrase(detail_phrase, prefer_modifier=True)

        for phrase_text in phrase_units:
            subject_text, predicate_text = self._split_copula_predicate(phrase_text)
            predicate_head = predicate_text.split()[0] if predicate_text and predicate_text.split() else ""
            subject_phrase = self._extract_subject_phrase(subject_text)

            if subject_text is not None and predicate_text is not None:
                if predicate_head in self._RELATION_HEAD_TOKENS:
                    route_detail_phrase(phrase_text)
                    route_entity_phrase(subject_phrase, preferred_slot="subject_entity")
                    continue
                if self._looks_like_active_predicate(predicate_head):
                    append_phrase("subject_action", phrase_text)
                    route_entity_phrase(subject_phrase, preferred_slot="subject_entity")
                    continue
                if self._looks_like_passive_predicate(predicate_head):
                    sanitized_phrase = self._sanitize_passive_phrase(phrase_text)
                    if not sanitized_phrase:
                        continue
                    append_phrase("object_passive", sanitized_phrase)
                    route_entity_phrase(subject_phrase, preferred_slot="object_entity")
                    continue
                route_entity_phrase(subject_phrase, preferred_slot="subject_entity")
                route_detail_phrase(phrase_text)
                continue

            clause_subject, clause_action, clause_object, clause_detail = self._split_noncopular_clause(phrase_text)
            if clause_subject and clause_action:
                append_phrase("subject_action", clause_action)
                route_entity_phrase(clause_subject, preferred_slot="subject_entity")
                if clause_object:
                    route_entity_phrase(clause_object, preferred_slot="object_entity")
                if clause_detail:
                    route_detail_phrase(clause_detail)
                continue

            if maybe_route_entity_relation_phrase(phrase_text):
                continue
            if self._looks_like_clean_scene_context_phrase(phrase_text):
                append_phrase("scene_context", phrase_text)
                continue
            if self._looks_like_instrument_phrase(phrase_text):
                append_phrase("instrument_detail", phrase_text)
                continue
            if self._is_relation_phrase(phrase_text):
                append_phrase("relation_detail", phrase_text)
                continue
            if self._looks_like_entity_phrase(phrase_text):
                route_entity_phrase(phrase_text)
                continue

            tokens = phrase_text.split()
            if tokens and self._looks_like_active_predicate(tokens[-1]):
                append_phrase("subject_action", phrase_text)
            else:
                append_phrase("entity_modifier", phrase_text)

        for phrase_text in attr_units or []:
            normalized = self._normalize_phrase_unit(phrase_text or "")
            if not normalized:
                continue
            if maybe_route_entity_relation_phrase(normalized):
                continue
            if self._looks_like_clean_scene_context_phrase(normalized):
                append_phrase("scene_context", normalized)
            elif self._looks_like_instrument_phrase(normalized):
                append_phrase("instrument_detail", normalized)
            else:
                append_phrase("entity_modifier", normalized)

        for phrase_text in scene_units or []:
            normalized = self._normalize_phrase_unit(phrase_text or "")
            if not normalized:
                continue
            if maybe_route_entity_relation_phrase(normalized):
                continue
            if self._looks_like_clean_scene_context_phrase(normalized):
                append_phrase("scene_context", normalized)
            elif self._looks_like_instrument_phrase(normalized):
                append_phrase("instrument_detail", normalized)
            elif self._is_relation_phrase(normalized):
                append_phrase("relation_detail", normalized)
            elif self._looks_like_entity_phrase(normalized):
                route_entity_phrase(normalized)
            else:
                append_phrase("entity_modifier", normalized)

        slot_units = [""] * self.max_phrase_slots
        ordered_units: List[str] = []
        bucket_offsets = {
            slot_type: 0
            for slot_type in self._TYPED_RICH_SEMANTIC_SLOT_FAMILY_ORDER
        }

        for slot_spec in self.phrase_slot_type_specs:
            slot_idx = int(slot_spec.get("slot_id", 0))
            slot_type = str(slot_spec.get("slot_type_family", slot_spec.get("slot_type", "entity_modifier")))
            bucket = slot_buckets.get(slot_type, [])
            bucket_offset = bucket_offsets.get(slot_type, 0)
            if bucket_offset >= len(bucket):
                continue
            phrase_text = bucket[bucket_offset]
            bucket_offsets[slot_type] = bucket_offset + 1
            slot_units[slot_idx] = phrase_text
            ordered_units.append(phrase_text)

        for slot_type in self._TYPED_RICH_SEMANTIC_SLOT_FAMILY_ORDER:
            bucket = slot_buckets.get(slot_type, [])
            bucket_offset = bucket_offsets.get(slot_type, 0)
            if bucket_offset < len(bucket):
                ordered_units.extend(bucket[bucket_offset:])

        return ordered_units, slot_units

    def _build_typed_rich_roleaware_phrase_targets(
        self,
        phrase_units: List[str],
        attr_units: Optional[List[str]] = None,
        scene_units: Optional[List[str]] = None,
        bundle_idx: Optional[int] = None,
    ) -> Tuple[List[str], List[str], List[List[str]]]:
        slot_buckets: Dict[str, List[str]] = {
            slot_type: []
            for slot_type in self._TYPED_RICH_ROLEAWARE_SLOT_FAMILY_ORDER
        }
        seen_phrases = set()

        def append_phrase(slot_type: str, phrase_text: Optional[str]) -> None:
            normalized = self._normalize_phrase_unit(phrase_text or "")
            if not normalized:
                return
            self._append_bucket_phrase(slot_buckets[slot_type], normalized, seen_phrases)

        def route_detail_phrase(phrase_text: Optional[str]) -> None:
            normalized = self._normalize_phrase_unit(phrase_text or "")
            if not normalized:
                return
            if self._is_low_information_relation_detail(normalized):
                return
            if self._looks_like_clean_scene_context_phrase(normalized):
                append_phrase("scene_context", normalized)
                return
            if self._looks_like_instrument_phrase(normalized):
                append_phrase("instrument_detail", normalized)
                return
            append_phrase("relation_detail", normalized)

        def route_entity_phrase(
            phrase_text: Optional[str],
            preferred_entity_slot: Optional[str] = None,
            *,
            allow_detail_split: bool = True,
        ) -> None:
            normalized = self._normalize_phrase_unit(phrase_text or "")
            if not normalized:
                return

            core_entity, modifier_phrase, detail_phrase = self._split_entity_core_and_modifier(normalized)
            entity_candidate = core_entity or normalized
            if not self._looks_like_entity_phrase(entity_candidate):
                if self._looks_like_entity_phrase(normalized):
                    entity_candidate = normalized
                    modifier_phrase = None
                    detail_phrase = None
                else:
                    return

            entity_slot = preferred_entity_slot
            if entity_slot is None:
                entity_slot = "subject_entity" if self._prefers_subject_entity_slot(entity_candidate) else "object_entity"
            modifier_slot = "subject_modifier" if entity_slot == "subject_entity" else "object_modifier"

            append_phrase(entity_slot, entity_candidate)
            if modifier_phrase and modifier_phrase != entity_candidate:
                append_phrase(modifier_slot, modifier_phrase)
            if allow_detail_split and detail_phrase:
                route_detail_phrase(detail_phrase)

        def maybe_route_entity_modifier_phrase(phrase_text: Optional[str]) -> bool:
            normalized = self._normalize_phrase_unit(phrase_text or "")
            if not normalized:
                return False
            core_entity, modifier_phrase, detail_phrase = self._split_entity_core_and_modifier(normalized)
            if core_entity == normalized and not modifier_phrase and not detail_phrase:
                return False
            if not self._looks_like_entity_phrase(core_entity):
                return False
            route_entity_phrase(normalized)
            return True

        for phrase_text in phrase_units:
            subject_text, predicate_text = self._split_copula_predicate(phrase_text)
            predicate_head = predicate_text.split()[0] if predicate_text and predicate_text.split() else ""
            subject_phrase = self._extract_subject_phrase(subject_text)

            if subject_text is not None and predicate_text is not None:
                if predicate_head in self._RELATION_HEAD_TOKENS:
                    route_detail_phrase(phrase_text)
                    route_entity_phrase(subject_phrase, preferred_entity_slot="subject_entity")
                    continue
                if self._looks_like_active_predicate(predicate_head):
                    append_phrase("subject_action", phrase_text)
                    route_entity_phrase(subject_phrase, preferred_entity_slot="subject_entity")
                    continue
                if self._looks_like_passive_predicate(predicate_head):
                    sanitized_phrase = self._sanitize_passive_phrase(phrase_text)
                    if not sanitized_phrase:
                        continue
                    append_phrase("object_passive", sanitized_phrase)
                    route_entity_phrase(subject_phrase, preferred_entity_slot="object_entity")
                    continue
                route_entity_phrase(subject_phrase, preferred_entity_slot="subject_entity")
                route_detail_phrase(phrase_text)
                continue

            clause_subject, clause_action, clause_object, clause_detail = self._split_noncopular_clause(phrase_text)
            if clause_subject and clause_action:
                append_phrase("subject_action", clause_action)
                route_entity_phrase(clause_subject, preferred_entity_slot="subject_entity")
                if clause_object:
                    route_entity_phrase(clause_object, preferred_entity_slot="object_entity")
                if clause_detail:
                    route_detail_phrase(clause_detail)
                continue

            if maybe_route_entity_modifier_phrase(phrase_text):
                continue
            if self._looks_like_clean_scene_context_phrase(phrase_text):
                append_phrase("scene_context", phrase_text)
                continue
            if self._looks_like_instrument_phrase(phrase_text):
                append_phrase("instrument_detail", phrase_text)
                continue
            if self._is_relation_phrase(phrase_text):
                route_detail_phrase(phrase_text)
                continue
            if self._looks_like_entity_phrase(phrase_text):
                route_entity_phrase(phrase_text)
                continue

            tokens = phrase_text.split()
            if tokens and self._looks_like_active_predicate(tokens[-1]):
                append_phrase("subject_action", phrase_text)
            else:
                append_phrase("relation_detail", phrase_text)

        for phrase_text in attr_units or []:
            normalized = self._normalize_phrase_unit(phrase_text or "")
            if not normalized:
                continue
            if maybe_route_entity_modifier_phrase(normalized):
                continue
            if self._looks_like_clean_scene_context_phrase(normalized):
                append_phrase("scene_context", normalized)
            elif self._looks_like_instrument_phrase(normalized):
                append_phrase("instrument_detail", normalized)
            elif self._looks_like_entity_phrase(normalized):
                route_entity_phrase(normalized)
            else:
                append_phrase("relation_detail", normalized)

        for phrase_text in scene_units or []:
            normalized = self._normalize_phrase_unit(phrase_text or "")
            if not normalized:
                continue
            if maybe_route_entity_modifier_phrase(normalized):
                continue
            if self._looks_like_clean_scene_context_phrase(normalized):
                append_phrase("scene_context", normalized)
            elif self._looks_like_instrument_phrase(normalized):
                append_phrase("instrument_detail", normalized)
            elif self._looks_like_entity_phrase(normalized):
                route_entity_phrase(normalized)
            else:
                append_phrase("relation_detail", normalized)

        slot_units = [""] * self.max_phrase_slots
        slot_reference_units: List[List[str]] = [[] for _ in range(self.max_phrase_slots)]
        ordered_units: List[str] = []
        bucket_offsets = {
            slot_type: 0
            for slot_type in self._TYPED_RICH_ROLEAWARE_SLOT_FAMILY_ORDER
        }
        resolved_bundle_idx = int(bundle_idx) if bundle_idx is not None and int(bundle_idx) >= 0 else None

        for slot_spec in self.phrase_slot_type_specs:
            slot_idx = int(slot_spec.get("slot_id", 0))
            slot_type = str(slot_spec.get("slot_type_family", slot_spec.get("slot_type", "relation_detail")))
            bucket = slot_buckets.get(slot_type, [])
            slot_reference_units[slot_idx] = list(bucket)
            if not bucket:
                continue
            if resolved_bundle_idx is None:
                bucket_offset = bucket_offsets.get(slot_type, 0)
                if bucket_offset >= len(bucket):
                    continue
                phrase_text = bucket[bucket_offset]
                bucket_offsets[slot_type] = bucket_offset + 1
            else:
                phrase_text = bucket[min(resolved_bundle_idx, len(bucket) - 1)]
            slot_units[slot_idx] = phrase_text
            ordered_units.append(phrase_text)

        if resolved_bundle_idx is None:
            for slot_type in self._TYPED_RICH_ROLEAWARE_SLOT_FAMILY_ORDER:
                bucket = slot_buckets.get(slot_type, [])
                bucket_offset = bucket_offsets.get(slot_type, 0)
                if bucket_offset < len(bucket):
                    ordered_units.extend(bucket[bucket_offset:])

        return ordered_units, slot_units, slot_reference_units

    def _build_family4_compact_phrase_targets(
        self,
        *,
        cap_info: Optional[dict],
        sample_key: str,
        family_bundle_idx: Optional[int] = None,
    ) -> Tuple[List[str], List[str], List[List[str]]]:
        slot_units = [""] * self.max_phrase_slots
        slot_reference_units: List[List[str]] = [[] for _ in range(self.max_phrase_slots)]
        ordered_units: List[str] = []

        for slot_spec in self.phrase_slot_type_specs[: self.max_phrase_slots]:
            slot_idx = int(slot_spec.get("slot_id", 0))
            slot_type = str(slot_spec.get("slot_type_family", slot_spec.get("slot_type", "generic")))
            family_items = self._caption_family_slot_items(cap_info, slot_type)
            if not family_items:
                continue
            if family_bundle_idx is not None and int(family_bundle_idx) >= 0:
                slot_text = self._select_family_phrase_target_by_bundle(
                    family_items,
                    bundle_idx=int(family_bundle_idx),
                )
            else:
                slot_text = self._select_family_phrase_target(
                    family_items,
                    slot_type=slot_type,
                    sample_key=sample_key,
                )
            slot_units[slot_idx] = slot_text
            slot_reference_units[slot_idx] = list(family_items)
            ordered_units.append(slot_text)

        return ordered_units, slot_units, slot_reference_units

    def _prepare_phrase_targets(
        self,
        phrase_units: List[str],
        caption_fallback: str,
        caption_text: Optional[str] = None,
        cap_info: Optional[dict] = None,
        sample_key: str = "",
        family_bundle_idx: Optional[int] = None,
        attr_units: Optional[List[str]] = None,
        scene_units: Optional[List[str]] = None,
        video_phrase_units: Optional[List[str]] = None,
        video_attributes: Optional[List[str]] = None,
        video_scenes: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[str], List[List[str]]]:
        caption_phrase_units = self._derive_caption_phrase_units(caption_text) if self.phrase_target_mode == "slot" else []
        video_phrase_units = (
            self._merge_phrase_unit_sources(video_phrase_units)
            if bool(getattr(self, "phrase_include_video_phrase_units", False))
            else []
        )
        phrase_units = self._merge_phrase_unit_sources(phrase_units, caption_phrase_units, video_phrase_units)
        attr_units = (
            self._merge_phrase_unit_sources(
                attr_units,
                video_attributes if bool(getattr(self, "phrase_include_video_attr_units", False)) else None,
            )
            if bool(getattr(self, "phrase_include_attr_units", False))
            else []
        )
        scene_units = (
            self._merge_phrase_unit_sources(
                scene_units,
                video_scenes if bool(getattr(self, "phrase_include_video_scene_units", False)) else None,
            )
            if bool(getattr(self, "phrase_include_scene_units", False))
            else []
        )

        merged_units = self._merge_phrase_unit_sources(phrase_units, attr_units, scene_units)
        if not merged_units and self.phrase_fallback_to_caption:
            fallback = self._normalize_phrase_unit(caption_fallback)
            if fallback:
                phrase_units = [fallback]
                merged_units = [fallback]

        slot_units = merged_units
        slot_reference_units = [[slot_text] if slot_text else [] for slot_text in slot_units[: self.max_phrase_slots]]
        if self.phrase_target_mode == "slot":
            if self.phrase_slot_schema == "typed":
                phrase_units, slot_units = self._build_typed_phrase_targets(
                    phrase_units=phrase_units,
                    attr_units=attr_units,
                    scene_units=scene_units,
                )
            elif self.phrase_slot_schema == "typed_rich":
                phrase_units, slot_units = self._build_typed_rich_phrase_targets(
                    phrase_units=phrase_units,
                    attr_units=attr_units,
                    scene_units=scene_units,
                )
            elif self.phrase_slot_schema == "typed_rich_semantic":
                phrase_units, slot_units = self._build_typed_rich_semantic_phrase_targets(
                    phrase_units=phrase_units,
                    attr_units=attr_units,
                    scene_units=scene_units,
                )
            elif self.phrase_slot_schema == "typed_rich_roleaware":
                phrase_units, slot_units, slot_reference_units = self._build_typed_rich_roleaware_phrase_targets(
                    phrase_units=phrase_units,
                    attr_units=attr_units,
                    scene_units=scene_units,
                    bundle_idx=family_bundle_idx,
                )
            elif self.phrase_slot_schema == "family4_compact":
                phrase_units, slot_units, slot_reference_units = self._build_family4_compact_phrase_targets(
                    cap_info=cap_info,
                    sample_key=sample_key,
                    family_bundle_idx=family_bundle_idx,
                )
            elif self.phrase_slot_schema == "anchored_sov_scene":
                phrase_units, slot_units, slot_reference_units = self._build_family4_compact_phrase_targets(
                    cap_info=cap_info,
                    sample_key=sample_key,
                    family_bundle_idx=family_bundle_idx,
                )
            else:
                phrase_units = merged_units
        else:
            phrase_units = merged_units

        active_slot_types_tuple = tuple(getattr(self, "phrase_slot_active_slot_types", tuple()) or tuple())
        if self.phrase_target_mode == "slot" and active_slot_types_tuple:
            active_slot_types = set(active_slot_types_tuple)
            filtered_slot_units = [""] * self.max_phrase_slots
            for slot_idx, slot_spec in enumerate(self.phrase_slot_type_specs[: self.max_phrase_slots]):
                slot_type = str(slot_spec.get("slot_type", "")).strip().lower()
                if slot_type in active_slot_types and slot_idx < len(slot_units):
                    filtered_slot_units[slot_idx] = slot_units[slot_idx]
            slot_units = filtered_slot_units
            phrase_units = [slot_text for slot_text in slot_units if slot_text]
            if not phrase_units and self.phrase_fallback_to_caption:
                fallback = self._normalize_phrase_unit(caption_fallback)
                if fallback:
                    phrase_units = [fallback]
            filtered_reference_units: List[List[str]] = [[] for _ in range(self.max_phrase_slots)]
            for slot_idx, slot_text in enumerate(slot_units[: self.max_phrase_slots]):
                if slot_text:
                    filtered_reference_units[slot_idx] = (
                        list(slot_reference_units[slot_idx])
                        if slot_idx < len(slot_reference_units) and slot_reference_units[slot_idx]
                        else [slot_text]
                    )
            slot_reference_units = filtered_reference_units

        while len(slot_reference_units) < self.max_phrase_slots:
            slot_reference_units.append([])

        if not hasattr(self, "phrase_slot_multiref_enable") and not hasattr(self, "phrase_slot_active_slot_types"):
            return phrase_units, slot_units
        return phrase_units, slot_units, slot_reference_units

    @staticmethod
    def _unpack_phrase_target_outputs(
        prepared_outputs,
        slot_units: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[str], List[List[str]]]:
        if isinstance(prepared_outputs, tuple) and len(prepared_outputs) == 3:
            phrase_units, resolved_slot_units, slot_reference_units = prepared_outputs
            return list(phrase_units), list(resolved_slot_units), list(slot_reference_units)
        if isinstance(prepared_outputs, tuple) and len(prepared_outputs) == 2:
            phrase_units, resolved_slot_units = prepared_outputs
            resolved_slot_units = list(resolved_slot_units)
            slot_reference_units = [
                [slot_text] if slot_text else []
                for slot_text in resolved_slot_units
            ]
            return list(phrase_units), resolved_slot_units, slot_reference_units
        resolved_slot_units = list(slot_units or [])
        return [], resolved_slot_units, [[slot_text] if slot_text else [] for slot_text in resolved_slot_units]

    def _encode_phrase_slot_reference_units(
        self,
        slot_units: List[str],
        slot_reference_units: List[List[str]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_refs = max(1, int(getattr(self, "phrase_slot_multiref_max_refs", 1)))
        phrase_slot_ref_ids = torch.full(
            (self.max_phrase_slots, max_refs, self.phrase_slot_max_len),
            fill_value=0,
            dtype=torch.long,
        )
        phrase_slot_ref_mask = torch.zeros(
            (self.max_phrase_slots, max_refs, self.phrase_slot_max_len),
            dtype=torch.long,
        )
        phrase_slot_ref_valid = torch.zeros((self.max_phrase_slots, max_refs), dtype=torch.bool)

        for slot_idx in range(self.max_phrase_slots):
            reference_units = []
            if slot_idx < len(slot_reference_units):
                reference_units = self._normalize_phrase_units(slot_reference_units[slot_idx])
            if not reference_units:
                slot_text = slot_units[slot_idx] if slot_idx < len(slot_units) else ""
                if slot_text:
                    reference_units = [slot_text]
            for ref_idx, ref_text in enumerate(reference_units[:max_refs]):
                ref_ids, ref_mask = self._encode_phrase_text(ref_text, self.phrase_slot_max_len)
                phrase_slot_ref_ids[slot_idx, ref_idx] = ref_ids
                phrase_slot_ref_mask[slot_idx, ref_idx] = ref_mask
                phrase_slot_ref_valid[slot_idx, ref_idx] = bool(ref_text)

        return phrase_slot_ref_ids, phrase_slot_ref_mask, phrase_slot_ref_valid

    def _encode_phrase_units(
        self,
        phrase_units: List[str],
        caption_fallback: str,
        caption_text: Optional[str] = None,
        cap_info: Optional[dict] = None,
        sample_key: str = "",
        family_bundle_idx: Optional[int] = None,
        attr_units: Optional[List[str]] = None,
        scene_units: Optional[List[str]] = None,
        video_phrase_units: Optional[List[str]] = None,
        video_attributes: Optional[List[str]] = None,
        video_scenes: Optional[List[str]] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        phrase_units, slot_units, slot_reference_units = self._unpack_phrase_target_outputs(
            self._prepare_phrase_targets(
                phrase_units=phrase_units,
                caption_fallback=caption_fallback,
                caption_text=caption_text,
                cap_info=cap_info,
                sample_key=sample_key,
                family_bundle_idx=family_bundle_idx,
                attr_units=attr_units,
                scene_units=scene_units,
                video_phrase_units=video_phrase_units,
                video_attributes=video_attributes,
                video_scenes=video_scenes,
            )
        )

        phrase_text = " ; ".join(phrase_units) if phrase_units else ""
        phrase_ids, phrase_mask = self._encode_phrase_text(phrase_text, self.phrase_max_len)

        phrase_slot_ids = torch.full(
            (self.max_phrase_slots, self.phrase_slot_max_len),
            fill_value=0,
            dtype=phrase_ids.dtype,
        )
        phrase_slot_mask = torch.zeros(
            (self.max_phrase_slots, self.phrase_slot_max_len),
            dtype=phrase_mask.dtype,
        )
        phrase_slot_valid = torch.zeros(self.max_phrase_slots, dtype=torch.bool)

        for slot_idx in range(self.max_phrase_slots):
            slot_text = slot_units[slot_idx] if slot_idx < len(slot_units) else ""
            slot_ids, slot_mask = self._encode_phrase_text(slot_text, self.phrase_slot_max_len)
            phrase_slot_ids[slot_idx] = slot_ids
            phrase_slot_mask[slot_idx] = slot_mask
            phrase_slot_valid[slot_idx] = bool(slot_text)

        phrase_slot_ref_ids = None
        phrase_slot_ref_mask = None
        phrase_slot_ref_valid = None
        if self.phrase_slot_multiref_enable:
            phrase_slot_ref_ids, phrase_slot_ref_mask, phrase_slot_ref_valid = self._encode_phrase_slot_reference_units(
                slot_units=slot_units,
                slot_reference_units=slot_reference_units,
            )

        return (
            phrase_ids,
            phrase_mask,
            phrase_slot_ids,
            phrase_slot_mask,
            phrase_slot_valid,
            phrase_slot_ref_ids,
            phrase_slot_ref_mask,
            phrase_slot_ref_valid,
        )

    def get_phrase_slot_reweight_stats(
        self,
        *,
        power: float = 0.5,
        min_weight: float = 1.0,
        max_weight: float = 4.0,
    ) -> Dict[str, Any]:
        cache_key = (float(power), float(min_weight), float(max_weight))
        cached = self._phrase_slot_reweight_stats_cache.get(cache_key)
        if cached is not None:
            return cached

        if self.phrase_target_mode != "slot":
            stats = {
                "enabled": False,
                "reason": "phrase_target_mode != slot",
                "slot_specs": list(self.phrase_slot_type_specs),
                "slot_valid_counts": [0 for _ in range(self.max_phrase_slots)],
                "slot_valid_ratios": [0.0 for _ in range(self.max_phrase_slots)],
                "slot_token_weights": [1.0 for _ in range(self.max_phrase_slots)],
                "slot_presence_pos_weights": [1.0 for _ in range(self.max_phrase_slots)],
                "sample_count": 0,
            }
            self._phrase_slot_reweight_stats_cache[cache_key] = stats
            return stats

        sample_list = getattr(self.base_dataset, "data_list", None)
        if sample_list is None:
            raise AttributeError("base_dataset must expose data_list for phrase slot reweight statistics.")

        total_samples = len(sample_list)
        slot_valid_counts = [0 for _ in range(self.max_phrase_slots)]

        for vid, caption_label, sen_id in sample_list:
            vid_str = str(vid)
            cap_info = self._video_caption_map(vid_str).get(str(sen_id))
            video_info = self.structured_videos.get(vid_str, {})
            phrase_units = cap_info.get("phrase_units", []) if isinstance(cap_info, dict) else []
            attr_units = cap_info.get("attr_units", []) if isinstance(cap_info, dict) else []
            scene_units = cap_info.get("scene_units", []) if isinstance(cap_info, dict) else []
            caption_text = (
                cap_info.get("caption", str(caption_label))
                if isinstance(cap_info, dict)
                else str(caption_label)
            )
            _, slot_units, _ = self._unpack_phrase_target_outputs(
                self._prepare_phrase_targets(
                    phrase_units=phrase_units,
                    caption_fallback=caption_text,
                    caption_text=caption_text,
                    cap_info=cap_info,
                    attr_units=attr_units,
                    scene_units=scene_units,
                    video_phrase_units=self._video_phrase_bank(vid_str),
                    video_attributes=video_info.get("attributes", None),
                    video_scenes=video_info.get("scenes", None),
                )
            )
            for slot_idx in range(self.max_phrase_slots):
                slot_text = slot_units[slot_idx] if slot_idx < len(slot_units) else ""
                if slot_text:
                    slot_valid_counts[slot_idx] += 1

        slot_valid_ratios = [
            float(count) / float(max(1, total_samples))
            for count in slot_valid_counts
        ]
        active_counts = [count for count in slot_valid_counts if count > 0]

        raw_weights: List[float] = []
        if active_counts:
            ref_count = float(max(active_counts))
            for count in slot_valid_counts:
                if count <= 0:
                    weight = float(max_weight)
                else:
                    weight = math.pow(ref_count / float(count), float(power))
                weight = min(float(max_weight), max(float(min_weight), float(weight)))
                raw_weights.append(weight)
            active_weight_mean = sum(
                raw_weights[idx]
                for idx, count in enumerate(slot_valid_counts)
                if count > 0
            ) / float(len(active_counts))
        else:
            raw_weights = [1.0 for _ in range(self.max_phrase_slots)]
            active_weight_mean = 1.0

        norm = max(1e-8, float(active_weight_mean))
        slot_token_weights = [float(weight) / norm for weight in raw_weights]
        slot_presence_pos_weights = list(slot_token_weights)

        stats = {
            "enabled": True,
            "reason": "ok",
            "sample_count": int(total_samples),
            "power": float(power),
            "min_weight": float(min_weight),
            "max_weight": float(max_weight),
            "slot_specs": list(self.phrase_slot_type_specs),
            "slot_valid_counts": [int(count) for count in slot_valid_counts],
            "slot_valid_ratios": slot_valid_ratios,
            "slot_token_weights": slot_token_weights,
            "slot_presence_pos_weights": slot_presence_pos_weights,
        }
        self._phrase_slot_reweight_stats_cache[cache_key] = stats
        return stats

    def __getitem__(self, idx: int):
        family_bundle_idx = None
        base_idx = int(idx)
        if self._sample_expansion_index is not None:
            base_idx, family_bundle_idx = self._sample_expansion_index[base_idx]
        base = self.base_dataset[base_idx]
        if len(base) < 7:
            raise ValueError("base_dataset item must have at least 7 fields")

        vid_feat, vid_mask, caption_ids, caption_mask, caption_label, vid, sen_id = base[:7]
        vid = str(vid)
        sen_id_str = str(sen_id)

        video_info = self.structured_videos.get(vid, {})
        entities = video_info.get("entities", [])
        actions = video_info.get("actions", [])
        attributes = video_info.get("attributes", None)
        scenes = video_info.get("scenes", None)
        attr_state = str(video_info.get("attributes_state", "known" if isinstance(attributes, list) else "unknown"))
        scene_state = str(video_info.get("scenes_state", "known" if isinstance(scenes, list) else "unknown"))

        cap_info = self._video_caption_map(vid).get(sen_id_str)
        phrase_units = []
        attr_units = []
        scene_units = []
        if isinstance(cap_info, dict):
            phrase_units = cap_info.get("phrase_units", []) or []
            attr_units = cap_info.get("attr_units", []) or []
            scene_units = cap_info.get("scene_units", []) or []
        caption_text = cap_info.get("caption", str(caption_label)) if isinstance(cap_info, dict) else str(caption_label)
        video_phrase_units = self._video_phrase_bank(vid)

        entity_target = self._multi_hot(entities, self.entity_to_idx, self.entity_dim)
        action_target = self._multi_hot(actions, self.action_to_idx, self.action_dim)
        attr_target = self._multi_hot(attributes if isinstance(attributes, list) else [], self.attribute_to_idx, self.attribute_dim)
        scene_target = self._multi_hot(scenes if isinstance(scenes, list) else [], self.scene_to_idx, self.scene_dim)
        caption_entity_target = self._multi_hot(self._caption_entity_items(cap_info), self.entity_to_idx, self.entity_dim)
        caption_action_target = self._multi_hot(self._caption_action_items(cap_info), self.action_to_idx, self.action_dim)
        caption_attr_target = self._multi_hot(self._caption_attribute_items(cap_info), self.attribute_to_idx, self.attribute_dim)
        caption_scene_target = self._multi_hot(self._caption_scene_items(cap_info), self.scene_to_idx, self.scene_dim)
        attr_known_mask = torch.tensor(1.0 if attr_state == "known" else 0.0, dtype=torch.float32)
        scene_known_mask = torch.tensor(1.0 if scene_state == "known" else 0.0, dtype=torch.float32)
        (
            phrase_ids,
            phrase_mask,
            phrase_slot_ids,
            phrase_slot_mask,
            phrase_slot_valid,
            phrase_slot_ref_ids,
            phrase_slot_ref_mask,
            phrase_slot_ref_valid,
        ) = self._encode_phrase_units(
            phrase_units,
            caption_text,
            caption_text=caption_text,
            cap_info=cap_info,
            sample_key=(
                f"{vid}::{sen_id_str}::bundle={int(family_bundle_idx)}"
                if family_bundle_idx is not None
                else f"{vid}::{sen_id_str}"
            ),
            family_bundle_idx=family_bundle_idx,
            attr_units=attr_units,
            scene_units=scene_units,
            video_phrase_units=video_phrase_units,
            video_attributes=attributes if isinstance(attributes, list) else None,
            video_scenes=scenes if isinstance(scenes, list) else None,
        )
        sample = (
            vid_feat,
            vid_mask,
            caption_ids,
            caption_mask,
            caption_label,
            vid,
            sen_id,
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
        )
        if self.phrase_slot_multiref_enable:
            sample_aux = {
                "phrase_slot_ref_ids": phrase_slot_ref_ids,
                "phrase_slot_ref_mask": phrase_slot_ref_mask,
                "phrase_slot_ref_valid": phrase_slot_ref_valid,
            }
            return sample + (sample_aux,)

        return sample
