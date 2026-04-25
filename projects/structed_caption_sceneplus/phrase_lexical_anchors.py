from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from dataloaders.dataset_structured_caption import StructuredCaptionDataset


SPECIAL_TOKEN_IDS = {0, 49406, 49407}
SLOT_FAMILY_FIELDS = {
    "subject_entity": "family_subject_phrases",
    "object_entity": "family_object_phrases",
    "subject_action": "family_action_phrases",
    "scene_context": "family_scene_phrases",
}
DEFAULT_DECODE_STOPWORDS = (
    "in",
    "on",
    "with",
    "and",
    "or",
    "her",
    "his",
    "their",
    "its",
    "him",
    "them",
    "there",
)


def safe_encode_phrase_token_sequence(tokenizer, text: str) -> List[int]:
    try:
        token_ids = tokenizer.encode(str(text), add_special_tokens=False)
    except TypeError:
        token_ids = tokenizer.encode(str(text))
    if not isinstance(token_ids, list):
        token_ids = list(token_ids)
    return [int(token_id) for token_id in token_ids if int(token_id) not in SPECIAL_TOKEN_IDS]


def build_source_label_token_banks(
    tokenizer,
    *,
    entity_vocab: Sequence[str],
    action_vocab: Sequence[str],
    attribute_vocab: Sequence[str],
    scene_vocab: Sequence[str],
) -> Dict[str, List[List[int]]]:
    def _encode_many(vocab_items: Sequence[str]) -> List[List[int]]:
        return [safe_encode_phrase_token_sequence(tokenizer, str(item)) for item in vocab_items]

    return {
        "entity": _encode_many(entity_vocab),
        "action": _encode_many(action_vocab),
        "attribute": _encode_many(attribute_vocab),
        "scene": _encode_many(scene_vocab),
    }


def build_decode_stopword_token_ids(
    tokenizer,
    stopwords: Optional[Sequence[str]] = None,
) -> List[int]:
    token_ids: List[int] = []
    seen = set()
    for word in stopwords or DEFAULT_DECODE_STOPWORDS:
        for token_id in safe_encode_phrase_token_sequence(tokenizer, str(word)):
            if token_id in seen:
                continue
            seen.add(token_id)
            token_ids.append(int(token_id))
    return token_ids


def iter_caption_infos(
    structured_payload_or_videos: Mapping[str, Any],
) -> Iterable[Mapping[str, Any]]:
    videos = structured_payload_or_videos
    if isinstance(structured_payload_or_videos, Mapping) and "videos" in structured_payload_or_videos:
        maybe_videos = structured_payload_or_videos.get("videos")
        if isinstance(maybe_videos, Mapping):
            videos = maybe_videos

    if not isinstance(videos, Mapping):
        return

    for video_info in videos.values():
        if not isinstance(video_info, Mapping):
            continue
        captions = video_info.get("captions")
        if isinstance(captions, Mapping):
            caption_items = captions.values()
        elif isinstance(captions, list):
            caption_items = captions
        else:
            caption_items = []
        for cap_info in caption_items:
            if isinstance(cap_info, Mapping):
                yield cap_info


def build_slot_family_token_priors(
    tokenizer,
    *,
    structured_payload_or_videos: Mapping[str, Any],
    phrase_slot_schema: str,
    max_phrase_slots: int,
    stopword_token_ids: Optional[Sequence[int]] = None,
    topk_tokens: int = 64,
    min_count: int = 2,
) -> Tuple[Dict[str, List[int]], Dict[str, List[float]]]:
    slot_specs = StructuredCaptionDataset.get_phrase_slot_type_specs(
        max_phrase_slots=max_phrase_slots,
        phrase_slot_schema=phrase_slot_schema,
    )
    selected_slot_types: List[str] = []
    for slot_spec in slot_specs[: max(1, int(max_phrase_slots))]:
        slot_type = str(slot_spec.get("slot_type_family", slot_spec.get("slot_type", ""))).strip().lower()
        if slot_type and slot_type in SLOT_FAMILY_FIELDS and slot_type not in selected_slot_types:
            selected_slot_types.append(slot_type)

    stopword_id_set = {int(token_id) for token_id in (stopword_token_ids or [])}
    counters: MutableMapping[str, Counter] = {
        slot_type: Counter()
        for slot_type in selected_slot_types
    }

    for cap_info in iter_caption_infos(structured_payload_or_videos):
        for slot_type in selected_slot_types:
            field_name = SLOT_FAMILY_FIELDS.get(slot_type)
            if not field_name:
                continue
            phrases = cap_info.get(field_name)
            if not isinstance(phrases, list):
                continue
            for phrase in phrases:
                token_ids = {
                    token_id
                    for token_id in safe_encode_phrase_token_sequence(tokenizer, str(phrase))
                    if token_id not in stopword_id_set
                }
                if not token_ids:
                    continue
                for token_id in token_ids:
                    counters[slot_type][int(token_id)] += 1

    slot_family_token_ids: Dict[str, List[int]] = {}
    slot_family_token_weights: Dict[str, List[float]] = {}
    for slot_type, counter in counters.items():
        rows = [
            (int(token_id), int(count))
            for token_id, count in counter.items()
            if int(count) >= max(1, int(min_count))
        ]
        rows.sort(key=lambda item: (-item[1], item[0]))
        if topk_tokens > 0:
            rows = rows[: int(topk_tokens)]
        total = float(sum(count for _, count in rows))
        slot_family_token_ids[slot_type] = [token_id for token_id, _ in rows]
        if total <= 0.0:
            slot_family_token_weights[slot_type] = []
        else:
            slot_family_token_weights[slot_type] = [float(count) / total for _, count in rows]
    return slot_family_token_ids, slot_family_token_weights


def build_phrase_lexical_anchor_kwargs(
    tokenizer,
    *,
    structured_payload_or_videos: Mapping[str, Any],
    entity_vocab: Sequence[str],
    action_vocab: Sequence[str],
    attribute_vocab: Sequence[str],
    scene_vocab: Sequence[str],
    phrase_slot_schema: str,
    max_phrase_slots: int,
    family_topk_tokens: int = 64,
    family_min_count: int = 2,
) -> Dict[str, Any]:
    source_banks = build_source_label_token_banks(
        tokenizer,
        entity_vocab=entity_vocab,
        action_vocab=action_vocab,
        attribute_vocab=attribute_vocab,
        scene_vocab=scene_vocab,
    )
    stopword_token_ids = build_decode_stopword_token_ids(tokenizer)
    slot_family_token_ids, slot_family_token_weights = build_slot_family_token_priors(
        tokenizer,
        structured_payload_or_videos=structured_payload_or_videos,
        phrase_slot_schema=phrase_slot_schema,
        max_phrase_slots=max_phrase_slots,
        stopword_token_ids=stopword_token_ids,
        topk_tokens=family_topk_tokens,
        min_count=family_min_count,
    )
    return {
        "entity_label_token_ids": source_banks["entity"],
        "action_label_token_ids": source_banks["action"],
        "attribute_label_token_ids": source_banks["attribute"],
        "scene_label_token_ids": source_banks["scene"],
        "slot_family_anchor_token_ids": slot_family_token_ids,
        "slot_family_anchor_token_weights": slot_family_token_weights,
        "decode_stopword_token_ids": stopword_token_ids,
    }
