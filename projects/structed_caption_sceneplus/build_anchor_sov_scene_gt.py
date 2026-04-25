#!/usr/bin/env python
"""Build subject-action/object/scene anchored caption phrase supervision.

This script keeps the compact structured GT vocabularies intact and rewrites
only sentence-level phrase supervision. It can reuse the processed
multi_stage_caption API output as a conservative source of slot anchors.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


ARTICLE_TOKENS = {
    "a",
    "an",
    "the",
    "some",
    "one",
    "two",
    "three",
    "four",
    "several",
    "many",
    "few",
    "another",
    "his",
    "her",
    "their",
    "its",
    "our",
    "my",
}
SUBJECT_PLURAL_HINTS = {
    "people",
    "men",
    "women",
    "children",
    "kids",
    "boys",
    "girls",
    "players",
    "dogs",
    "cats",
    "horses",
    "birds",
}
PREPOSITION_PATTERNS = (
    "in front of",
    "next to",
    "out of",
    "on top of",
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
    "toward",
    "towards",
    "along",
    "into",
    "onto",
    "from",
    "with",
)
VERB_STOP_TOKENS = {
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "am",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
}
IRREGULAR_GERUNDS = {
    "lie": "lying",
    "die": "dying",
    "tie": "tying",
    "run": "running",
    "swim": "swimming",
    "sit": "sitting",
    "get": "getting",
    "put": "putting",
    "cut": "cutting",
    "hit": "hitting",
    "stop": "stopping",
    "begin": "beginning",
    "shop": "shopping",
}
VERB_ALIASES = {
    "ate": "eat",
    "eats": "eat",
    "eating": "eat",
    "rode": "ride",
    "rides": "ride",
    "riding": "ride",
    "rid": "ride",
    "ran": "run",
    "runs": "run",
    "running": "run",
    "swam": "swim",
    "swims": "swim",
    "swimming": "swim",
    "played": "play",
    "plays": "play",
    "playing": "play",
    "walked": "walk",
    "walks": "walk",
    "walking": "walk",
    "danced": "dance",
    "dances": "dance",
    "dancing": "dance",
    "cooked": "cook",
    "cooks": "cook",
    "cooking": "cook",
    "mixed": "mix",
    "mixes": "mix",
    "mixing": "mix",
    "cutting": "cut",
    "cuts": "cut",
    "jumped": "jump",
    "jumps": "jump",
    "jumping": "jump",
}


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("'", "")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\bit s\b", "its", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def token_list(text: Any) -> List[str]:
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def unique_strings(values: Iterable[Any], *, limit: int = 6) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        text = normalize_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def first_nonempty(*values: Any) -> str:
    for value in values:
        text = normalize_text(value)
        if text:
            return text
    return ""


def strip_trailing_aux_phrase(text: str) -> str:
    phrase = normalize_text(text)
    if not phrase:
        return ""
    phrase = re.sub(
        r"\s+(?:is|are|was|were|be|been|being)\s+(?:being\s+)?[a-z0-9]+(?:ed|en|t|d|eaten|done|made|seen|ridden|played|held|used|cut|put|hit)(?:\s.*)?$",
        "",
        phrase,
    )
    return phrase.strip()


def phrase_before_copula(text: str) -> str:
    phrase = normalize_text(text)
    match = re.search(r"\b(?:is|are|was|were|be|being)\b", phrase)
    if not match:
        return ""
    return phrase[: match.start()].strip()


def be_for_np(np_text: str) -> str:
    tokens = token_list(np_text)
    if not tokens:
        return "is"
    first = tokens[0]
    head = tokens[-1]
    if first in {"two", "three", "four", "several", "many"} or head in SUBJECT_PLURAL_HINTS:
        return "are"
    if head.endswith("s") and head not in {"grass", "glass"} and first not in {"a", "an"}:
        return "are"
    return "is"


def to_lemma(word: str) -> str:
    token = normalize_text(word)
    if not token:
        return ""
    if token in VERB_ALIASES:
        return VERB_ALIASES[token]
    if token.endswith("ing") and len(token) > 5:
        stem = token[:-3]
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        if stem.endswith("mak"):
            return stem + "e"
        return stem
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    return token


def to_gerund(verb: str) -> str:
    lemma = to_lemma(verb)
    if not lemma:
        return ""
    if lemma in IRREGULAR_GERUNDS:
        return IRREGULAR_GERUNDS[lemma]
    if lemma.endswith("ing"):
        return lemma
    if lemma.endswith("ie") and len(lemma) > 3:
        return lemma[:-2] + "ying"
    if lemma.endswith("e") and not lemma.endswith(("ee", "ye", "oe")) and len(lemma) > 3:
        return lemma[:-1] + "ing"
    if (
        len(lemma) >= 3
        and lemma[-1] not in "aeiouyw"
        and lemma[-2] in "aeiou"
        and lemma[-3] not in "aeiou"
        and lemma not in {"play", "show", "snow"}
    ):
        return lemma + lemma[-1] + "ing"
    return lemma + "ing"


def action_head(multistage: Optional[Mapping[str, Any]], caption: str, phrase_units: Sequence[str]) -> str:
    if isinstance(multistage, Mapping):
        words = multistage.get("action_words")
        if isinstance(words, list):
            for word in words:
                lemma = to_lemma(str(word))
                if lemma:
                    return lemma
        action = normalize_text(multistage.get("subject_action", ""))
        for word in token_list(action):
            if word not in VERB_STOP_TOKENS and word not in ARTICLE_TOKENS:
                return to_lemma(word)
    for phrase in phrase_units:
        normalized = normalize_text(phrase)
        match = re.search(r"\b(?:is|are|was|were)\s+([a-z0-9]+)", normalized)
        if match:
            return to_lemma(match.group(1))
    for word in token_list(caption):
        if word not in VERB_STOP_TOKENS and word not in ARTICLE_TOKENS:
            lemma = to_lemma(word)
            if lemma in VERB_ALIASES.values():
                return lemma
    return ""


def find_np_in_caption(caption: str, anchor: str, *, max_left: int = 3) -> str:
    words = token_list(caption)
    anchor_tokens = token_list(anchor)
    if not words or not anchor_tokens:
        return normalize_text(anchor)
    span_len = len(anchor_tokens)
    for idx in range(0, len(words) - span_len + 1):
        if words[idx : idx + span_len] != anchor_tokens:
            continue
        start = idx
        left_bound = max(0, idx - max_left)
        for left in range(idx - 1, left_bound - 1, -1):
            if words[left] in ARTICLE_TOKENS:
                start = left
                break
            if idx - left <= 2 and words[left] not in VERB_STOP_TOKENS:
                start = left
        return " ".join(words[start : idx + span_len]).strip()
    return normalize_text(anchor)


def object_np_candidates(
    caption: str,
    object_anchor: str,
    phrase_units: Sequence[str],
    video_fallback: str = "",
) -> List[str]:
    candidates: List[str] = []
    if object_anchor:
        candidates.append(find_np_in_caption(caption, object_anchor, max_left=4))
    for phrase in phrase_units:
        stripped = strip_trailing_aux_phrase(phrase)
        normalized_phrase = normalize_text(phrase)
        passive_like = bool(re.search(r"\b(?:is|are|was|were|being|been)\b", normalized_phrase)) and stripped != normalized_phrase
        if stripped and (
            (object_anchor and (object_anchor in stripped.split() or object_anchor in stripped))
            or (not object_anchor and passive_like)
        ):
            candidates.append(stripped)
    if video_fallback:
        candidates.append(video_fallback)
    if object_anchor:
        candidates.append(object_anchor)
    return unique_strings(candidates, limit=4)


def subject_np_candidates(
    caption: str,
    subject_anchor: str,
    phrase_units: Sequence[str],
    video_fallback: str = "",
) -> List[str]:
    candidates: List[str] = []
    for phrase in phrase_units:
        before = phrase_before_copula(phrase)
        if before and (not subject_anchor or subject_anchor in before.split() or subject_anchor in before):
            candidates.append(before)
    if subject_anchor:
        candidates.append(find_np_in_caption(caption, subject_anchor, max_left=4))
    if video_fallback:
        candidates.append(video_fallback)
    if subject_anchor:
        candidates.append(subject_anchor)
    return unique_strings(candidates, limit=4)


def subject_action_from_phrase(
    subject_np: str,
    caption: str,
    phrase_units: Sequence[str],
    action_lemma: str,
) -> str:
    subject_tokens = token_list(subject_np)
    for phrase in phrase_units:
        normalized = normalize_text(phrase)
        if not normalized:
            continue
        phrase_tokens = normalized.split()
        if subject_tokens and phrase_tokens[: len(subject_tokens)] == subject_tokens:
            if re.search(r"\b(?:is|are|was|were)\b", normalized):
                return normalized
    if subject_np and action_lemma:
        return f"{subject_np} {be_for_np(subject_np)} {to_gerund(action_lemma)}"

    caption_norm = normalize_text(caption)
    if subject_np and caption_norm.startswith(subject_np):
        words = caption_norm.split()
        subject_len = len(subject_np.split())
        tail = []
        for word in words[subject_len:]:
            if word in ARTICLE_TOKENS or word in PREPOSITION_PATTERNS:
                break
            tail.append(word)
            if len(tail) >= 3:
                break
        if tail:
            return " ".join(subject_np.split() + tail)
    return ""


def extract_scene_phrases(caption: str, scene_context: str = "", video_fallback: str = "") -> List[str]:
    candidates: List[str] = []
    if scene_context:
        candidates.append(scene_context)
    normalized = normalize_text(caption)
    words = normalized.split()
    for idx in range(len(words)):
        for prep in PREPOSITION_PATTERNS:
            prep_words = prep.split()
            if words[idx : idx + len(prep_words)] != prep_words:
                continue
            end = min(len(words), idx + len(prep_words) + 6)
            candidates.append(" ".join(words[idx:end]))
    if video_fallback:
        candidates.append(video_fallback)
    cleaned = []
    for item in candidates:
        text = scene_with_preposition(item)
        if not text or len(text.split()) > 9:
            continue
        if text in ARTICLE_TOKENS:
            continue
        cleaned.append(text)
    return unique_strings(cleaned, limit=4)


def relation_phrase(subject_action: str, object_np: str, scene_phrase: str, action_lemma: str) -> str:
    gerund = ""
    if subject_action:
        match = re.search(r"\b(?:is|are|was|were)\s+(.+)$", subject_action)
        if match:
            gerund = match.group(1).split()[0]
    if not gerund and action_lemma:
        gerund = to_gerund(action_lemma)
    parts = []
    if gerund:
        parts.append(gerund)
    if object_np:
        parts.append(object_np)
    if scene_phrase and not object_np:
        parts.append(scene_phrase)
    text = " ".join(parts).strip()
    if text == subject_action or text == object_np:
        return ""
    return normalize_text(text)


def scene_with_preposition(text: str) -> str:
    phrase = normalize_text(text)
    if not phrase:
        return ""
    first = phrase.split()[0]
    prep_heads = {prep.split()[0] for prep in PREPOSITION_PATTERNS}
    if first in prep_heads:
        return phrase
    words = set(phrase.split())
    on_scene_words = {
        "road",
        "street",
        "track",
        "field",
        "court",
        "stage",
        "floor",
        "ground",
        "beach",
        "ice",
        "water",
        "snow",
        "grass",
    }
    in_scene_words = {
        "room",
        "kitchen",
        "bedroom",
        "office",
        "gym",
        "pool",
        "house",
        "building",
        "yard",
        "park",
        "forest",
        "zoo",
        "stadium",
    }
    if words & on_scene_words:
        return f"on the {phrase}"
    if words & in_scene_words:
        return f"in the {phrase}"
    return phrase


def load_multistage_samples(path: Path) -> Dict[Tuple[str, str], Mapping[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples", payload if isinstance(payload, list) else [])
    mapping: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    if not isinstance(samples, list):
        return mapping
    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        vid = str(sample.get("video_id", "")).strip()
        sen_id = str(sample.get("sen_id", "")).strip()
        if vid and sen_id:
            mapping[(vid, sen_id)] = sample
    return mapping


def rank_video_fallbacks(
    compact_videos: Mapping[str, Any],
    multistage_map: Mapping[Tuple[str, str], Mapping[str, Any]],
) -> Dict[str, Dict[str, str]]:
    ranked: Dict[str, Dict[str, str]] = {}
    for vid, video_info in compact_videos.items():
        if not isinstance(video_info, Mapping):
            continue
        counters = {
            "subject": Counter(),
            "object": Counter(),
            "scene": Counter(),
        }
        for cap_info in video_info.get("captions", []) or []:
            if not isinstance(cap_info, Mapping):
                continue
            sample = multistage_map.get((str(vid), str(cap_info.get("sen_id", "")).strip()))
            if not isinstance(sample, Mapping):
                continue
            for key, target in (
                ("subject_entity", "subject"),
                ("object_entity", "object"),
                ("scene_context", "scene"),
            ):
                value = normalize_text(sample.get(key, ""))
                if value:
                    counters[target][value] += 1
        ranked[vid] = {
            name: (counter.most_common(1)[0][0] if counter else "")
            for name, counter in counters.items()
        }
    return ranked


def caption_anchor_payload(
    *,
    vid: str,
    cap_info: Mapping[str, Any],
    multistage: Optional[Mapping[str, Any]],
    video_fallbacks: Mapping[str, str],
    max_refs: int,
) -> Dict[str, Any]:
    caption = normalize_text(cap_info.get("caption", ""))
    phrase_units = unique_strings(cap_info.get("phrase_units", []) or [], limit=8)
    subject_anchor = first_nonempty(
        multistage.get("subject_entity", "") if isinstance(multistage, Mapping) else "",
        video_fallbacks.get("subject", ""),
    )
    object_anchor = first_nonempty(
        multistage.get("object_entity", "") if isinstance(multistage, Mapping) else "",
    )
    scene_anchor = first_nonempty(
        multistage.get("scene_context", "") if isinstance(multistage, Mapping) else "",
        video_fallbacks.get("scene", ""),
    )
    action_lemma = action_head(multistage, caption, phrase_units)

    subject_nps = subject_np_candidates(
        caption,
        subject_anchor,
        phrase_units,
        video_fallback=video_fallbacks.get("subject", ""),
    )
    object_nps = object_np_candidates(
        caption,
        object_anchor,
        phrase_units,
        video_fallback="",
    )
    subject_np = subject_nps[0] if subject_nps else subject_anchor
    object_np = object_nps[0] if object_nps else object_anchor
    subject_action = subject_action_from_phrase(subject_np, caption, phrase_units, action_lemma)
    scene_phrases = extract_scene_phrases(
        caption,
        scene_context=scene_anchor,
        video_fallback=video_fallbacks.get("scene", ""),
    )
    scene_phrase = scene_phrases[0] if scene_phrases else ""
    relation = relation_phrase(subject_action, object_np, scene_phrase, action_lemma)

    object_refs = unique_strings(object_nps + ([f"{object_np} {scene_phrase}"] if object_np and scene_phrase else []), limit=max_refs)
    action_refs = unique_strings([subject_action] + phrase_units, limit=max_refs)
    scene_refs = unique_strings(scene_phrases, limit=max_refs)
    relation_refs = unique_strings([relation], limit=max_refs)
    subject_refs = unique_strings(subject_nps, limit=max_refs)
    ordered_phrase_units = unique_strings(
        [subject_action, object_np, scene_phrase, relation, subject_np],
        limit=5,
    )

    payload = {
        "caption": caption,
        "subject_entities": unique_strings([subject_anchor], limit=2),
        "object_entities": unique_strings([object_anchor], limit=2),
        "subject_action_phrases": action_refs,
        "object_passive_phrases": unique_strings(phrase_units[1:], limit=max_refs),
        "relation_phrases": relation_refs,
        "scene_phrases": scene_refs,
        "scene_units": scene_refs,
        "family_action_phrases": action_refs,
        "family_object_phrases": object_refs,
        "family_scene_phrases": scene_refs,
        "family_relation_phrases": relation_refs,
        "family_subject_phrases": subject_refs,
        "phrase_units": ordered_phrase_units,
        "anchor_sov_slots": {
            "subject_action": subject_action,
            "object_entity": object_np,
            "scene_context": scene_phrase,
            "relation_detail": relation,
            "subject_entity": subject_np,
        },
    }
    return payload


def build_anchor_sov_scene_gt(
    compact_data: Mapping[str, Any],
    multistage_map: Mapping[Tuple[str, str], Mapping[str, Any]],
    *,
    max_refs: int,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    result = deepcopy(compact_data)
    videos = result.get("videos")
    if not isinstance(videos, MutableMapping):
        raise ValueError("Compact GT must contain a videos mapping.")

    video_fallbacks = rank_video_fallbacks(videos, multistage_map)
    stats = {
        "videos_total": 0,
        "captions_total": 0,
        "captions_with_multistage": 0,
        "subject_action_filled": 0,
        "object_entity_filled": 0,
        "scene_context_filled": 0,
        "relation_detail_filled": 0,
        "subject_entity_filled": 0,
    }

    for vid, video_info in videos.items():
        if not isinstance(video_info, MutableMapping):
            continue
        stats["videos_total"] += 1
        captions = video_info.get("captions")
        if not isinstance(captions, list):
            continue
        for cap_info in captions:
            if not isinstance(cap_info, MutableMapping):
                continue
            stats["captions_total"] += 1
            sen_id = str(cap_info.get("sen_id", "")).strip()
            multistage = multistage_map.get((str(vid), sen_id))
            if isinstance(multistage, Mapping):
                stats["captions_with_multistage"] += 1
            payload = caption_anchor_payload(
                vid=str(vid),
                cap_info=cap_info,
                multistage=multistage,
                video_fallbacks=video_fallbacks.get(str(vid), {}),
                max_refs=max_refs,
            )
            cap_info.update(payload)
            slots = payload["anchor_sov_slots"]
            for key in (
                "subject_action",
                "object_entity",
                "scene_context",
                "relation_detail",
                "subject_entity",
            ):
                if normalize_text(slots.get(key, "")):
                    stats[f"{key}_filled"] += 1

    meta = dict(result.get("meta") or {})
    meta["anchor_sov_scene"] = {
        "source": "compact structured GT plus processed multi_stage_caption API slots",
        "slot_order": [
            "subject_action",
            "object_entity",
            "scene_context",
            "relation_detail",
            "subject_entity",
        ],
        "max_reference_phrases_per_slot": int(max_refs),
        **stats,
    }
    result["meta"] = meta
    result.pop("attribute_vocab", None)
    result.pop("scene_vocab", None)
    return result, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact_gt", required=True)
    parser.add_argument("--multistage_gt", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_refs", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compact_path = Path(args.compact_gt)
    multistage_path = Path(args.multistage_gt)
    output_path = Path(args.output_path)
    compact_data = json.loads(compact_path.read_text(encoding="utf-8"))
    multistage_map = load_multistage_samples(multistage_path)
    result, stats = build_anchor_sov_scene_gt(
        compact_data,
        multistage_map,
        max_refs=max(1, int(args.max_refs)),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
