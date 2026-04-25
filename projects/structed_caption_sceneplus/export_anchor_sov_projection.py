#!/usr/bin/env python3
"""Export anchored SOV scene demo slots from caption predictions.

The learned phrase decoder can drift when a slot is weak.  This exporter keeps
the demo explanation anchored to the generated sentence:

  subject_action: parsed from the generated caption
  object_entity: parsed from the generated caption
  scene_context: explicit caption scene, otherwise a plausible scene slot
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_AUX_RE = re.compile(r"\b(?:is|are|was|were|be|being|been)\b", re.I)
_REPEAT_RE = re.compile(r"\b([a-z]+)\b(?:\s+\1\b){3,}", re.I)
_SCENE_PREPS = (
    "in",
    "on",
    "at",
    "near",
    "inside",
    "outside",
    "over",
    "under",
    "through",
    "across",
    "around",
    "by",
    "beside",
    "from",
    "into",
    "onto",
    "along",
)
_SCENE_NOUN_PREP = {
    "kitchen": "in",
    "room": "in",
    "house": "in",
    "restaurant": "in",
    "gym": "in",
    "pool": "in",
    "water": "in",
    "sea": "in",
    "ocean": "in",
    "lake": "in",
    "road": "on",
    "street": "on",
    "floor": "on",
    "stage": "on",
    "field": "on",
    "beach": "on",
    "snow": "on",
    "grass": "on",
}
_PHRASAL_PARTICLES = {
    ("putting", "on"),
    ("takes", "off"),
    ("taking", "off"),
    ("picks", "up"),
    ("picking", "up"),
    ("puts", "down"),
    ("putting", "down"),
}
_OBJECT_PREP_VERBS = {
    ("playing", "with"),
    ("talking", "to"),
    ("speaking", "to"),
    ("looking", "at"),
}
_SIMPLE_VERBS = {
    "rides",
    "ride",
    "plays",
    "play",
    "runs",
    "run",
    "walks",
    "walk",
    "jumps",
    "jump",
    "cuts",
    "cut",
    "throws",
    "throw",
    "talks",
    "talk",
    "dances",
    "dance",
    "swims",
    "swim",
    "drinks",
    "drink",
    "eats",
    "eat",
    "cooks",
    "cook",
    "sings",
    "sing",
    "drives",
    "drive",
}


def _clean_text(text: Any) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .,!?:;\"'")
    return text


def _bad_phrase(text: str) -> bool:
    text = _clean_text(text)
    if not text:
        return True
    if _REPEAT_RE.search(text):
        return True
    toks = text.split()
    if len(toks) > 10:
        return True
    if text in {"something", "someone", "the something", "a a", "the the"}:
        return True
    return False


def _valid_object(text: str, subject: str = "") -> bool:
    text = _clean_text(text)
    subject = _clean_text(subject)
    if _bad_phrase(text):
        return False
    if _AUX_RE.search(text):
        return False
    if subject and text == subject:
        return False
    if text in {"a man", "the man", "a woman", "the woman", "a person"} and subject:
        return False
    return True


def _valid_scene(text: str) -> bool:
    text = _clean_text(text)
    if _bad_phrase(text):
        return False
    return any(text.startswith(prep + " ") for prep in _SCENE_PREPS)


def _normalize_scene(text: str) -> str:
    text = _clean_text(text)
    if not text:
        return ""
    parts = text.split()
    if len(parts) < 2:
        return text
    prep, rest = parts[0], parts[1:]
    noun = rest[-1]
    desired = _SCENE_NOUN_PREP.get(noun)
    if desired:
        prep = desired
    return " ".join([prep] + rest)


def _split_first_scene(rest: str) -> Tuple[str, str]:
    rest = _clean_text(rest)
    if not rest:
        return "", ""
    tokens = rest.split()
    best_idx: Optional[int] = None
    for idx, tok in enumerate(tokens):
        if tok in _SCENE_PREPS:
            best_idx = idx
            break
    if best_idx is None:
        return rest, ""
    if best_idx == 0:
        return "", " ".join(tokens)
    return " ".join(tokens[:best_idx]), " ".join(tokens[best_idx:])


def _strip_leading_object_prep(verb: str, rest: str) -> Tuple[str, str]:
    rest = _clean_text(rest)
    if not rest:
        return "", ""
    tokens = rest.split()
    first = tokens[0]
    if (verb, first) in _PHRASAL_PARTICLES:
        return first, " ".join(tokens[1:])
    if (verb, first) in _OBJECT_PREP_VERBS:
        return "", " ".join(tokens[1:])
    return "", rest


def parse_caption_slots(caption: str) -> Dict[str, str]:
    caption = _clean_text(caption)
    result = {
        "subject_action": "",
        "object_entity": "",
        "scene_context": "",
        "subject": "",
    }
    if not caption:
        return result

    # Common generated-caption form: "a man is riding a bike on the road".
    m = re.match(
        r"^(?P<subject>.+?)\s+(?P<aux>is|are|was|were)\s+(?P<verb>[a-z]+(?:ing|ed)?)\b(?P<rest>.*)$",
        caption,
    )
    if m:
        subject = _clean_text(m.group("subject"))
        aux = _clean_text(m.group("aux"))
        verb = _clean_text(m.group("verb"))
        rest = _clean_text(m.group("rest"))
        particle, rest = _strip_leading_object_prep(verb, rest)
        subject_action = " ".join(x for x in (subject, aux, verb, particle) if x)
        obj, scene = _split_first_scene(rest)
        result.update(
            {
                "subject_action": subject_action,
                "object_entity": obj,
                "scene_context": _normalize_scene(scene),
                "subject": subject,
            }
        )
        return result

    # Simpler fallback: "a boy rides a bike".
    tokens = caption.split()
    verb_idx = -1
    for idx, tok in enumerate(tokens[1:], start=1):
        if tok in _SIMPLE_VERBS or (idx >= 2 and tok.endswith("s")):
            verb_idx = idx
            break
    if verb_idx > 0:
        subject = " ".join(tokens[:verb_idx])
        verb = tokens[verb_idx]
        rest = " ".join(tokens[verb_idx + 1 :])
        obj, scene = _split_first_scene(rest)
        result.update(
            {
                "subject_action": " ".join(x for x in (subject, verb) if x),
                "object_entity": obj,
                "scene_context": _normalize_scene(scene),
                "subject": subject,
            }
        )
    return result


def _slot_by_type(slots: Iterable[Dict[str, Any]], slot_type: str) -> Dict[str, Any]:
    for slot in slots:
        if str(slot.get("slot_type") or slot.get("type") or "") == slot_type:
            return slot
    return {}


def project_record(record: Dict[str, Any], scene_min_prob: float = 0.0) -> Dict[str, Any]:
    caption = _clean_text(record.get("final_caption") or record.get("sentence_stage") or "")
    raw_slots = record.get("phrase_slots") or []
    parsed = parse_caption_slots(caption)
    subject = parsed.get("subject", "")

    raw_subject = _clean_text(_slot_by_type(raw_slots, "subject_action").get("text"))
    raw_object = _clean_text(_slot_by_type(raw_slots, "object_entity").get("text"))
    raw_scene_slot = _slot_by_type(raw_slots, "scene_context")
    raw_scene = _clean_text(raw_scene_slot.get("text"))
    try:
        raw_scene_prob = float(raw_scene_slot.get("presence_prob") or 0.0)
    except (TypeError, ValueError):
        raw_scene_prob = 0.0

    subject_action = parsed.get("subject_action") or raw_subject or caption
    object_entity = parsed.get("object_entity") or ""
    object_source = "caption_parse"
    if not _valid_object(object_entity, subject) and _valid_object(raw_object, subject):
        object_entity = raw_object
        object_source = "phrase_slot"
    elif not _valid_object(object_entity, subject):
        object_entity = ""
        object_source = "empty"

    scene_context = parsed.get("scene_context") or ""
    scene_source = "caption_parse" if scene_context else "empty"
    if (
        not _valid_scene(scene_context)
        and _valid_scene(raw_scene)
        and raw_scene_prob >= float(scene_min_prob)
    ):
        scene_context = _normalize_scene(raw_scene)
        scene_source = "phrase_slot"
    elif not _valid_scene(scene_context):
        scene_context = ""
        scene_source = "empty"

    slots = [
        {
            "slot_type": "subject_action",
            "text": subject_action,
            "source": "caption_parse" if parsed.get("subject_action") else "phrase_slot",
            "presence": bool(subject_action),
        },
        {
            "slot_type": "object_entity",
            "text": object_entity,
            "source": object_source,
            "presence": bool(object_entity),
        },
        {
            "slot_type": "scene_context",
            "text": scene_context,
            "source": scene_source,
            "presence": bool(scene_context),
        },
    ]
    return {
        "video_id": record.get("video_id"),
        "final_caption": caption,
        "anchor_sov_units": [slot["text"] for slot in slots if slot["text"]],
        "anchor_sov_slots": slots,
        "raw_predicted_phrase_units": record.get("predicted_phrase_units"),
    }


def export_projection(
    input_jsonl: Path, output_jsonl: Path, scene_min_prob: float = 0.0
) -> Dict[str, Any]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    object_filled = 0
    scene_filled = 0
    bad_repeats = 0
    with input_jsonl.open("r", encoding="utf-8") as src, output_jsonl.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            projected = project_record(json.loads(line), scene_min_prob=scene_min_prob)
            total += 1
            units = " ".join(projected["anchor_sov_units"])
            if _REPEAT_RE.search(units):
                bad_repeats += 1
            slot_map = {s["slot_type"]: s for s in projected["anchor_sov_slots"]}
            object_filled += int(bool(slot_map["object_entity"]["text"]))
            scene_filled += int(bool(slot_map["scene_context"]["text"]))
            dst.write(json.dumps(projected, ensure_ascii=False) + "\n")
    return {
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "scene_min_prob": float(scene_min_prob),
        "records": total,
        "object_filled": object_filled,
        "scene_filled": scene_filled,
        "repeat_bad": bad_repeats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--scene_min_prob", type=float, default=0.0)
    parser.add_argument("--preview", type=int, default=12)
    args = parser.parse_args()

    summary = export_projection(
        Path(args.input_jsonl),
        Path(args.output_jsonl),
        scene_min_prob=float(args.scene_min_prob),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.preview > 0:
        with Path(args.output_jsonl).open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= args.preview:
                    break
                record = json.loads(line)
                print("---")
                print(record["video_id"])
                print(record["final_caption"])
                print(" | ".join(record["anchor_sov_units"]))


if __name__ == "__main__":
    main()
