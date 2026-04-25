#!/usr/bin/env python
"""Merge scene phrase hints into the compact structured GT without expanding vocabularies."""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple


CAPTION_SCENE_KEYS = ("scene_units", "scene_phrases", "family_scene_phrases")
VIDEO_SCENE_KEYS = ("scenes", "scenes_state")


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _unique_strings(values: Any, *, limit: int = 4) -> List[str]:
    seen = set()
    result: List[str] = []
    if not isinstance(values, list):
        return result
    for value in values:
        text = _normalize_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _caption_maps(captions: Any) -> Tuple[Dict[str, Mapping[str, Any]], Dict[str, Mapping[str, Any]]]:
    by_sen_id: Dict[str, Mapping[str, Any]] = {}
    by_caption: Dict[str, Mapping[str, Any]] = {}
    if not isinstance(captions, list):
        return by_sen_id, by_caption
    for cap_info in captions:
        if not isinstance(cap_info, Mapping):
            continue
        sen_id = str(cap_info.get("sen_id", "")).strip()
        if sen_id:
            by_sen_id[sen_id] = cap_info
        caption = _normalize_text(cap_info.get("caption"))
        if caption:
            by_caption[caption] = cap_info
    return by_sen_id, by_caption


def _match_rich_caption(
    compact_caption: Mapping[str, Any],
    rich_by_sen_id: Mapping[str, Mapping[str, Any]],
    rich_by_caption: Mapping[str, Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    sen_id = str(compact_caption.get("sen_id", "")).strip()
    if sen_id and sen_id in rich_by_sen_id:
        return rich_by_sen_id[sen_id]
    caption = _normalize_text(compact_caption.get("caption"))
    if caption:
        return rich_by_caption.get(caption)
    return None


def _copy_caption_scene_hints(
    compact_caption: MutableMapping[str, Any],
    rich_caption: Mapping[str, Any],
    *,
    max_scene_units: int,
) -> bool:
    copied = False
    for key in CAPTION_SCENE_KEYS:
        scene_units = _unique_strings(rich_caption.get(key), limit=max_scene_units)
        if scene_units:
            compact_caption[key] = scene_units
            copied = True
    if copied and "scene_units" not in compact_caption:
        for key in ("scene_phrases", "family_scene_phrases"):
            if key in compact_caption:
                compact_caption["scene_units"] = list(compact_caption[key])
                break
    return copied


def merge_scene_hints(
    compact_data: Mapping[str, Any],
    rich_data: Mapping[str, Any],
    *,
    max_scene_units: int,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    merged = deepcopy(compact_data)
    compact_videos = merged.get("videos")
    rich_videos = rich_data.get("videos")
    if not isinstance(compact_videos, MutableMapping) or not isinstance(rich_videos, Mapping):
        raise ValueError("Both compact and rich GT files must contain a videos mapping.")

    stats = {
        "videos_total": len(compact_videos),
        "videos_with_scene_hints": 0,
        "captions_total": 0,
        "captions_with_scene_hints": 0,
    }

    for vid, compact_video in compact_videos.items():
        if not isinstance(compact_video, MutableMapping):
            continue
        rich_video = rich_videos.get(vid)
        if not isinstance(rich_video, Mapping):
            continue

        video_has_scene = False
        for key in VIDEO_SCENE_KEYS:
            value = rich_video.get(key)
            if value:
                compact_video[key] = deepcopy(value)
                video_has_scene = True

        rich_by_sen_id, rich_by_caption = _caption_maps(rich_video.get("captions"))
        compact_captions = compact_video.get("captions")
        if isinstance(compact_captions, list):
            for compact_caption in compact_captions:
                if not isinstance(compact_caption, MutableMapping):
                    continue
                stats["captions_total"] += 1
                rich_caption = _match_rich_caption(compact_caption, rich_by_sen_id, rich_by_caption)
                if rich_caption is None:
                    continue
                if _copy_caption_scene_hints(
                    compact_caption,
                    rich_caption,
                    max_scene_units=max_scene_units,
                ):
                    stats["captions_with_scene_hints"] += 1
                    video_has_scene = True

        if video_has_scene:
            stats["videos_with_scene_hints"] += 1

    meta = dict(merged.get("meta") or {})
    meta["scenehint"] = {
        "source": "compact structured GT plus scene phrase hints from rich role-aware GT",
        "max_scene_units_per_caption": max_scene_units,
        **stats,
    }
    merged["meta"] = meta
    merged.pop("scene_vocab", None)
    merged.pop("attribute_vocab", None)
    return merged, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact_gt", required=True)
    parser.add_argument("--rich_gt", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_scene_units", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compact_data = json.loads(Path(args.compact_gt).read_text(encoding="utf-8"))
    rich_data = json.loads(Path(args.rich_gt).read_text(encoding="utf-8"))
    merged, stats = merge_scene_hints(
        compact_data,
        rich_data,
        max_scene_units=max(1, int(args.max_scene_units)),
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
