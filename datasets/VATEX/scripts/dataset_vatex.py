from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


SPLIT_FILES = {
    "train": "vatex_training_v1.0.json",
    "val": "vatex_validation_v1.0.json",
    "validation": "vatex_validation_v1.0.json",
    "test": "vatex_public_test_english_v1.1.json",
    "public_test": "vatex_public_test_english_v1.1.json",
    "private_test": "vatex_private_test_without_annotations.json",
}


VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".avi")


@dataclass(frozen=True)
class VATEXRecord:
    clip_id: str
    video_id: str
    start: Optional[float]
    end: Optional[float]
    caption: Optional[str]
    caption_index: Optional[int]
    captions: Tuple[str, ...]
    chinese_captions: Tuple[str, ...]


def parse_vatex_clip_id(clip_id: str) -> Tuple[str, Optional[float], Optional[float]]:
    """Split VATEX clip id into YouTube id and temporal window.

    VATEX ids usually look like ``Ptf_2VRj-V0_000122_000132``.  The
    YouTube id itself may contain underscores, so split from the right.
    """

    parts = clip_id.rsplit("_", 2)
    if len(parts) != 3:
        return clip_id, None, None
    video_id, start_s, end_s = parts
    try:
        return video_id, float(int(start_s)), float(int(end_s))
    except ValueError:
        return clip_id, None, None


def _default_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"VATEX annotation must be a list: {path}")
    return data


def _load_features(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def _as_feature_pair(value: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(value, dict):
        feat = value.get("features", value.get("feat", value.get("video")))
        mask = value.get("mask", value.get("video_mask"))
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        feat, mask = value
    else:
        feat = value
        mask = np.ones((np.asarray(feat).shape[0],), dtype=np.int64)

    feat_tensor = torch.as_tensor(np.asarray(feat), dtype=torch.float32)
    mask_tensor = torch.as_tensor(np.asarray(mask), dtype=torch.long)
    return feat_tensor, mask_tensor


def _find_raw_video(raw_dir: Optional[Path], keys: Sequence[str]) -> Optional[str]:
    if raw_dir is None:
        return None
    for key in keys:
        for ext in VIDEO_EXTENSIONS:
            candidate = raw_dir / f"{key}{ext}"
            if candidate.exists():
                return str(candidate)
    return None


class VATEXCaptionDataset(Dataset):
    """VATEX annotation/features dataset.

    The loader intentionally does not force a single upstream feature format.
    It supports the project baseline pickle format ``{video_id: (feat, mask)}``
    and dict variants such as ``{"features": ..., "mask": ...}``.
    """

    def __init__(
        self,
        root: Optional[str | Path] = None,
        split: str = "train",
        annotation_path: Optional[str | Path] = None,
        features_path: Optional[str | Path] = None,
        raw_video_dir: Optional[str | Path] = None,
        caption_lang: str = "en",
        caption_mode: str = "auto",
        tokenizer: Optional[Callable[[str], Any]] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root) if root is not None else _default_root()
        self.split = split
        self.caption_lang = caption_lang
        self.caption_mode = caption_mode
        self.tokenizer = tokenizer

        if annotation_path is None:
            try:
                annotation_name = SPLIT_FILES[split]
            except KeyError as exc:
                valid = ", ".join(sorted(SPLIT_FILES))
                raise ValueError(f"Unknown VATEX split '{split}'. Valid: {valid}") from exc
            annotation_path = self.root / "annotations" / annotation_name
        self.annotation_path = Path(annotation_path)
        self.features_path = Path(features_path) if features_path is not None else None
        self.raw_video_dir = Path(raw_video_dir) if raw_video_dir is not None else self.root / "raw"

        raw_items = _load_json(self.annotation_path)
        self.features = _load_features(self.features_path)
        self.records = self._build_records(raw_items)

    def _build_records(self, raw_items: Iterable[Dict[str, Any]]) -> List[VATEXRecord]:
        records: List[VATEXRecord] = []
        for item in raw_items:
            clip_id = str(item["videoID"])
            video_id, start, end = parse_vatex_clip_id(clip_id)
            en_caps = tuple(str(x).strip() for x in item.get("enCap", []) if str(x).strip())
            zh_caps = tuple(str(x).strip() for x in item.get("chCap", []) if str(x).strip())
            captions = en_caps if self.caption_lang == "en" else zh_caps

            mode = self.caption_mode
            if mode == "auto":
                mode = "all" if self.split == "train" else "first"
                if not captions:
                    mode = "video"

            if mode == "all" and captions:
                for cap_idx, caption in enumerate(captions):
                    records.append(VATEXRecord(clip_id, video_id, start, end, caption, cap_idx, en_caps, zh_caps))
            elif mode == "first" and captions:
                records.append(VATEXRecord(clip_id, video_id, start, end, captions[0], 0, en_caps, zh_caps))
            elif mode == "video":
                caption = captions[0] if captions else None
                records.append(VATEXRecord(clip_id, video_id, start, end, caption, None, en_caps, zh_caps))
            else:
                raise ValueError("caption_mode must be one of: auto, all, first, video")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def _get_features(self, record: VATEXRecord) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.features is None:
            return None, None
        for key in (record.clip_id, record.video_id):
            if key in self.features:
                return _as_feature_pair(self.features[key])
        return None, None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        feature, feature_mask = self._get_features(record)
        tokenized = self.tokenizer(record.caption) if self.tokenizer and record.caption is not None else None
        raw_video_path = _find_raw_video(self.raw_video_dir, (record.clip_id, record.video_id))

        return {
            "dataset": "VATEX",
            "split": self.split,
            "clip_id": record.clip_id,
            "video_id": record.video_id,
            "youtube_url": f"https://www.youtube.com/watch?v={record.video_id}",
            "start": record.start,
            "end": record.end,
            "caption": record.caption,
            "caption_index": record.caption_index,
            "captions": list(record.captions),
            "chinese_captions": list(record.chinese_captions),
            "tokens": tokenized,
            "feature": feature,
            "feature_mask": feature_mask,
            "raw_video_path": raw_video_path,
        }

    def get_references(self) -> Dict[str, List[str]]:
        refs: Dict[str, List[str]] = {}
        for record in self.records:
            if record.captions:
                refs[record.clip_id] = list(record.captions)
        return refs


def collate_vatex(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {key: [sample[key] for sample in batch] for key in batch[0].keys()}
    features = [sample["feature"] for sample in batch]
    masks = [sample["feature_mask"] for sample in batch]
    if all(x is not None for x in features):
        shapes = {tuple(x.shape) for x in features if x is not None}
        if len(shapes) == 1:
            output["feature"] = torch.stack([x for x in features if x is not None], dim=0)
            output["feature_mask"] = torch.stack([x for x in masks if x is not None], dim=0)
    return output

