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
    "train": "train.json",
    "val": "val_1.json",
    "validation": "val_1.json",
    "test": "val_2.json",
    "val_1": "val_1.json",
    "val_2": "val_2.json",
}


VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".avi")


@dataclass(frozen=True)
class ActivityNetSegment:
    video_id: str
    duration: float
    start: Optional[float]
    end: Optional[float]
    caption: Optional[str]
    segment_index: Optional[int]
    timestamps: Tuple[Tuple[float, float], ...]
    sentences: Tuple[str, ...]


def _default_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"ActivityNet Captions annotation must be a dict: {path}")
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


def _find_raw_video(raw_dir: Optional[Path], video_id: str) -> Optional[str]:
    if raw_dir is None:
        return None
    candidates = [video_id]
    if video_id.startswith("v_"):
        candidates.append(video_id[2:])
    else:
        candidates.append(f"v_{video_id}")
    for key in candidates:
        for ext in VIDEO_EXTENSIONS:
            candidate = raw_dir / f"{key}{ext}"
            if candidate.exists():
                return str(candidate)
    return None


class ActivityNetCaptionsDataset(Dataset):
    """ActivityNet Captions annotation/features dataset.

    ``train`` maps to ``train.json``.  ``val`` maps to ``val_1.json`` and
    ``test`` maps to ``val_2.json``, which is the common open-source proxy
    split because the real ActivityNet test captions are not public.
    """

    def __init__(
        self,
        root: Optional[str | Path] = None,
        split: str = "train",
        annotation_path: Optional[str | Path] = None,
        features_path: Optional[str | Path] = None,
        raw_video_dir: Optional[str | Path] = None,
        sample_mode: str = "segments",
        tokenizer: Optional[Callable[[str], Any]] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root) if root is not None else _default_root()
        self.split = split
        self.sample_mode = sample_mode
        self.tokenizer = tokenizer

        if annotation_path is None:
            try:
                annotation_name = SPLIT_FILES[split]
            except KeyError as exc:
                valid = ", ".join(sorted(SPLIT_FILES))
                raise ValueError(f"Unknown ActivityNet split '{split}'. Valid: {valid}") from exc
            annotation_path = self.root / "annotations" / annotation_name
        self.annotation_path = Path(annotation_path)
        self.features_path = Path(features_path) if features_path is not None else None
        self.raw_video_dir = Path(raw_video_dir) if raw_video_dir is not None else self.root / "raw"

        raw_items = _load_json(self.annotation_path)
        self.features = _load_features(self.features_path)
        self.records = self._build_records(raw_items.items())

    def _build_records(self, items: Iterable[Tuple[str, Dict[str, Any]]]) -> List[ActivityNetSegment]:
        records: List[ActivityNetSegment] = []
        for video_id, info in items:
            duration = float(info.get("duration", 0.0))
            raw_timestamps = info.get("timestamps", [])
            raw_sentences = info.get("sentences", [])
            timestamps = tuple((float(x[0]), float(x[1])) for x in raw_timestamps)
            sentences = tuple(str(x).strip() for x in raw_sentences)

            if self.sample_mode == "segments":
                for seg_idx, (timestamp, sentence) in enumerate(zip(timestamps, sentences)):
                    records.append(
                        ActivityNetSegment(
                            video_id=video_id,
                            duration=duration,
                            start=timestamp[0],
                            end=timestamp[1],
                            caption=sentence,
                            segment_index=seg_idx,
                            timestamps=timestamps,
                            sentences=sentences,
                        )
                    )
            elif self.sample_mode == "video":
                caption = sentences[0] if sentences else None
                start_end = timestamps[0] if timestamps else (None, None)
                records.append(
                    ActivityNetSegment(
                        video_id=video_id,
                        duration=duration,
                        start=start_end[0],
                        end=start_end[1],
                        caption=caption,
                        segment_index=None,
                        timestamps=timestamps,
                        sentences=sentences,
                    )
                )
            else:
                raise ValueError("sample_mode must be one of: segments, video")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def _get_features(self, record: ActivityNetSegment) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.features is None:
            return None, None
        candidates = (record.video_id, record.video_id[2:] if record.video_id.startswith("v_") else f"v_{record.video_id}")
        for key in candidates:
            if key in self.features:
                return _as_feature_pair(self.features[key])
        return None, None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        feature, feature_mask = self._get_features(record)
        tokenized = self.tokenizer(record.caption) if self.tokenizer and record.caption is not None else None
        raw_video_path = _find_raw_video(self.raw_video_dir, record.video_id)

        return {
            "dataset": "ActivityNetCaptions",
            "split": self.split,
            "video_id": record.video_id,
            "duration": record.duration,
            "start": record.start,
            "end": record.end,
            "timestamp": [record.start, record.end] if record.start is not None else None,
            "caption": record.caption,
            "segment_index": record.segment_index,
            "timestamps": [list(x) for x in record.timestamps],
            "sentences": list(record.sentences),
            "tokens": tokenized,
            "feature": feature,
            "feature_mask": feature_mask,
            "raw_video_path": raw_video_path,
        }

    def get_references(self) -> Dict[str, List[str]]:
        refs: Dict[str, List[str]] = {}
        for record in self.records:
            refs[record.video_id] = list(record.sentences)
        return refs


def collate_activitynet(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {key: [sample[key] for sample in batch] for key in batch[0].keys()}
    features = [sample["feature"] for sample in batch]
    masks = [sample["feature_mask"] for sample in batch]
    if all(x is not None for x in features):
        shapes = {tuple(x.shape) for x in features if x is not None}
        if len(shapes) == 1:
            output["feature"] = torch.stack([x for x in features if x is not None], dim=0)
            output["feature_mask"] = torch.stack([x for x in masks if x is not None], dim=0)
    return output

