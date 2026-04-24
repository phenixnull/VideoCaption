from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
from torch.utils.data import DataLoader

from dataset_activitynet_captions import ActivityNetCaptionsDataset, collate_activitynet


def _dataset_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _make_tiny_features(root: Path, splits: Iterable[str], limit_per_split: int, out_path: Path) -> None:
    rng = np.random.default_rng(20260424)
    features: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in splits:
        dataset = ActivityNetCaptionsDataset(root=root, split=split, features_path=None)
        seen = []
        for sample in dataset:
            if sample["video_id"] in seen:
                continue
            seen.append(sample["video_id"])
            feat = rng.normal(size=(16, 512)).astype("float32")
            mask = np.ones((16,), dtype="int64")
            features[sample["video_id"]] = (feat, mask)
            if len(seen) >= limit_per_split:
                break
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(features, f)
    print(f"[tiny-features] wrote {len(features)} entries -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test ActivityNetCaptionsDataset with tiny feature fixtures.")
    parser.add_argument("--root", type=Path, default=_dataset_root())
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--feature-path", type=Path, default=None)
    parser.add_argument("--skip-create-features", action="store_true")
    args = parser.parse_args()

    feature_path = args.feature_path or (args.root / "samples" / "tiny_activitynet_features.pickle")
    splits = ["train", "val", "test"]
    if not args.skip_create_features:
        _make_tiny_features(args.root, splits, args.limit, feature_path)

    for split in splits:
        print(f"\n== ActivityNet split={split} ==")
        metadata_dataset = ActivityNetCaptionsDataset(root=args.root, split=split)
        print(f"segment samples: {len(metadata_dataset)}")
        for idx in range(min(args.limit, len(metadata_dataset))):
            sample = metadata_dataset[idx]
            print(
                f"  [{idx}] video_id={sample['video_id']} "
                f"timestamp={sample['timestamp']} "
                f"caption={sample['caption']!r}"
            )

        feature_dataset = ActivityNetCaptionsDataset(root=args.root, split=split, features_path=feature_path)
        loader = DataLoader(feature_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_activitynet)
        batch = next(iter(loader))
        feature = batch["feature"]
        mask = batch["feature_mask"]
        print(f"feature batch type={type(feature).__name__} shape={getattr(feature, 'shape', None)}")
        print(f"mask batch type={type(mask).__name__} shape={getattr(mask, 'shape', None)}")
        assert getattr(feature, "shape", None) == (2, 16, 512), "feature fixture did not stack as expected"
        assert getattr(mask, "shape", None) == (2, 16), "mask fixture did not stack as expected"

    video_dataset = ActivityNetCaptionsDataset(root=args.root, split="val", sample_mode="video")
    video_sample = video_dataset[0]
    print("\n== ActivityNet sample_mode=video ==")
    print(f"video samples: {len(video_dataset)}")
    print(f"  [0] video_id={video_sample['video_id']} sentences={len(video_sample['sentences'])}")
    print("\nActivityNet smoke test passed.")


if __name__ == "__main__":
    main()

