# -*- coding: utf-8 -*-
"""dataset_clip4clip_patch.py

Clip4Clip caption dataset with patch token features.
"""

import os
import sys
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CLIP4CLIP_PATH = os.path.join(PROJECT_ROOT, 'clip4clip_caption')
if CLIP4CLIP_PATH not in sys.path:
    sys.path.insert(0, CLIP4CLIP_PATH)

from clip4clip_caption.dataloaders.dataset_msvd_feats_clip4clip import MSVD_FeaturesDataset
from clip4clip_caption.dataloaders.dataset_msrvtt_feats_clip4clip import MSRVTT_FeaturesDataset


class Clip4ClipPatchCaptionDataset(Dataset):
    """
    Dataset that returns global clip4clip features with patch tokens.

    Returns:
        vid_feat, vid_mask, patch_feat, caption_ids, caption_mask, caption_label, vid, sen_id
    """

    def __init__(
        self,
        features_path: str,
        annotations_path: str,
        patch_root: str,
        dataset_type: str = 'msvd',
        split: str = 'train',
        patch_block: int = 5,
        tkz_type: str = 'clip',
        use_precomputed_clusters: bool = False,
        n_clusters: int = 32,
    ):
        super().__init__()

        if dataset_type == 'msvd':
            self.base_dataset = MSVD_FeaturesDataset(
                features_path=features_path,
                annotations_path=annotations_path,
                split=split,
                tkz_type=tkz_type,
            )
        elif dataset_type == 'msrvtt':
            self.base_dataset = MSRVTT_FeaturesDataset(
                features_path=features_path,
                json_path=annotations_path,
                split=split,
                tkz_type=tkz_type,
            )
        else:
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")

        self.split = split
        self.patch_block = patch_block
        self.use_precomputed_clusters = use_precomputed_clusters
        self.n_clusters = n_clusters

        self.patch_root = Path(patch_root)
        self.patch_dir = self.patch_root / f"clip4clip_block{patch_block}" / split
        if not self.patch_dir.exists():
            raise ValueError(f"Patch directory not found: {self.patch_dir}")

        self.cluster_dir = None
        if use_precomputed_clusters:
            cluster_dir = self.patch_root / f"clip4clip_block{patch_block}" / f"clustering_k{n_clusters}" / split
            if cluster_dir.exists():
                self.cluster_dir = cluster_dir
            else:
                self.use_precomputed_clusters = False

    def __len__(self):
        return len(self.base_dataset)

    def _load_patch_feat(self, vid: str) -> torch.Tensor:
        if self.use_precomputed_clusters and self.cluster_dir is not None:
            cluster_path = self.cluster_dir / f"{vid}.pt"
            if cluster_path.exists():
                cluster_data = torch.load(cluster_path, weights_only=False)
                return cluster_data['cluster_centers'].float()
            return torch.zeros(self.n_clusters, 768, dtype=torch.float32)

        patch_path = self.patch_dir / f"{vid}.pt"
        if patch_path.exists():
            patch_data = torch.load(patch_path, weights_only=False)
            if isinstance(patch_data, dict):
                patch_feat = patch_data['feats']
            else:
                patch_feat = patch_data
            return patch_feat.float()

        return torch.zeros(12, 50, 768, dtype=torch.float32)

    def __getitem__(self, idx: int):
        vid_feat, vid_mask, caption_ids, caption_mask, caption_label, vid, sen_id = self.base_dataset[idx]
        patch_feat = self._load_patch_feat(vid)
        return (
            vid_feat,
            vid_mask,
            patch_feat,
            caption_ids,
            caption_mask,
            caption_label,
            vid,
            sen_id,
        )


def collate_patch_caption_fn(batch):
    vid_feats = torch.stack([item[0] for item in batch])
    vid_masks = torch.stack([item[1] for item in batch])
    patch_feats = torch.stack([item[2] for item in batch])
    caption_ids = torch.stack([item[3] for item in batch])
    caption_masks = torch.stack([item[4] for item in batch])
    caption_labels = [item[5] for item in batch]
    vid_ids = [item[6] for item in batch]
    sen_ids = [item[7] for item in batch]

    return {
        'video_feats': vid_feats,
        'video_mask': vid_masks,
        'patch_feat': patch_feats,
        'caption_ids': caption_ids,
        'caption_mask': caption_masks,
        'caption_label': caption_labels,
        'video_id': vid_ids,
        'sen_id': sen_ids,
    }
