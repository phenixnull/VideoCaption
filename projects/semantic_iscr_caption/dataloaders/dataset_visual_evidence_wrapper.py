import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset


class VisualEvidenceDatasetWrapper(Dataset):
    """Append raw CLIP global and patch evidence tensors to an existing dataset item."""

    RAW_GLOBAL_FALLBACK_SHAPE = (12, 512)
    PATCH_FALLBACK_SHAPE = (12, 50, 768)

    def __init__(
        self,
        base_dataset: Dataset,
        *,
        raw_global_feats_path: Optional[str] = None,
        patch_root: Optional[str] = None,
        patch_block: int = 6,
        raw_global_enable: bool = True,
        patch_enable: bool = True,
        split: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.raw_global_enable = bool(raw_global_enable and raw_global_feats_path)
        self.patch_enable = bool(patch_enable and patch_root)
        self.patch_block = int(patch_block)
        self.split = str(
            split
            or getattr(base_dataset, "data_split", "")
            or getattr(base_dataset, "split", "")
            or "train"
        ).strip()

        self.raw_global_feats_path = Path(raw_global_feats_path).expanduser() if raw_global_feats_path else None
        self.patch_root = Path(patch_root).expanduser() if patch_root else None
        self.patch_dir = None
        self._raw_global_store: Dict[str, Any] = {}

        if self.raw_global_enable:
            if self.raw_global_feats_path is None or not self.raw_global_feats_path.exists():
                raise FileNotFoundError(f"raw_global_feats_path not found: {self.raw_global_feats_path}")
            with self.raw_global_feats_path.open("rb") as f:
                payload = pickle.load(f)
            if not isinstance(payload, dict):
                raise TypeError("raw_global_feats_path must contain a dict mapping video_id to features.")
            self._raw_global_store = payload

        if self.patch_enable:
            if self.patch_root is None or not self.patch_root.exists():
                raise FileNotFoundError(f"patch_root not found: {self.patch_root}")
            patch_dir = self.patch_root / f"clip_block{self.patch_block}" / self.split
            if not patch_dir.exists():
                raise FileNotFoundError(f"patch directory not found: {patch_dir}")
            self.patch_dir = patch_dir

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getattr__(self, name: str) -> Any:
        if name == "base_dataset":
            raise AttributeError(name)
        return getattr(self.base_dataset, name)

    @staticmethod
    def _to_float_tensor(value: Any) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.detach().cpu().float()
        return torch.as_tensor(value, dtype=torch.float32)

    @staticmethod
    def _to_mask_tensor(value: Any) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.detach().cpu().bool()
        return torch.as_tensor(value, dtype=torch.bool)

    def _load_raw_global(self, vid: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.raw_global_enable:
            return (
                torch.zeros(self.RAW_GLOBAL_FALLBACK_SHAPE, dtype=torch.float32),
                torch.zeros(self.RAW_GLOBAL_FALLBACK_SHAPE[0], dtype=torch.bool),
            )

        item = self._raw_global_store.get(str(vid))
        if item is None:
            return (
                torch.zeros(self.RAW_GLOBAL_FALLBACK_SHAPE, dtype=torch.float32),
                torch.zeros(self.RAW_GLOBAL_FALLBACK_SHAPE[0], dtype=torch.bool),
            )

        feats = None
        mask = None
        if isinstance(item, dict):
            feats = item.get("feats", item.get("feat"))
            mask = item.get("mask")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            feats, mask = item[0], item[1]
        else:
            feats = item

        feat_tensor = self._to_float_tensor(feats)
        if feat_tensor.dim() != 2:
            feat_tensor = feat_tensor.reshape(self.RAW_GLOBAL_FALLBACK_SHAPE)

        if mask is None:
            mask_tensor = feat_tensor.abs().sum(dim=-1) > 0
        else:
            mask_tensor = self._to_mask_tensor(mask).flatten()
        return feat_tensor, mask_tensor

    def _load_patch(self, vid: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.patch_enable or self.patch_dir is None:
            return (
                torch.zeros(self.PATCH_FALLBACK_SHAPE, dtype=torch.float32),
                torch.zeros(self.PATCH_FALLBACK_SHAPE[0], dtype=torch.bool),
            )

        patch_path = self.patch_dir / f"{vid}.pt"
        if not patch_path.exists():
            return (
                torch.zeros(self.PATCH_FALLBACK_SHAPE, dtype=torch.float32),
                torch.zeros(self.PATCH_FALLBACK_SHAPE[0], dtype=torch.bool),
            )

        payload = torch.load(patch_path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            feats = payload.get("feats", payload.get("feat"))
            mask = payload.get("mask")
        else:
            feats = payload
            mask = None

        feat_tensor = self._to_float_tensor(feats)
        if feat_tensor.dim() != 3:
            feat_tensor = feat_tensor.reshape(self.PATCH_FALLBACK_SHAPE)

        if mask is None:
            mask_tensor = feat_tensor.abs().sum(dim=(-1, -2)) > 0
        else:
            mask_tensor = self._to_mask_tensor(mask).flatten()
        return feat_tensor, mask_tensor

    def __getitem__(self, idx: int):
        base = self.base_dataset[idx]
        if not isinstance(base, (tuple, list)) or len(base) < 7:
            raise ValueError("base_dataset item must be a tuple/list with at least 7 fields")

        vid = str(base[5])
        raw_global_feats, raw_global_mask = self._load_raw_global(vid)
        patch_feats, patch_mask = self._load_patch(vid)
        return tuple(base) + (
            raw_global_feats,
            raw_global_mask,
            patch_feats,
            patch_mask,
        )
