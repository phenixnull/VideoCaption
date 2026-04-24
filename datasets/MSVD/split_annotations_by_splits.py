# -*- coding: utf-8 -*-
"""
split_annotations_by_splits.py — 基于已带 sen_id 的 annotations_preprocessed.txt
和官方提供的 train/val/test.txt 中的 video_id 划分出三份子集：

输入文件（同目录下）：
  - annotations_preprocessed.txt : video_id sen_id caption
  - train.txt / val.txt / test.txt: video_id caption

输出文件：
  - train_preprocessed.txt
  - val_preprocessed.txt
  - test_preprocessed.txt

注意：
- 不重新分配 sen_id，直接复用 annotations_preprocessed.txt 里已有的 sen_id。
- 仅根据 video_id 归属到 train/val/test 三个 split。
"""

import os
from typing import Dict, Set


def load_split_ids(split_path: str) -> Set[str]:
    """从 train/val/test.txt 读取该 split 包含的 video_id 集合。

    每行格式：video_id caption
    只取第一个 token 作为 video_id。
    """
    ids: Set[str] = set()
    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            vid = parts[0]
            ids.add(vid)
    print(f"[INFO] {os.path.basename(split_path)}: load {len(ids)} unique video ids")
    return ids


def split_preprocessed_annotations(base_dir: str) -> None:
    """根据 train/val/test.txt 中的 video_id，将 annotations_preprocessed.txt 划分到三份输出文件。

    Args:
        base_dir: 包含 annotations_preprocessed.txt 以及 train/val/test.txt 的目录
    """
    ann_path = os.path.join(base_dir, "annotations_preprocessed.txt")
    train_ids_path = os.path.join(base_dir, "train.txt")
    val_ids_path = os.path.join(base_dir, "val.txt")
    test_ids_path = os.path.join(base_dir, "test.txt")

    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"annotations_preprocessed.txt not found: {ann_path}")
    for p in [train_ids_path, val_ids_path, test_ids_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"split file not found: {p}")

    print(f"[INFO] Loading split ids from train/val/test.txt ...")
    split_id_map: Dict[str, Set[str]] = {
        "train": load_split_ids(train_ids_path),
        "val": load_split_ids(val_ids_path),
        "test": load_split_ids(test_ids_path),
    }

    # 打开三个输出文件
    out_paths = {
        "train": os.path.join(base_dir, "train_preprocessed.txt"),
        "val": os.path.join(base_dir, "val_preprocessed.txt"),
        "test": os.path.join(base_dir, "test_preprocessed.txt"),
    }
    out_files = {name: open(path, "w", encoding="utf-8") for name, path in out_paths.items()}

    try:
        total_lines = 0
        kept = {"train": 0, "val": 0, "test": 0}

        print(f"[INFO] Reading {ann_path} and dispatching lines to splits ...")
        with open(ann_path, "r", encoding="utf-8") as f:
            for line in f:
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                # 格式: video_id sen_id caption
                parts = line_stripped.split(" ", 2)
                if len(parts) < 3:
                    continue
                vid, sen_id, caption = parts
                # 根据 video_id 所属的 split 写入相应文件
                for split_name, id_set in split_id_map.items():
                    if vid in id_set:
                        out_files[split_name].write(line_stripped + "\n")
                        kept[split_name] += 1
                        break
                total_lines += 1

        print(f"[OK] Done. Processed {total_lines} lines from annotations_preprocessed.txt")
        for split_name in ["train", "val", "test"]:
            print(f"      {split_name}: wrote {kept[split_name]} lines -> {out_paths[split_name]}")

    finally:
        for f in out_files.values():
            f.close()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    split_preprocessed_annotations(script_dir)
