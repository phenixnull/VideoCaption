# -*- coding: utf-8 -*-
"""
dataset_msvd_phrase.py - MSVD 短语数据集加载类

用于 Stage 2 短语生成训练

短语标注格式 (JSON):
    {
        "video_objs": {
            "noun1": ["phrase1", "phrase2", ...],
            "noun2": ["phrase1", "phrase2", ...],
            ...
        }
    }

返回:
    - video_feats: [T, D] 视频特征
    - video_mask: [T] 视频掩码
    - phrases: List[str] 所有短语文本
    - phrase_tokens: [num_phrases, seq_len] 短语 token IDs
    - phrase_mask: [num_phrases, seq_len] 短语 attention mask
    - num_phrases: int 短语数量
    - noun_labels: List[int] 每个短语对应的名词 token ID
    - video_id: str 视频ID
"""

import os
import sys
import json
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Tuple
from collections import defaultdict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class MSVD_PhraseDataset(Dataset):
    """
    MSVD 短语数据集

    用于 Stage 2 的 Slot-Query 短语生成训练

    Args:
        features_path: 视频特征 pickle 文件路径
        annotations_path: 标注文件路径 (用于获取 video ID 列表)
        phrase_dir: 短语标注目录 (包含 train/val/test 子目录)
        split: 数据集划分 ['train', 'val', 'test']
        max_phrases: 每个视频最大短语数量
        max_phrase_len: 每个短语最大 token 长度
        vocab_size: 词表大小
    """

    def __init__(
        self,
        features_path: str,
        annotations_path: str,
        phrase_dir: str,
        split: str = 'train',
        max_phrases: int = 64,
        max_phrase_len: int = 20,
        vocab_size: int = 49412,
        deduplicate_phrases: bool = True,
    ):
        self.split = split
        self.max_phrases = max_phrases
        self.max_phrase_len = max_phrase_len
        self.vocab_size = vocab_size
        self.deduplicate_phrases = deduplicate_phrases

        # 1. 加载视频特征
        print(f"[MSVD_PhraseDataset] Loading features from: {features_path}")
        with open(features_path, 'rb') as f:
            self.features = pickle.load(f)
        print(f"  -> Loaded {len(self.features)} video features")

        # 2. 初始化 tokenizer
        from load_tokenizers import CLIPTokenizer_Custom
        self.tokenizer = CLIPTokenizer_Custom()

        # 3. 确定短语标注目录
        self.phrase_split_dir = os.path.join(phrase_dir, split)
        if not os.path.exists(self.phrase_split_dir):
            raise ValueError(f"Phrase directory not found: {self.phrase_split_dir}")

        # 4. 获取视频 ID 列表 (从短语标注文件)
        phrase_files = [f for f in os.listdir(self.phrase_split_dir) if f.endswith('.json')]
        self.video_ids = [f.replace('.json', '') for f in phrase_files]

        # 过滤掉没有特征的视频
        available_ids = set(self.features.keys())
        self.video_ids = [vid for vid in self.video_ids if vid in available_ids]
        self.video_ids = sorted(self.video_ids)

        print(f"  -> Split '{split}': {len(self.video_ids)} videos with phrase annotations")

        # 5. 预加载所有短语标注 (避免每次 __getitem__ 都读文件)
        self.phrase_cache = {}
        self._preload_phrases()

    def _preload_phrases(self):
        """预加载所有短语标注"""
        print(f"[MSVD_PhraseDataset] Preloading phrase annotations...")
        for vid in self.video_ids:
            phrase_file = os.path.join(self.phrase_split_dir, f"{vid}.json")
            try:
                with open(phrase_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.phrase_cache[vid] = data.get('video_objs', {})
            except Exception as e:
                print(f"  Warning: Failed to load {phrase_file}: {e}")
                self.phrase_cache[vid] = {}

        total_phrases = sum(
            sum(len(phrases) for phrases in obj.values())
            for obj in self.phrase_cache.values()
        )
        print(f"  -> Loaded {total_phrases} phrases from {len(self.phrase_cache)} videos")

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx: int) -> Dict:
        vid = self.video_ids[idx]

        # 1. 获取视频特征
        feat, vid_mask = self.features[vid]
        video_feats = torch.tensor(feat, dtype=torch.float32)
        video_mask = torch.tensor(vid_mask, dtype=torch.long)

        # 2. 获取短语标注
        video_objs = self.phrase_cache.get(vid, {})
        all_phrases = []
        noun_labels = []

        for noun, phrases in video_objs.items():
            # 获取名词的 token ID
            noun_encoded = self.tokenizer.encode_plus(
                noun, padding=False, truncation=True, max_length=10, return_tensors='pt'
            )
            # 取第一个非特殊 token 作为名词 ID
            noun_ids = noun_encoded['input_ids'].squeeze(0).tolist()
            # 跳过 BOS (49406)
            noun_id = noun_ids[1] if len(noun_ids) > 1 else noun_ids[0]

            for phrase in phrases:
                if self.deduplicate_phrases and phrase in all_phrases:
                    continue
                all_phrases.append(phrase)
                noun_labels.append(noun_id)

        # 3. 限制短语数量
        if len(all_phrases) > self.max_phrases:
            indices = np.random.choice(len(all_phrases), self.max_phrases, replace=False)
            all_phrases = [all_phrases[i] for i in indices]
            noun_labels = [noun_labels[i] for i in indices]

        num_phrases = len(all_phrases)

        # 4. Tokenize 短语
        if num_phrases > 0:
            phrase_tokens_list = []
            phrase_mask_list = []

            for phrase in all_phrases:
                encoded = self.tokenizer.encode_plus(
                    phrase,
                    padding='max_length',
                    max_length=self.max_phrase_len,
                    truncation=True,
                    return_tensors='pt'
                )
                phrase_tokens_list.append(encoded['input_ids'].squeeze(0))
                phrase_mask_list.append(encoded['attention_mask'].squeeze(0))

            phrase_tokens = torch.stack(phrase_tokens_list, dim=0)  # [num_phrases, seq_len]
            phrase_mask = torch.stack(phrase_mask_list, dim=0)
        else:
            # 空占位
            phrase_tokens = torch.zeros(1, self.max_phrase_len, dtype=torch.long)
            phrase_mask = torch.zeros(1, self.max_phrase_len, dtype=torch.long)
            num_phrases = 0

        return {
            'video_feats': video_feats,      # [T, D]
            'video_mask': video_mask,        # [T]
            'phrase_tokens': phrase_tokens,  # [num_phrases, seq_len]
            'phrase_mask': phrase_mask,      # [num_phrases, seq_len]
            'num_phrases': num_phrases,
            'phrases': all_phrases,          # List[str]
            'noun_labels': noun_labels,      # List[int]
            'video_id': vid,
        }


def collate_phrase_batch(batch: List[Dict]) -> Dict:
    """
    短语数据集的 collate 函数

    处理变长短语数量: 对齐到 batch 中最大短语数
    """
    # 找出 batch 中最大短语数
    max_num_phrases = max(item['num_phrases'] for item in batch)
    max_num_phrases = max(max_num_phrases, 1)  # 至少 1

    batch_video_feats = []
    batch_video_mask = []
    batch_phrase_tokens = []
    batch_phrase_mask = []
    batch_phrase_valid_mask = []  # 标记哪些 slot 是有效的
    batch_num_phrases = []
    batch_video_ids = []

    for item in batch:
        batch_video_feats.append(item['video_feats'])
        batch_video_mask.append(item['video_mask'])
        batch_video_ids.append(item['video_id'])
        batch_num_phrases.append(item['num_phrases'])

        num_phrases = item['num_phrases']
        phrase_tokens = item['phrase_tokens']
        phrase_mask = item['phrase_mask']
        seq_len = phrase_tokens.shape[-1]

        # Padding 到 max_num_phrases
        if num_phrases < max_num_phrases:
            pad_size = max_num_phrases - num_phrases
            if num_phrases > 0:
                phrase_tokens = torch.cat([
                    phrase_tokens,
                    torch.zeros(pad_size, seq_len, dtype=torch.long)
                ], dim=0)
                phrase_mask = torch.cat([
                    phrase_mask,
                    torch.zeros(pad_size, seq_len, dtype=torch.long)
                ], dim=0)
            else:
                phrase_tokens = torch.zeros(max_num_phrases, seq_len, dtype=torch.long)
                phrase_mask = torch.zeros(max_num_phrases, seq_len, dtype=torch.long)

        batch_phrase_tokens.append(phrase_tokens)
        batch_phrase_mask.append(phrase_mask)

        # 有效 slot 掩码
        valid_mask = torch.zeros(max_num_phrases, dtype=torch.bool)
        valid_mask[:num_phrases] = True
        batch_phrase_valid_mask.append(valid_mask)

    return {
        'video_feats': torch.stack(batch_video_feats, dim=0),        # [B, T, D]
        'video_mask': torch.stack(batch_video_mask, dim=0),          # [B, T]
        'phrase_tokens': torch.stack(batch_phrase_tokens, dim=0),    # [B, M, S]
        'phrase_mask': torch.stack(batch_phrase_mask, dim=0),        # [B, M, S]
        'phrase_valid_mask': torch.stack(batch_phrase_valid_mask, dim=0),  # [B, M]
        'num_phrases': batch_num_phrases,
        'video_ids': batch_video_ids,
    }


def test_phrase_dataset():
    """测试短语数据集"""
    print("=" * 60)
    print("Testing MSVD_PhraseDataset")
    print("=" * 60)

    # 路径
    base_path = "/mnt/sda/Disk_D/zhangwei/projects/VC/datasets/MSVD"
    features_path = os.path.join(base_path, "feats", "ViT-B-32_k_split_ks12_features.pickle")
    annotations_path = os.path.join(base_path, "annotations_preprocessed.txt")
    phrase_dir = os.path.join(base_path, "annotations", "nouns")

    # 创建数据集
    dataset = MSVD_PhraseDataset(
        features_path=features_path,
        annotations_path=annotations_path,
        phrase_dir=phrase_dir,
        split='train',
        max_phrases=64,
        max_phrase_len=20,
    )

    print(f"\nDataset size: {len(dataset)}")

    # 测试单个样本
    print("\n1. Testing single sample:")
    sample = dataset[0]
    print(f"  video_id: {sample['video_id']}")
    print(f"  video_feats: {sample['video_feats'].shape}")
    print(f"  video_mask: {sample['video_mask'].shape}")
    print(f"  num_phrases: {sample['num_phrases']}")
    print(f"  phrase_tokens: {sample['phrase_tokens'].shape}")
    print(f"  First 3 phrases: {sample['phrases'][:3]}")

    # 测试 DataLoader
    print("\n2. Testing DataLoader with collate:")
    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_phrase_batch,
        num_workers=0,
    )

    batch = next(iter(loader))
    print(f"  video_feats: {batch['video_feats'].shape}")
    print(f"  video_mask: {batch['video_mask'].shape}")
    print(f"  phrase_tokens: {batch['phrase_tokens'].shape}")
    print(f"  phrase_mask: {batch['phrase_mask'].shape}")
    print(f"  phrase_valid_mask: {batch['phrase_valid_mask'].shape}")
    print(f"  num_phrases: {batch['num_phrases']}")

    # 统计短语数量分布
    print("\n3. Phrase count statistics:")
    phrase_counts = []
    for i in range(min(100, len(dataset))):
        sample = dataset[i]
        phrase_counts.append(sample['num_phrases'])

    print(f"  Min: {min(phrase_counts)}")
    print(f"  Max: {max(phrase_counts)}")
    print(f"  Mean: {np.mean(phrase_counts):.2f}")

    print("\nTest completed!")


if __name__ == "__main__":
    test_phrase_dataset()
