# -*- coding: utf-8 -*-
"""
dataset_msvd_feats.py — MSVD 数据集特征加载类

基于预提取的 CLIP 视觉特征和预处理后的标注文件

特征格式 (pickle):
    {video_id: (feats[12,512], mask[12,])}

标注格式 (annotations_preprocessed.txt):
    video_id sen_id caption
    -4wsuPCjDBc_5_15 0 a squirrel is eating a peanut

MSVD 数据集划分 (共1970个视频):
    - train: 1200 个视频
    - val: 100 个视频  
    - test: 670 个视频
"""

import pickle
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import sys
import os
from collections import defaultdict
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)


class MSVD_FeaturesDataset(Dataset):
    """
    MSVD 特征数据集加载类
    
    Args:
        features_path: 视频特征 pickle 文件路径
        annotations_path: 预处理后的标注文件路径 (annotations_preprocessed.txt)
        split: 数据集划分 ['train', 'val', 'test', 'all']
        tkz_type: tokenizer 类型 ['clip', 'bert']
    
    Returns:
        vid_feat_tensor: [T, D] 视频特征
        vid_mask_tensor: [T] 视频掩码
        caption_ids: [max_len] caption token ids
        caption_mask: [max_len] caption attention mask
        caption_label: str 原始 caption 文本
        vid: str 视频ID
        sen_id: int 句子ID
    """
    
    def __init__(self, features_path, annotations_path, split='train', tkz_type='clip'):
        self.data_split = split
        super(MSVD_FeaturesDataset, self).__init__()
        
        # 1. 加载视频特征
        print(f"[MSVD_FeaturesDataset] Loading features from: {features_path}")
        with open(features_path, 'rb') as f:
            self.features = pickle.load(f)
        print(f"  -> Loaded {len(self.features)} video features")
        
        # 2. 优先根据官方 train/val/test.txt 中的 video_id 做划分
        base_dir = os.path.dirname(annotations_path)
        split_ids = None

        def _load_split_ids(path):
            ids = set()
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(" ", 1)
                    vid = parts[0]
                    ids.add(vid)
            return ids

        train_ids_path = os.path.join(base_dir, "train.txt")
        val_ids_path = os.path.join(base_dir, "val.txt")
        test_ids_path = os.path.join(base_dir, "test.txt")

        if os.path.exists(train_ids_path) and os.path.exists(val_ids_path) and os.path.exists(test_ids_path):
            train_ids = _load_split_ids(train_ids_path)
            val_ids = _load_split_ids(val_ids_path)
            test_ids = _load_split_ids(test_ids_path)

            if split == 'train':
                split_ids = train_ids
            elif split == 'val':
                split_ids = val_ids
            elif split == 'test':
                split_ids = test_ids
            elif split == 'trainval':
                split_ids = train_ids.union(val_ids)
            elif split == 'all':
                split_ids = train_ids.union(val_ids).union(test_ids)
            else:
                raise ValueError(f'Invalid split: {split}. Valid options: [train, val, test, all, trainval]')
        else:
            # 如果缺少 train/val/test.txt，则回退到旧的 1200/100/670 划分逻辑
            split_ids = None

        # 3. 加载标注文件并解析
        print(f"[MSVD_FeaturesDataset] Loading annotations from: {annotations_path}")
        self.captions_data = defaultdict(list)  # {video_id: [(caption, sen_id), ...]}
        self.all_video_ids = set()
        
        with open(annotations_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 格式: video_id sen_id caption
                parts = line.split(' ', 2)
                if len(parts) < 3:
                    continue
                video_id, sen_id, caption = parts
                sen_id = int(sen_id)

                # 如果提供了官方划分，则只保留属于当前 split 的 video
                if (split_ids is not None) and (video_id not in split_ids):
                    continue

                self.all_video_ids.add(video_id)
                self.captions_data[video_id].append((caption, sen_id))

        # 4. 确定最终使用的视频 ID 列表
        if split_ids is not None:
            # 按照 train/val/test.txt 中的 video_id 与实际存在的标注取交集
            self.video_ids = sorted(list(self.captions_data.keys()))
            total_videos = len(self.video_ids)
            print(f"  -> Split '{split}': {total_videos} videos (from train/val/test.txt)")
        else:
            # 回退逻辑：按全部 video_id 字典序划分 1200/100/剩余
            sorted_video_ids = sorted(list(self.all_video_ids))
            total_videos = len(sorted_video_ids)
            print(f"  -> Total videos: {total_videos}")
            
            # MSVD 常见划分: train 1200, val 100, test 670
            train_end = 1200
            val_end = 1200 + 100
            
            if split == 'train':
                self.video_ids = sorted_video_ids[:train_end]
            elif split == 'val':
                self.video_ids = sorted_video_ids[train_end:val_end]
            elif split == 'test':
                self.video_ids = sorted_video_ids[val_end:]
            elif split == 'all':
                self.video_ids = sorted_video_ids
            elif split == 'trainval':
                self.video_ids = sorted_video_ids[:val_end]
            else:
                raise ValueError(f'Invalid split: {split}. Valid options: [train, val, test, all, trainval]')
            
            print(f"  -> Split '{split}': {len(self.video_ids)} videos (fallback 1200/100/670)")
        
        # 4. 初始化 tokenizer
        self.tkz_type = tkz_type
        if self.tkz_type == 'clip':
            self.max_len = 77
            from load_tokenizers import CLIPTokenizer_Custom
            self.tokenizer = CLIPTokenizer_Custom()
        elif self.tkz_type == 'bert':
            self.max_len = 73
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        else:
            raise ValueError(f"Invalid tokenizer type: {tkz_type}. Valid options: ['clip', 'bert']")
        
        # 5. 构建数据列表
        self.data_list = []
        missing_features = 0
        
        for vid in self.video_ids:
            # 检查是否有对应的特征
            if vid not in self.features:
                missing_features += 1
                continue
                
            if vid in self.captions_data:
                if self.data_split in ['train', 'all', 'trainval']:
                    # 训练时使用所有 caption
                    for (cap, sen_id) in self.captions_data[vid]:
                        self.data_list.append((vid, cap, sen_id))
                else:
                    # 测试/验证时只用第一个 caption
                    cap, sen_id = self.captions_data[vid][0]
                    self.data_list.append((vid, cap, sen_id))
        
        if missing_features > 0:
            print(f"  -> Warning: {missing_features} videos missing features")
        print(f"  -> Total samples: {len(self.data_list)}")

    def get_references(self):
        """获取所有参考 captions 用于评估"""
        refs = {}
        for vid in self.video_ids:
            if vid in self.captions_data:
                caps = [c for (c, _sid) in self.captions_data[vid]]
                refs[vid] = caps if caps else [""]
            else:
                refs[vid] = [""]
        return refs

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        vid, caption, sen_id = self.data_list[idx]
        
        # 获取视频特征和 mask
        feat, vid_mask = self.features[vid]  # feat: (T, D), vid_mask: (T,)
        vid_feat_tensor = torch.tensor(feat, dtype=torch.float32)
        vid_mask_tensor = torch.tensor(vid_mask, dtype=torch.long)
        
        # 对 caption 进行分词编码
        if self.tkz_type == 'bert':
            encoded = self.tokenizer(
                caption,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            caption_ids = encoded['input_ids'].squeeze(0)
            caption_mask = encoded['attention_mask'].squeeze(0)
        elif self.tkz_type == 'clip':
            encoded = self.tokenizer.encode_plus(
                caption,
                padding='max_length',
                max_length=self.max_len,
                truncation=True,
                return_tensors='pt'
            )
            caption_ids = encoded['input_ids'].squeeze(0)
            caption_mask = encoded['attention_mask'].squeeze(0)
        
        caption_label = caption
        
        return vid_feat_tensor, vid_mask_tensor, caption_ids, caption_mask, caption_label, vid, sen_id


class MSVD_NounVectorDataset(Dataset):
    def __init__(
        self,
        features_path,
        annotations_path,
        noun_vectors_dir=None,
        split='train',
        vocab_size: int = 49412,
        missing_noun_vector: str = 'skip',
    ):
        self.data_split = split
        super(MSVD_NounVectorDataset, self).__init__()
        self.vocab_size = int(vocab_size)
        self.missing_noun_vector = missing_noun_vector

        print(f"[MSVD_NounVectorDataset] Loading features from: {features_path}")
        with open(features_path, 'rb') as f:
            self.features = pickle.load(f)
        print(f"  -> Loaded {len(self.features)} video features")

        base_dir = os.path.dirname(annotations_path)

        def _load_split_ids(path):
            ids = set()
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(" ", 1)
                    vid = parts[0]
                    ids.add(vid)
            return ids

        train_ids_path = os.path.join(base_dir, "train.txt")
        val_ids_path = os.path.join(base_dir, "val.txt")
        test_ids_path = os.path.join(base_dir, "test.txt")

        split_ids = None
        search_splits = None
        if split in ['train', 'val', 'test']:
            search_splits = [split]
        elif split == 'trainval':
            search_splits = ['train', 'val']
        elif split == 'all':
            search_splits = ['train', 'val', 'test']
        else:
            raise ValueError(f"Invalid split: {split}. Valid options: [train, val, test, all, trainval]")

        if os.path.exists(train_ids_path) and os.path.exists(val_ids_path) and os.path.exists(test_ids_path):
            train_ids = _load_split_ids(train_ids_path)
            val_ids = _load_split_ids(val_ids_path)
            test_ids = _load_split_ids(test_ids_path)

            if split == 'train':
                split_ids = train_ids
            elif split == 'val':
                split_ids = val_ids
            elif split == 'test':
                split_ids = test_ids
            elif split == 'trainval':
                split_ids = train_ids.union(val_ids)
            elif split == 'all':
                split_ids = train_ids.union(val_ids).union(test_ids)
        else:
            split_ids = None

        if noun_vectors_dir is None:
            candidate_roots = [
                os.path.join(base_dir, 'annotations', 'nouns', 'noun_vectors'),
                os.path.join(base_dir, 'annotations', 'noun_vectors'),
            ]
        else:
            candidate_roots = [noun_vectors_dir]

        self.video_ids = []
        self.noun_vector_paths = {}

        all_video_ids = set()
        if split_ids is None:
            with open(annotations_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(' ', 2)
                    if len(parts) < 3:
                        continue
                    video_id = parts[0]
                    all_video_ids.add(video_id)

            sorted_video_ids = sorted(list(all_video_ids))
            train_end = 1200
            val_end = 1200 + 100
            if split == 'train':
                split_ids = set(sorted_video_ids[:train_end])
            elif split == 'val':
                split_ids = set(sorted_video_ids[train_end:val_end])
            elif split == 'test':
                split_ids = set(sorted_video_ids[val_end:])
            elif split == 'all':
                split_ids = set(sorted_video_ids)
            elif split == 'trainval':
                split_ids = set(sorted_video_ids[:val_end])

        missing_features = 0
        missing_vectors = 0
        for vid in sorted(list(split_ids)):
            if vid not in self.features:
                missing_features += 1
                continue

            vec_path = None
            for root in candidate_roots:
                for sp in search_splits:
                    p = os.path.join(root, sp, f"{vid}.npy")
                    if os.path.exists(p):
                        vec_path = p
                        break
                if vec_path is not None:
                    break

            if vec_path is None:
                missing_vectors += 1
                if self.missing_noun_vector == 'skip':
                    continue
                elif self.missing_noun_vector == 'zeros':
                    vec_path = None
                else:
                    raise ValueError("missing_noun_vector must be one of ['skip', 'zeros']")

            self.video_ids.append(vid)
            self.noun_vector_paths[vid] = vec_path

        if missing_features > 0:
            print(f"  -> Warning: {missing_features} videos missing features")
        if missing_vectors > 0:
            print(f"  -> Warning: {missing_vectors} videos missing noun_vectors")
        print(f"  -> Split '{split}': {len(self.video_ids)} videos")

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]

        feat, vid_mask = self.features[vid]
        vid_feat_tensor = torch.tensor(feat, dtype=torch.float32)
        vid_mask_tensor = torch.tensor(vid_mask, dtype=torch.long)

        vec_path = self.noun_vector_paths.get(vid)
        if vec_path is None:
            noun_vector = torch.zeros(self.vocab_size, dtype=torch.float32)
        else:
            vec = np.load(vec_path)
            if vec.shape[0] != self.vocab_size:
                raise ValueError(f"noun_vector vocab mismatch: expected {self.vocab_size}, got {vec.shape[0]} for {vid}")
            noun_vector = torch.from_numpy(vec).float()

        return vid_feat_tensor, vid_mask_tensor, noun_vector, vid


class MSVD_NounVectorCompactDataset(Dataset):
    """
    名词聚焦版本的数据集 - 只输出实际使用的名词子集

    使用 noun_vocab.json 将 49412 维向量压缩到 ~3000 维

    Args:
        features_path: 视频特征 pickle 文件路径
        annotations_path: 预处理后的标注文件路径
        noun_vectors_dir: 名词向量目录
        noun_vocab_path: 名词词表映射文件路径 (noun_vocab.json)
        split: 数据集划分
        missing_noun_vector: 缺失向量处理方式

    Returns:
        vid_feat_tensor: [T, D] 视频特征
        vid_mask_tensor: [T] 视频掩码
        noun_vector: [N] 紧凑名词向量 (N = noun_count)
        vid: str 视频ID
    """
    def __init__(
        self,
        features_path,
        annotations_path,
        noun_vectors_dir=None,
        noun_vocab_path=None,
        split='train',
        missing_noun_vector: str = 'skip',
    ):
        import json

        self.data_split = split
        super(MSVD_NounVectorCompactDataset, self).__init__()
        self.missing_noun_vector = missing_noun_vector

        # 加载名词词表
        if noun_vocab_path is None:
            # 默认路径
            noun_vocab_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'multi_stage_caption', 'noun_vocab.json'
            )

        print(f"[MSVD_NounVectorCompactDataset] Loading noun vocab from: {noun_vocab_path}")
        with open(noun_vocab_path, 'r', encoding='utf-8') as f:
            noun_vocab = json.load(f)

        self.clip_to_noun = {int(k): v for k, v in noun_vocab['clip_to_noun'].items()}
        self.noun_to_clip = noun_vocab['noun_to_clip']  # list
        self.noun_count = noun_vocab['noun_count']
        self.clip_vocab_size = noun_vocab['clip_vocab_size']
        print(f"  -> Noun vocabulary: {self.noun_count} nouns (from {self.clip_vocab_size} vocab)")

        # 加载视频特征
        print(f"[MSVD_NounVectorCompactDataset] Loading features from: {features_path}")
        with open(features_path, 'rb') as f:
            self.features = pickle.load(f)
        print(f"  -> Loaded {len(self.features)} video features")

        base_dir = os.path.dirname(annotations_path)

        def _load_split_ids(path):
            ids = set()
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(" ", 1)
                    vid = parts[0]
                    ids.add(vid)
            return ids

        train_ids_path = os.path.join(base_dir, "train.txt")
        val_ids_path = os.path.join(base_dir, "val.txt")
        test_ids_path = os.path.join(base_dir, "test.txt")

        split_ids = None
        search_splits = None
        if split in ['train', 'val', 'test']:
            search_splits = [split]
        elif split == 'trainval':
            search_splits = ['train', 'val']
        elif split == 'all':
            search_splits = ['train', 'val', 'test']
        else:
            raise ValueError(f"Invalid split: {split}. Valid options: [train, val, test, all, trainval]")

        if os.path.exists(train_ids_path) and os.path.exists(val_ids_path) and os.path.exists(test_ids_path):
            train_ids = _load_split_ids(train_ids_path)
            val_ids = _load_split_ids(val_ids_path)
            test_ids = _load_split_ids(test_ids_path)

            if split == 'train':
                split_ids = train_ids
            elif split == 'val':
                split_ids = val_ids
            elif split == 'test':
                split_ids = test_ids
            elif split == 'trainval':
                split_ids = train_ids.union(val_ids)
            elif split == 'all':
                split_ids = train_ids.union(val_ids).union(test_ids)
        else:
            split_ids = None

        if noun_vectors_dir is None:
            candidate_roots = [
                os.path.join(base_dir, 'annotations', 'nouns', 'noun_vectors'),
                os.path.join(base_dir, 'annotations', 'noun_vectors'),
            ]
        else:
            candidate_roots = [noun_vectors_dir]

        self.video_ids = []
        self.noun_vector_paths = {}

        all_video_ids = set()
        if split_ids is None:
            with open(annotations_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(' ', 2)
                    if len(parts) < 3:
                        continue
                    video_id = parts[0]
                    all_video_ids.add(video_id)

            sorted_video_ids = sorted(list(all_video_ids))
            train_end = 1200
            val_end = 1200 + 100
            if split == 'train':
                split_ids = set(sorted_video_ids[:train_end])
            elif split == 'val':
                split_ids = set(sorted_video_ids[train_end:val_end])
            elif split == 'test':
                split_ids = set(sorted_video_ids[val_end:])
            elif split == 'all':
                split_ids = set(sorted_video_ids)
            elif split == 'trainval':
                split_ids = set(sorted_video_ids[:val_end])

        missing_features = 0
        missing_vectors = 0
        for vid in sorted(list(split_ids)):
            if vid not in self.features:
                missing_features += 1
                continue

            vec_path = None
            for root in candidate_roots:
                for sp in search_splits:
                    p = os.path.join(root, sp, f"{vid}.npy")
                    if os.path.exists(p):
                        vec_path = p
                        break
                if vec_path is not None:
                    break

            if vec_path is None:
                missing_vectors += 1
                if self.missing_noun_vector == 'skip':
                    continue
                elif self.missing_noun_vector == 'zeros':
                    vec_path = None
                else:
                    raise ValueError("missing_noun_vector must be one of ['skip', 'zeros']")

            self.video_ids.append(vid)
            self.noun_vector_paths[vid] = vec_path

        if missing_features > 0:
            print(f"  -> Warning: {missing_features} videos missing features")
        if missing_vectors > 0:
            print(f"  -> Warning: {missing_vectors} videos missing noun_vectors")
        print(f"  -> Split '{split}': {len(self.video_ids)} videos")

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]

        feat, vid_mask = self.features[vid]
        vid_feat_tensor = torch.tensor(feat, dtype=torch.float32)
        vid_mask_tensor = torch.tensor(vid_mask, dtype=torch.long)

        vec_path = self.noun_vector_paths.get(vid)
        if vec_path is None:
            # 返回紧凑的零向量
            noun_vector = torch.zeros(self.noun_count, dtype=torch.float32)
        else:
            # 加载原始 49412 维向量，转换为紧凑向量
            full_vec = np.load(vec_path)
            compact_vec = np.zeros(self.noun_count, dtype=np.float32)

            # 将 CLIP 索引映射到紧凑索引
            clip_indices = np.where(full_vec > 0)[0]
            for clip_idx in clip_indices:
                if clip_idx in self.clip_to_noun:
                    noun_idx = self.clip_to_noun[clip_idx]
                    compact_vec[noun_idx] = full_vec[clip_idx]

            noun_vector = torch.from_numpy(compact_vec)

        return vid_feat_tensor, vid_mask_tensor, noun_vector, vid

    def get_noun_clip_indices(self):
        """返回名词对应的 CLIP 索引列表，用于模型获取 CLIP embedding"""
        return self.noun_to_clip


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == '__main__':
    print("--- 测试 MSVD_FeaturesDataset ---")
    
    # 路径配置
    features_path = "../../datasets/MSVD/feats/ViT-B-32_k_split_ks12_features.pickle"
    annotations_path = "../../datasets/MSVD/annotations_preprocessed.txt"
    
    # 检查文件是否存在
    if not os.path.exists(features_path):
        print(f"特征文件不存在: {features_path}")
        print("请先运行 extract_clip_feats.py 提取特征")
        exit(1)
    
    if not os.path.exists(annotations_path):
        print(f"标注文件不存在: {annotations_path}")
        print("请先运行 preprocess_annotations.py 生成预处理标注")
        exit(1)
    
    # 测试各个 split
    for split in ['train', 'val', 'test']:
        print(f"\n{'='*50}")
        print(f"Testing split: {split}")
        print('='*50)
        
        dataset = MSVD_FeaturesDataset(
            features_path=features_path,
            annotations_path=annotations_path,
            split=split,
            tkz_type='clip'
        )
        
        # 获取第一个样本
        sample = dataset[0]
        vid_feat, vid_mask, cap_ids, cap_mask, cap_label, vid, sen_id = sample
        
        print(f"\n第一个样本:")
        print(f"  视频ID: {vid}")
        print(f"  句子ID: {sen_id}")
        print(f"  Caption: {cap_label}")
        print(f"  特征形状: {vid_feat.shape}")
        print(f"  掩码形状: {vid_mask.shape}")
        print(f"  有效帧数: {vid_mask.sum().item()}")
        print(f"  Caption IDs 形状: {cap_ids.shape}")
        print(f"  Caption Mask 形状: {cap_mask.shape}")
