import pickle
import json
import torch
import copy
import time
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer,pipeline
import sys
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 延迟导入，避免模块级别执行
# from load_tokenizers import CLIPTokenizer_Custom

# 原始原始dataset加载
class MSRVTT_FeaturesDataset(Dataset):
    def __init__(self, features_path, json_path, nous_feats_path=None,split='train', tkz_type='clip'):
        '''
        :param features_path: vision features path
        :param json_path:
        :param split:
        :param tkz_tpye: ['clip','bert'] #bert_base_uncased
        '''
        self.data_split = split
        super(MSRVTT_FeaturesDataset, self).__init__()
        # 从 pickle 文件中加载视频特征和 mask
        with open(features_path, 'rb') as f:
            self.features = pickle.load(f)
        # 从 JSON 文件中加载视频字幕信息
        with open(json_path, 'r') as f:
            data = json.load(f)
        # 根据指定的划分（split）筛选对应的视频ID列表
        self.video_ids = []
        # Train: video0 : video6512 (6513)
        # Val: video6513 : video7009 (497)
        # Test: video7010 : video9999 (2990)

        if split == 'train':
            self.video_ids = ['video' + str(i) for i in range(0, 6512 + 1)]
        elif split == 'val':
            self.video_ids = ['video' + str(i) for i in range(6513, 7009 + 1)]
        elif split == 'test':
            self.video_ids = ['video' + str(i) for i in range(7010, 9999 + 1)]
        elif split == 'all':
            self.video_ids = ['video' + str(i) for i in range(0, 9999 + 1)]
        elif split == 'testval':
            self.video_ids = ['video' + str(i) for i in range(0, 7009+1)]
        else:
            raise ValueError('Invalid datasets split, the valid options are [train, val, test]')
        # ----------------------------存下所有caption和vid
        self.captions_data = {}#这个已经报
        # JSON 可能将字幕单独存放在 'sentences' 或 'annotations' 中
        if 'sentences' in data:
            for sent in data['sentences']:
                vid = sent['video_id']
                cap = sent['caption']
                if vid in self.video_ids:
                    self.captions_data.setdefault(vid, []).append((cap,sent['sen_id']))
        # {'video0':[(str_cap1,id1),(str_cap2,id2),str_cap3]}
        else:
            # 如果 JSON 中未找到字幕，抛出错误
            raise ValueError("Captions not found in JSON data")

        self.tkz_type = tkz_type
        if self.tkz_type == 'clip':
            self.max_len = 77
            # 延迟导入
            from load_tokenizers import CLIPTokenizer_Custom
            self.tokenizer = CLIPTokenizer_Custom()
        elif self.tkz_type == 'bert':
            self.max_len = 73
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        else:
            raise ValueError("Invalid tokenizer type, the valid options are ['clip','bert']")

        # 构建数据列表，每项为 (train_list.csv, caption文本)
        self.data_list = []

        for vid in self.video_ids:
            if vid in self.captions_data:
                if self.data_split in ['train','all','trainval']:
                        # 训练集时，每个视频随机选择一个 caption
                    for (cap,sen_id) in self.captions_data[vid]:
                        # ('video0',cap0-1，cap0-1_id)
                        # ('video0',cap0-2,cap0-2_id)
                        # ... 130260
                        self.data_list.append((vid, cap,sen_id))
                else:#测试的时候是不需要用到那么多句子的
                    cap,sen_id = self.captions_data[vid][0]#(cap,cap_id)
                    self.data_list.append((vid,cap,sen_id))


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        vid, caption,sen_id = self.data_list[idx]#这个主要针对训练，测试的时候直接加载上面的那个data_captions字典
        # 获取视频帧特征和 mask
        feat, vid_mask = self.features[vid]  # feat: numpy数组 (20,512), vid_mask: numpy数组 (20,)
        # 转换为 PyTorch 张量
        vid_feat_tensor = torch.tensor(feat, dtype=torch.float32)
        vid_mask_tensor = torch.tensor(vid_mask, dtype=torch.long)

        # 对 caption 文本进行分词编码
        if self.tkz_type == 'bert':
            # BERT tokenizer returns a dictionary with tensors
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
            # 使用encode_plus方法进行编码，与BERT tokenizer处理方式类似
            encoded = self.tokenizer.encode_plus(
                caption,
                padding='max_length',
                max_length=self.max_len,
                truncation=True,
                return_tensors='pt'
            )
            # 直接从encode_plus返回的字典中提取张量
            caption_ids = encoded['input_ids'].squeeze(0)
            caption_mask = encoded['attention_mask'].squeeze(0)

        caption_label = caption
        # print(caption_ids)

        # 每个nous是不一样的
        return vid_feat_tensor, vid_mask_tensor, caption_ids, caption_mask, caption_label, vid, sen_id


# 适配多种模型的数据集加载
class MSRVTT_FeaturesDataset_MulStage(Dataset):
    def __init__(self,
                 features_global_path=None,
                 features_local_dir=None,
                 json_path=None,
                 all_nouns_list_path=None,#整个的所有nouns词表，顺序和feats对应
                 all_nouns_feats_list_path = None,#nouns的feats，顺序和feats对应
                 vids_nouns_path=None,#一阶段GT,一个视频20条caption的所有nouns
                 vids_sent_nouns_path = None,# GT一个视频单句的nouns
                 vids_sent_verbs_path = None,# GT一个视频单句的verb
                 vids_sent_nouns_verbs_path = None,# 一GT一个视频单句的名词和动词穿插
                 all_nouns_freq_weights_path = None,
                 split='train'):
        '''
        :param features_path: vision features path
        :param json_path:
        :param split:
        :param tkz_tpye: ['clip','bert'] #bert_base_uncased
        '''
        self.data_split = split
        self.features_local_dir = features_local_dir
        super(MSRVTT_FeaturesDataset_MulStage, self).__init__()

        self.all_nouns_list = None
        self.nouns_vocab_size = 20507#20507是MSRVTT提取的名词表大小，区分形式man和men是两个单词
        if all_nouns_list_path is not None:
            self.all_nouns_list = json.load(open(all_nouns_list_path, 'r'))
            self.nouns_vocab_size = len(self.all_nouns_list)

        self.nouns_freq_weights = None
        self.noun_weights =  None
        # 在现有加载代码后添加
        if all_nouns_freq_weights_path is not None:
            print(f"[{self.__class__.__name__}] Loading noun frequency weights from: {all_nouns_freq_weights_path}")
            with open(all_nouns_freq_weights_path, 'r') as f:
                self.nouns_freq_weights = json.load(f)

            # 构建权重张量（按照all_nouns_list的顺序）
            self.noun_weights = torch.zeros(self.nouns_vocab_size)
            for i, noun in enumerate(self.all_nouns_list):
                if noun in self.nouns_freq_weights:
                    self.noun_weights[i] = self.nouns_freq_weights[noun]['weight']
                else:
                    # 未见过的词给默认权重
                    self.noun_weights[i] = 1.0


        self.features_gloabl = None
        # 从 pickle 文件中加载视频特征和 mask
        if features_global_path is not None:
            print(f"[{self.__class__.__name__}] Loading CLIP Global features from: {features_global_path}")
            with open(features_global_path, 'rb') as f:
                self.features_gloabl = pickle.load(f)
                # 从 pickle 文件中加载视频特征和 mask
        self.vids_sent_nouns = None
        self.vids_sent_nouns_verbs = None
        self.all_nouns_feats = None
        if vids_sent_nouns_path is not None:
            print(f"[{self.__class__.__name__}] Loading vids-sent nouns from: {vids_sent_nouns_path}")
            with open(vids_sent_nouns_path, 'r') as f:
                self.vids_sent_nouns = json.load(f)
        if vids_sent_nouns_verbs_path is not None:
            print(f"[{self.__class__.__name__}] Loading vids-sent nouns-verbs from: {vids_sent_nouns_verbs_path}")
            with open(vids_sent_nouns_verbs_path, 'r') as f:
                self.vids_sent_nouns_verbs = json.load(f)
        if all_nouns_feats_list_path is not None:
            print(f"[{self.__class__.__name__}] Loading all nouns embedding from: {all_nouns_feats_list_path}")
            with open(all_nouns_feats_list_path, 'rb') as f:
                self.all_nouns_feats = pickle.load(f)
        # 从 JSON 文件中加载视频字幕信息
        with open(json_path, 'r') as f:
            data = json.load(f)
        # 根据指定的划分（split）筛选对应的视频ID列表
        self.video_ids = []
        # Train: video0 : video6512 (6513)
        # Val: video6513 : video7009 (497)
        # Test: video7010 : video9999 (2990)
        if split == 'train':
            self.video_ids = ['video' + str(i) for i in range(0, 6512 + 1)]
        elif split == 'val':
            self.video_ids = ['video' + str(i) for i in range(6513, 7009 + 1)]
        elif split == 'test':
            self.video_ids = ['video' + str(i) for i in range(7010, 9999 + 1)]
        elif split == 'all':
            self.video_ids = ['video' + str(i) for i in range(0, 9999 + 1)]
        elif split == 'testval':
            self.video_ids = ['video' + str(i) for i in range(0, 7009+1)]
        else:
            raise ValueError('Invalid datasets split, the valid options are [train, val, test]')
        # ----------------------------存下所有caption和vid
        self.captions_data = {}#这个已经报
        # JSON 可能将字幕单独存放在 'sentences' 或 'annotations' 中
        if 'sentences' in data:
            for sent in data['sentences']:
                vid = sent['video_id']
                cap = sent['caption']
                if vid in self.video_ids:
                    self.captions_data.setdefault(vid, []).append((cap,sent['sen_id']))
        # {'video0':[(str_cap1,id1),(str_cap2,id2),str_cap3]}
        else:
            # 如果 JSON 中未找到字幕，抛出错误
            raise ValueError("Captions not found in JSON data")

        self.max_len = 77
        # 延迟导入
        from load_tokenizers import CLIPTokenizer_Custom
        self.tokenizer = CLIPTokenizer_Custom()



        self.data_list = []
        self.vids_nouns_dict = None
        if vids_nouns_path is not None:
            self.vids_nouns_dict = json.load(open(vids_nouns_path, 'r'))
        for vid in self.video_ids:
            if vid in self.captions_data:
                if self.data_split in ['train','all','trainval']:
                        # 训练集时，每个视频随机选择一个 caption
                    for (cap,sen_id) in self.captions_data[vid]:
                        # ('video0',cap0-1，cap0-1_id)
                        # ('video0',cap0-2,cap0-2_id)
                        # ... 130260
                        self.data_list.append((vid, cap,sen_id))
                else:#测试的时候是不需要用到那么多句子的
                    cap,sen_id = self.captions_data[vid][0]#(cap,cap_id)
                    self.data_list.append((vid,cap,sen_id))


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        vid_str, caption,sen_id_int = self.data_list[idx]#这个主要针对训练，测试的时候直接加载上面的那个data_captions字典
        # 获取视频帧特征和 mask
        feat, vid_mask = self.features_gloabl[vid_str]  # feat: numpy数组 (20,512), vid_mask: numpy数组 (20,)
        # 转换为 PyTorch 张量
        feat_tensor = torch.tensor(feat, dtype=torch.float32)

        # feat的形状现在是 (20, 50, 768), vid_mask的形状是 (20,)
        if self.features_local_dir is not None:
            with open(os.path.join(self.features_local_dir, vid_str + '.pickle'), 'rb') as f:
                self.features_local = pickle.load(f)
            feat_local, vid_mask_local = self.features_local
            # 转换为 PyTorch 张量
            feat_tensor_local = torch.tensor(feat_local, dtype=torch.float32)
            vid_mask_tensor_local = torch.tensor(vid_mask_local, dtype=torch.long)
        else:
            feat_tensor_local = torch.zeros((feat.shape[0], 50, 768), dtype=torch.float32)
            vid_mask_tensor_local = torch.zeros((feat.shape[0],), dtype=torch.long)

        vid_mask_tensor = torch.tensor(vid_mask, dtype=torch.long)

        encoded = self.tokenizer.encode_plus(
            caption,
            padding='max_length',
            max_length=self.max_len,
            truncation=True,
            return_tensors='pt'
        )
        # 直接从encode_plus返回的字典中提取张量
        caption_ids_tensor = encoded['input_ids'].squeeze(0)
        caption_mask_tensor = encoded['attention_mask'].squeeze(0)
        caption_label_str = caption
        # print(caption_ids_tensor)

        vid_captions_nouns_tensor_label_GT = torch.tensor([0] * self.nouns_vocab_size, dtype=torch.long)
        psedo_caption_ids_tensor = torch.tensor([0] * self.max_len, dtype=torch.long)
        psedo_caption_mask_tensor = torch.tensor([0] * self.max_len, dtype=torch.long)
        vid_psedo_nouns_caption_ids_tensor = torch.tensor([0] * self.max_len, dtype=torch.long)
        vid_psedo_nouns_caption_mask_tensor = torch.tensor([0] * self.max_len, dtype=torch.long)
        vid_sent_psedo_nouns_caption_ids_tensor = torch.tensor([0] * self.max_len, dtype=torch.long)
        vid_sent_psedo_nouns_caption_mask_tensor = torch.tensor([0] * self.max_len, dtype=torch.long)
        if self.data_split == 'train':
            if self.vids_nouns_dict is not None:
                # 名词多标签GT
                pseudo_nouns_caption_str = " ".join(self.vids_nouns_dict[vid_str])
                pseudo_nouns_caption_ids = self.tokenizer.encode_plus(pseudo_nouns_caption_str,
                                                                  padding='max_length',
                                                                  max_length=self.max_len,
                                                                  truncation=True,
                                                                  return_tensors='pt')
                vid_psedo_nouns_caption_ids_tensor = pseudo_nouns_caption_ids['input_ids'].squeeze(0)
                vid_psedo_nouns_caption_mask_tensor = pseudo_nouns_caption_ids['attention_mask'].squeeze(0)

            if self.vids_sent_nouns is not None:
                sent_pseudo_nouns_caption_str = " ".join(self.vids_sent_nouns[vid_str][str(sen_id_int)])
                # print(vid_str)
                # print("caption=",caption_label_str)
                # print("nouns =",sent_pseudo_nouns_caption_str)
                sent_pseudo_nouns_caption_ids = self.tokenizer.encode_plus(sent_pseudo_nouns_caption_str,
                                                                  padding='max_length',
                                                                  max_length=self.max_len,
                                                                  truncation=True,
                                                                  return_tensors='pt')
                vid_sent_psedo_nouns_caption_ids_tensor = sent_pseudo_nouns_caption_ids['input_ids'].squeeze(0)
                vid_sent_psedo_nouns_caption_mask_tensor = sent_pseudo_nouns_caption_ids['attention_mask'].squeeze(0)

            # print("caption_ids=",vid_sent_psedo_nouns_caption_ids_tensor)
            # print ("caption_mask=",vid_sent_psedo_nouns_caption_mask_tensor)
            # 这个是一个视频的所有caption的名词,的多标签[0,1,1,0,1,.....]存在的名词对应位置为1
            if self.vids_nouns_dict is not None:
                vid_captions_nouns_label_GT = [0] * self.nouns_vocab_size
                for noun in self.vids_nouns_dict[vid_str]:
                    if noun in self.all_nouns_list:
                        vid_captions_nouns_label_GT[self.all_nouns_list.index(noun)] = 1
                vid_captions_nouns_tensor_label_GT = torch.tensor(vid_captions_nouns_label_GT, dtype=torch.long)#[20507]:[1,0,0,1,0,...,1]

            if self.vids_sent_nouns_verbs is not None:
                # 这个是单个句子的名词，多个名词链接在一起，具体表现形式是tokenzie以后的tensors
                #动词+名词多标签GT
                nouns_verbs_psedo_caption = " ".join(self.vids_sent_nouns_verbs[vid_str][str(sen_id_int)])
                encoded_nouns_verbs_psedo_caption = self.tokenizer.encode_plus(
                    nouns_verbs_psedo_caption,
                    padding='max_length',
                    max_length=self.max_len,
                    truncation=True,
                    return_tensors='pt'
                )
                psedo_caption_ids_tensor = encoded_nouns_verbs_psedo_caption['input_ids'].squeeze(0)
                psedo_caption_mask_tensor = encoded_nouns_verbs_psedo_caption['attention_mask'].squeeze(0)



        # 每个nous是不一样的

        # print("caption=",caption_label_str)
        # print("caption ids=",caption_ids_tensor[:15])
        # print("nouns=",sent_pseudo_nouns_caption_str)
        # print("nouns ids=",vid_sent_psedo_nouns_caption_ids_tensor[:15])
        # print("nouns verbs=",nouns_verbs_psedo_caption)
        # print("nouns verbs ids=",psedo_caption_ids_tensor[:15])
        return (feat_tensor, vid_mask_tensor, feat_tensor_local,vid_mask_tensor_local,
                # 20是帧，50是一个CLIP第6层的CLS token和其余patches tokens
                # 全局[20,512],[20,512] CLIP的第6个Block的输出[20,50,512],[20,50,512]

                # 这个是[77,],[77,]输入句子的掩码
                caption_ids_tensor, caption_mask_tensor,

                # 14876长度，本来是准备做名词的，暂时用不上
                vid_captions_nouns_tensor_label_GT,

                # 一阶段名词，单个句子所有名词链接在一起的，具体表现形式是tokenzie以后的tensors
                vid_sent_psedo_nouns_caption_ids_tensor,vid_sent_psedo_nouns_caption_mask_tensor,

                # 一阶段名词，所有video的20条句子名词链接在一起的,具体表现形式是tokenzie以后的tensors
                vid_psedo_nouns_caption_ids_tensor,vid_psedo_nouns_caption_mask_tensor,

                # # 二阶段名词+动词,ids是交叉穿插的，也是tensor ids
                psedo_caption_ids_tensor, psedo_caption_mask_tensor,
                caption_label_str, vid_str, sen_id_int)# 真实句子+video_id_str+句子id


# ===== 新增：Router 离线缓存版数据集 ==========================================
class MSRVTT_RouterCachedDataset(Dataset):
    """
    读取 train 阶段离线缓存（每个视频在各候选长度的句子与CIDEr分数，及最佳长度label_idx），
    返回单条样本（每视频只一条）：
        vid_feat_tensor: [T, D]
        vid_mask_tensor: [T]
        label_idx:       []  (int，最佳候选在 candidate_lengths 里的下标)
        vid:             str
        cand_lengths:    [K] (int list) —— 便于调试，不参与训练
        cand_scores:     [K] (float list) —— 便于调试，不参与训练
    """
    def __init__(self,
                 features_path: str,
                 cache_json_path: str,
                 split: str = 'train'):
        assert split == 'train', "MSRVTT_RouterCachedDataset 仅用于 train。"
        super(MSRVTT_RouterCachedDataset, self).__init__()
        self.split = split
        self.features_path = features_path
        self.cache_json_path = cache_json_path

        # 加载视频特征
        import pickle, json, torch
        with open(features_path, 'rb') as f:
            self.features = pickle.load(f)

        with open(cache_json_path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        self.meta = obj.get("meta", {})
        self.data = obj["data"]  # {vid: {"candidates":[{length,text,cider},...], "label_idx": i}}

        # 仅使用缓存中出现的视频（与 train split 一致）
        self.video_ids = sorted(list(self.data.keys()))
        # 构建 data_list（每视频一个样本）
        self.data_list = []
        for vid in self.video_ids:
            dd = self.data[vid]
            label_idx = int(dd["label_idx"])
            cands = dd["candidates"]
            cand_lengths = [int(x["length"]) for x in cands]
            cand_scores  = [float(x["cider"])  for x in cands]
            self.data_list.append((vid, label_idx, cand_lengths, cand_scores))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        import torch
        vid, label_idx, cand_lengths, cand_scores = self.data_list[idx]

        # 取视频特征
        feat, vid_mask = self.features[vid]  # feat: numpy (T,D), vid_mask: numpy (T,)
        vid_feat_tensor = torch.tensor(feat, dtype=torch.float32)     # [T,D]
        vid_mask_tensor = torch.tensor(vid_mask, dtype=torch.long)    # [T]
        label_tensor    = torch.tensor(label_idx, dtype=torch.long)   # []

        cand_lens_tensor   = torch.tensor(cand_lengths, dtype=torch.long)
        cand_scores_tensor = torch.tensor(cand_scores, dtype=torch.float32)

        return vid_feat_tensor, vid_mask_tensor, label_tensor, vid, cand_lens_tensor, cand_scores_tensor


def get_caption_statistics(split='val'):
    video_ids = ['video' + str(i) for i in range(10000)]
    if split == 'train':
        video_ids = ['video' + str(i) for i in range(0, 6512 + 1)]
    elif split == 'val':
        video_ids = ['video' + str(i) for i in range(6513, 7009 + 1)]
    elif split == 'test':
        video_ids = ['video' + str(i) for i in range(7010, 9999 + 1)]
    """
    计算数据集的统计信息，例如句子长度、单词数量、单词频率等。

当前词汇表大小: 49408
统计数据集： test
总句子数： 59800
句子长度统计[token级别，包括SEP和CLS]：
最大token长度为 70，
最小长度为 5，
平均长度为 11.57


当前词汇表大小: 49408
统计数据集： train
总句子数： 130260
句子长度统计[token级别，包括SEP和CLS]：
最大token长度为 76，
最小长度为 4，
平均长度为 11.53

当前词汇表大小: 49408
统计数据集： val
总句子数： 9940
句子长度统计[token级别，包括SEP和CLS]：
最大token长度为 54，
最小长度为 4，
平均长度为 11.46

    """
    dataset = MSRVTT_FeaturesDataset(
        features_path="../../datasets/MSRVTT/feats/ViT-B-32_k_split_ks12_features.pickle",
        json_path="../../datasets/MSRVTT/MSRVTT_data.json",
        split='all',
    )

    max_token_len = -1
    min_token_len = 9999#单个句子最大token长度【包括开始长度】
    all_caption_tokens_len = 0
    for (vid_feat_tensor, vid_mask_tensor, caption_ids, caption_mask, caption_label, vid, sen_id) in dataset:
        if vid in video_ids:
            single_caption_length = sum(caption_mask.detach().cpu().numpy())
            max_token_len = max(max_token_len, single_caption_length)
            min_token_len = min(min_token_len, single_caption_length)
            all_caption_tokens_len += single_caption_length
    mean_token_len = all_caption_tokens_len / (len(video_ids)*20)

    print("统计数据集：", split)
    print("总句子数：", (len(video_ids)*20))
    print(f"句子长度统计[token级别，包括SEP和CLS]：\n"
          f"最大token长度为 {max_token_len}，\n"
          f"最小长度为 {min_token_len}，\n"
          f"平均长度为 {mean_token_len:.2f}")

if __name__ == '__main__':
    # # 假设你已经有 MSRVTT_FeaturesDataset 的测试代码...
    # print("--- 正在测试 MSRVTT_FeaturesDataset ---")
    # dataset_msrvtt_feats_orig = MSRVTT_FeaturesDataset(
    #     features_path="../../datasets/MSRVTT/feats/ViT-B-32_k_split_ks12_features.pickle",
    #     json_path="../../datasets/MSRVTT/MSRVTT_data.json",
    #     split='train',
    # )
    # a1 = dataset_msrvtt_feats_orig[0]
    # print(f"原始数据集样本:")
    # print(f"特征形状: {a1[0].shape}")
    # print(f"视频掩码形状: {a1[1].shape}")
    # print(f"字幕ID形状: {a1[2].shape}")
    # print(f"字幕掩码形状: {a1[3].shape}")
    # print(f"字幕: {a1[4]}")
    # print(f"视频ID: {a1[5]}")
    # print(f"句子ID: {a1[6]}")
    # print("-" * 30)
    pass

# 注释掉，避免模块级别执行
# get_caption_statistics()


# ============================================================================
# 新增：带对象分类标签的数据集类
# ============================================================================
class MSRVTT_FeaturesDataset_AddClass(Dataset):
    """
    扩展的 MSRVTT 数据集，添加对象分类标签
    
    新增功能：
    - 使用扩展后的 CLIP tokenizer（包含特殊 token）
    - 加载 anno_objs 中的对象标注
    - 返回多标签分类向量 video_objs_idx [vocab_size]
    - 返回对象词列表 video_objs List[str]
    - 返回句子级对象 sentence_objs List[str]
    - 返回带修饰的对象短语 video_objs_with_adj List[List[str]]
    
    依赖要求：
    - 需要先运行 test_we_expand_classifier.py 生成扩展后的 tokenizer
    - 需要先运行 generate_multilabel_vectors_batch.py 生成 video_objs_idx
    
    词表信息：
    - 原始 CLIP 词表：49408 (id: 0-49407)
    - 新增特殊 token：5 个
      * [MASK]    -> 49408
      * [OBJ_CLS] -> 49409
      * [OBJ_END] -> 49410
      * [OBJ_SEP] -> 49411
      * [NO_OBJ]  -> 49412
    - 扩展后词表：49413
    
    使用示例：
    ```python
    # 基础用法（不使用对比学习）
    dataset = MSRVTT_FeaturesDataset_AddClass(
        features_path='path/to/features.pkl',
        json_path='path/to/captions.json',
        anno_objs_dir='path/to/anno_objs',
        updated_tokenizer_dir='path/to/updated_tokenizer',
        split='train',
        use_contrastive=False  # 默认False
    )
    
    # 对比学习用法（返回句子对）
    dataset = MSRVTT_FeaturesDataset_AddClass(
        features_path='path/to/features.pkl',
        json_path='path/to/captions.json',
        anno_objs_dir='path/to/anno_objs',
        anno_cider_path='../datasets/MSRVTT/anno_msrvtt_cider.json',  # 加载CIDEr评分
        use_contrastive=True  # 启用对比学习
    )
    
    # 获取样本（23个返回值）
    (vid_feat, vid_mask, 
     caption_i_ids, caption_i_mask, caption_i_label, sent_i_id,
     caption_j_ids, caption_j_mask, caption_j_label, sent_j_id,
     capscore_i, capscore_j, vid,
     video_objs, video_objs_idx, video_objs_ids, video_objs_with_adj,
     video_objs_selected, video_objs_ids_selected,
     video_phrases_selected, video_phrases_ids_selected,
     video_objs_ids_selected_mask, video_phrases_ids_selected_mask) = dataset[0]
    
    # 训练多标签分类器
    logits = classifier(vid_feat)  # [B, T, D] -> [B, vocab_size]
    loss = F.binary_cross_entropy_with_logits(logits, video_objs_idx)
    
    # 使用选择后的数据（固定64个）
    # video_objs_selected: [64] List[str]
    # video_phrases_ids_selected: [64, 77] Tensor - 固定shape，可直接batch
    decoder_output = decoder(vid_feat, video_phrases_ids_selected)
    ```
    """
    
    def __init__(self, 
                 features_path, 
                 json_path, 
                 anno_objs_dir,  # 新增：anno_objs 目录路径
                 updated_tokenizer_dir=None,  # 新增：扩展后的 tokenizer 目录
                 # 句子质量评分（可选）
                 anno_cider_path=None,  # CIDEr评分JSON路径
                 anno_bleu_path=None,   # BLEU-4评分JSON路径（预留）
                 anno_meteor_path=None, # METEOR评分JSON路径（预留）
                 anno_rouge_path=None,  # ROUGE-L评分JSON路径（预留）
                 anno_spice_path=None,  # SPICE评分JSON路径（预留）
                 nous_feats_path=None, 
                 split='train', 
                 tkz_type='clip',
                 use_contrastive=True,  # 是否使用对比学习（返回句子对）
                 # 预提取特征路径（可选，用于加速训练）
                 precomputed_phrase_dir=None,  # ../datasets/MSRVTT/selected_phrases_feats/{split}/
                 precomputed_sentence_dir=None):  # ../datasets/MSRVTT/selected_sentence_feats/
        """
        Args:
            features_path: 视频特征路径 (pickle)
            json_path: MSRVTT caption JSON 路径
            anno_objs_dir: anno_objs 目录路径（包含对象标注）
            updated_tokenizer_dir: 扩展后的 tokenizer 目录路径
            anno_cider_path: CIDEr评分文件路径（可选）
            use_contrastive: 是否返回句子对（用于对比学习）
            split: 数据集划分 ['train', 'val', 'test', 'all']
            tkz_type: tokenizer 类型，暂时只支持 'clip'（扩展版）
        """
        # 注意：不调用父类 __init__，因为需要使用扩展后的 tokenizer
        # 手动初始化必要的属性
        self.data_split = split
        self.use_contrastive = use_contrastive
        
        # 预提取特征路径
        self.precomputed_phrase_dir = Path(precomputed_phrase_dir) if precomputed_phrase_dir else None
        self.precomputed_sentence_dir = Path(precomputed_sentence_dir) if precomputed_sentence_dir else None
        self.use_precomputed = (self.precomputed_phrase_dir is not None)
        
        if self.use_precomputed:
            print(f"\n[INFO] 启用预提取特征")
            print(f"  Phrase 特征目录: {self.precomputed_phrase_dir}")
            print(f"  Sentence 特征目录: {self.precomputed_sentence_dir}")
            
            # 检查目录是否存在
            if self.precomputed_phrase_dir and not self.precomputed_phrase_dir.exists():
                print(f"  ⚠️ 警告：Phrase 特征目录不存在: {self.precomputed_phrase_dir}")
            elif self.precomputed_phrase_dir:
                phrase_files = list(self.precomputed_phrase_dir.glob("*.npy"))
                print(f"  ✅ Phrase 特征文件数量: {len(phrase_files)}")
            
            if self.precomputed_sentence_dir and not self.precomputed_sentence_dir.exists():
                print(f"  ⚠️ 警告：Sentence 特征目录不存在: {self.precomputed_sentence_dir}")
            elif self.precomputed_sentence_dir:
                sentence_files = list(self.precomputed_sentence_dir.glob("*.npy"))
                print(f"  ✅ Sentence 特征文件数量: {len(sentence_files)}")
            
            # 初始化统计计数器
            self._phrase_loaded_count = 0
            self._phrase_missing_count = 0
            self._sentence_loaded_count = 0
            self._sentence_missing_count = 0
        
        # 1. 加载视频特征（lazy加载，不预转换tensor以节省内存）
        print(f"[INFO] 加载视频特征...")
        with open(features_path, 'rb') as f:
            self.features = pickle.load(f)
        print(f"  ✅ 加载完成")
        
        # 2. 加载 captions
        print(f"[INFO] 加载 captions...")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 3. 根据 split 筛选视频ID
        if split == 'train':
            self.video_ids = ['video' + str(i) for i in range(0, 6512 + 1)]
        elif split == 'val':
            self.video_ids = ['video' + str(i) for i in range(6513, 7009 + 1)]
        elif split == 'test':
            self.video_ids = ['video' + str(i) for i in range(7010, 9999 + 1)]
        elif split == 'all':
            self.video_ids = ['video' + str(i) for i in range(0, 9999 + 1)]
        elif split == 'testval':
            self.video_ids = ['video' + str(i) for i in range(0, 7009+1)]
        else:
            raise ValueError('Invalid split')
        
        # 4. 加载 captions 数据
        self.captions_data = {}
        if 'sentences' in data:
            for sent in data['sentences']:
                vid = sent['video_id']
                cap = sent['caption']
                if vid in self.video_ids:
                    self.captions_data.setdefault(vid, []).append((cap, sent['sen_id']))
        else:
            raise ValueError("Captions not found in JSON data")
        
        # 5. 加载扩展后的 tokenizer（关键！）
        print(f"\n[INFO] 加载扩展后的 CLIP tokenizer...")
        
        if updated_tokenizer_dir is None:
            # 使用默认路径
            updated_tokenizer_dir = './models/clip_tokenizer/models--openai--clip-vit-base-patch32/updated_with_obj_tokens'
        
        print(f"  路径: {updated_tokenizer_dir}")
        
        if not os.path.exists(updated_tokenizer_dir):
            raise FileNotFoundError(
                f"扩展后的 tokenizer 不存在：{updated_tokenizer_dir}\n"
                f"请先运行 test_we_expand_classifier.py 生成扩展后的 tokenizer！"
            )
        
        # 加载扩展后的 tokenizer
        try:
            from transformers import CLIPTokenizerFast as CLIPTokenizer
        except Exception:
            from transformers import CLIPTokenizer
        
        self.tokenizer = CLIPTokenizer.from_pretrained(updated_tokenizer_dir, local_files_only=True)
        
        # 显式设置 pad_token_id = 0（CLIP扩展后的配置）
        if self.tokenizer.pad_token_id != 0:
            original_token_at_0 = self.tokenizer.convert_ids_to_tokens([0])[0]
            self.tokenizer.pad_token = original_token_at_0
            self.tokenizer.pad_token_id = 0
            print(f"  ⚙️  已设置 pad_token_id = 0 ('{original_token_at_0}')")
        
        self.vocab_size = len(self.tokenizer)
        self.max_len = 77
        
        print(f"  ✅ 加载成功")
        print(f"  词表大小: {self.vocab_size}")
        
        # 验证新增 token
        vocab = self.tokenizer.get_vocab()
        special_tokens = ["[MASK]", "[OBJ_CLS]", "[OBJ_END]", "[OBJ_SEP]", "[NO_OBJ]"]
        print(f"  验证特殊 token：")
        for token in special_tokens:
            tid = vocab.get(token, -1)
            if tid >= 0:
                print(f"    ✅ '{token}' -> id={tid}")
            else:
                print(f"    ❌ 警告：'{token}' 不存在！")
        
        # 6. 构建数据列表
        self.data_list = []
        for vid in self.video_ids:
            if vid in self.captions_data:
                if self.data_split in ['train', 'all', 'trainval']:
                    for (cap, sen_id) in self.captions_data[vid]:
                        self.data_list.append((vid, cap, sen_id))
                else:
                    cap, sen_id = self.captions_data[vid][0]
                    self.data_list.append((vid, cap, sen_id))
        
        print(f"  样本总数: {len(self.data_list)}")
        
        # 7. 加载句子质量评分（可选）
        self.anno_cider = None
        if anno_cider_path and os.path.exists(anno_cider_path):
            print(f"\n[INFO] 加载CIDEr评分...")
            print(f"  路径: {anno_cider_path}")
            with open(anno_cider_path, 'r', encoding='utf-8') as f:
                self.anno_cider = json.load(f)
            print(f"  ✅ 加载完成")
        
        # 预留其他评分（未来使用）
        self.anno_bleu = None
        self.anno_meteor = None
        self.anno_rouge = None
        self.anno_spice = None
        
        # 8. 预加载 anno_objs 数据到内存并预计算 token ids
        self.anno_objs_dir = Path(anno_objs_dir)
        self.anno_objs_cache = {}  # {video_id: anno_data} 预加载的数据缓存
        self.video_objs_ids_cache = {}  # {video_id: video_objs_ids} 预计算的对象token ids缓存
        
        # 预构建 id_to_token 映射（用于快速检查，避免每次decode）
        # 注意：这个映射在预计算token ids时使用，但不修改原有的检查逻辑
        print(f"\n[INFO] 构建 token 映射（用于加速预计算）...")
        vocab = self.tokenizer.get_vocab()
        self.id_to_token = {v: k for k, v in vocab.items()}
        print(f"  ✅ 完成（词表大小: {len(self.id_to_token)}）")
        
        print(f"\n[INFO] 预加载 anno_objs 数据到内存并预计算 token ids...")
        print(f"  目录: {self.anno_objs_dir}")
        print(f"  视频数量: {len(self.video_ids)}")
        
        # 预加载所有视频的 anno_objs 数据
        load_start = time.time()
        loaded_count = 0
        missing_count = 0
        error_count = 0
        
        # 定义空数据模板（避免重复创建）
        empty_data_template = {
            "video_objs": [],
            "sentence_objs": {},
            "video_objs_with_adj": [],
            "video_objs_idx": []
        }
        
        # 第一步：加载所有 anno_objs 文件
        for vid in tqdm(self.video_ids, desc="  加载 anno_objs", leave=False):
            anno_path = self.anno_objs_dir / f"{vid}.json"
            
            if anno_path.exists():
                try:
                    with open(anno_path, 'r', encoding='utf-8') as f:
                        anno_data = json.load(f)
                    self.anno_objs_cache[vid] = anno_data
                    loaded_count += 1
                except Exception as e:
                    # 加载失败，使用空数据（深拷贝模板）
                    self.anno_objs_cache[vid] = copy.deepcopy(empty_data_template)
                    self.video_objs_ids_cache[vid] = []
                    error_count += 1
                    if error_count <= 3:  # 只打印前3个错误
                        print(f"\n⚠️ 警告：加载 anno_objs 失败: {anno_path.name}")
                        print(f"  错误: {e}")
            else:
                # 文件不存在，使用空数据（深拷贝模板）
                self.anno_objs_cache[vid] = copy.deepcopy(empty_data_template)
                self.video_objs_ids_cache[vid] = []
                missing_count += 1
        
        load_time = time.time() - load_start
        
        # 第二步：预计算所有视频的 video_objs_ids（优化性能）
        tokenize_start = time.time()
        tokenized_count = 0
        
        for vid in tqdm(self.video_ids, desc="  预计算 token ids", leave=False):
            anno_data = self.anno_objs_cache.get(vid)
            if anno_data:
                video_objs = anno_data.get("video_objs", [])
                if video_objs:
                    video_objs_ids = self._convert_objs_to_ids_fast(video_objs)
                    self.video_objs_ids_cache[vid] = video_objs_ids
                    tokenized_count += 1
                else:
                    self.video_objs_ids_cache[vid] = []
        
        tokenize_time = time.time() - tokenize_start
        total_time = time.time() - load_start
        
        print(f"  ✅ 预加载完成")
        print(f"  加载成功: {loaded_count}/{len(self.video_ids)}")
        print(f"  文件缺失: {missing_count}/{len(self.video_ids)}")
        if error_count > 0:
            print(f"  加载错误: {error_count}/{len(self.video_ids)}")
        print(f"  预计算 token ids: {tokenized_count}/{len(self.video_ids)}")
        print(f"  总耗时: {total_time:.2f}s (加载: {load_time:.2f}s, tokenize: {tokenize_time:.2f}s)")
        print(f"  平均耗时: {total_time/len(self.video_ids)*1000:.2f}ms/视频")
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.data_list)
    
    def _load_anno_obj(self, vid):
        """
        从预加载的缓存中获取 anno_objs 数据
        
        Args:
            vid: 视频ID，如 'video0'
        
        Returns:
            anno_data: 包含 video_objs, sentence_objs 等字段的字典
        """
        # 直接从预加载的缓存中获取
        if vid in self.anno_objs_cache:
            return self.anno_objs_cache[vid]
        
        # 如果缓存中没有（理论上不应该发生，因为已经预加载），返回空数据
        # 这种情况不应该发生，但为了安全起见保留此逻辑
        empty_data = {
            "video_objs": [],
            "sentence_objs": {},
            "video_objs_with_adj": [],
            "video_objs_idx": []
        }
        return copy.deepcopy(empty_data)
    
    def _convert_objs_to_ids_fast(self, obj_list):
        """
        将对象词列表转换为对应的token ids（基于BPE单词边界）- 快速版本
        
        使用预构建的 id_to_token 映射，避免每次decode
        
        Args:
            obj_list: 对象词列表，如 ["audi", "car", "berrymore", "road"]
        
        Returns:
            obj_ids_list: List[List[int]] 每个对象的完整token序列
        """
        if not obj_list:
            return []
        
        obj_ids_list = []
        
        CLS_ID = 49406
        SEP_ID = 49407
        PAD_ID = 0
        
        for obj in obj_list:
            # 编码单个对象词
            encoded = self.tokenizer(
                obj,
                add_special_tokens=True,  # 会添加CLS和SEP
                return_tensors='pt'
            )
            token_ids = encoded['input_ids'][0]  # [seq_len]
            
            # 提取属于当前单词的所有tokens（基于</w>边界）
            word_tokens = []
            
            for tid in token_ids:
                tid_val = tid.item()
                
                # 跳过特殊token
                if tid_val in [CLS_ID, SEP_ID, PAD_ID]:
                    continue
            
                # 添加当前token
                word_tokens.append(tid_val)
                
                # 快速检查：使用预构建的 id_to_token 映射（避免decode）
                token_str = self.id_to_token.get(tid_val, '')
                if token_str.endswith('</w>'):
                    # 单词结束，停止收集
                    break
                # 如果没有</w>，继续收集下一个token
            
            if word_tokens:
                obj_ids_list.append(word_tokens)
            else:
                # 如果没有有效token（理论上不应该），使用[0]
                obj_ids_list.append([0])
        
        return obj_ids_list
    
    def _convert_objs_to_ids(self, obj_list):
        """
        将对象词列表转换为对应的token ids（基于BPE单词边界）
        
        Args:
            obj_list: 对象词列表，如 ["audi", "car", "berrymore", "road"]
        
        Returns:
            obj_ids_list: List[List[int]] 每个对象的完整token序列
                         基于BPE的</w>标记判断单词边界
                         如 [
                             [2214],           # 'audi</w>' (单token)
                             [235],            # 'car</w>' (单token)
                             [15334, 3172],    # 'berry' + 'more</w>' (多token)
                             [231]             # 'road</w>' (单token)
                         ]
        
        工作原理：
            - 检查每个token解码后是否有</w>后缀
            - 如果没有</w>：说明单词未结束，继续收集下一个token
            - 如果有</w>：说明单词结束，停止收集
            - 这样可以正确处理BPE分词的多token单词
        """
        if not obj_list:
            return []
        
        obj_ids_list = []
        
        CLS_ID = 49406
        SEP_ID = 49407
        PAD_ID = 0
        
        for obj in obj_list:
            # 编码单个对象词
            encoded = self.tokenizer(
                obj,
                add_special_tokens=True,  # 会添加CLS和SEP
                return_tensors='pt'
            )
            token_ids = encoded['input_ids'][0]  # [seq_len]
            
            # 提取属于当前单词的所有tokens（基于</w>边界）
            word_tokens = []
            
            for tid in token_ids:
                tid_val = tid.item()
                
                # 跳过特殊token
                if tid_val in [CLS_ID, SEP_ID, PAD_ID]:
                    continue
            
                # 添加当前token
                word_tokens.append(tid_val)
                
                # 检查是否有</w>后缀（单词结束标记）
                decoded_token = self.tokenizer.decode([tid_val])
                if decoded_token.endswith('</w>'):
                    # 单词结束，停止收集
                    break
                # 如果没有</w>，继续收集下一个token
            
            if word_tokens:
                obj_ids_list.append(word_tokens)
            else:
                # 如果没有有效token（理论上不应该），使用[0]
                obj_ids_list.append([0])
        
        return obj_ids_list
    
    def _select_and_pad_objs(self, video_objs, video_objs_with_adj, video_objs_ids=None, qmax=64):
        """
        选择并padding对象到固定数量Qmax
        同时为每个对象随机选择一个短语
        
        Args:
            video_objs: 视频对象词列表，如 ['car', 'man', ...]
            video_objs_with_adj: 对应的词组列表，如 [['a car', 'the car'], ['a man'], ...]
            video_objs_ids: 预计算的对象token ids（可选，用于优化性能）
            qmax: 最大对象数量，默认64
        
        Returns:
            video_objs_selected: List[str] 长度为qmax
                                如果N>64：随机选择64个
                                如果N<64：原样+用"[NO_OBJ]"补足到64
            video_objs_ids_selected: List[List[int]] 长度为qmax
                                     每个obj的token ids（BPE完整）
            video_phrases_selected: List[str] 长度为qmax
                                   每个obj对应的随机选择的词组
            video_phrases_ids_selected: Tensor[qmax, 77]
                                       词组的tokenized结果
            video_objs_ids_selected_mask: Tensor[qmax]
                                         标注真实对象的位置为1，[NO_OBJ]位置为0
            video_phrases_ids_selected_mask: Tensor[qmax, 77]
                                            短语的attention mask（从CLS到SEP）
        """
        import random
        rng = random.Random(42)  # 固定seed
        
        num_objs = len(video_objs)
        
        # 1. 选择对象（如果超过qmax则随机采样）
        if num_objs > qmax:
            # 随机选择qmax个（保持顺序）
            selected_indices = sorted(rng.sample(range(num_objs), qmax))
            video_objs_selected = [video_objs[i] for i in selected_indices]
            video_objs_with_adj_selected = [video_objs_with_adj[i] if i < len(video_objs_with_adj) else [] 
                                           for i in selected_indices]
            # 从预计算的 video_objs_ids 中选择对应的token ids（优化性能）
            if video_objs_ids:
                video_objs_ids_selected = [video_objs_ids[i] for i in selected_indices]
            else:
                video_objs_ids_selected = self._convert_objs_to_ids_fast(video_objs_selected)
        elif num_objs < qmax:
            # 不足qmax，用"[NO_OBJ]"补足
            video_objs_selected = video_objs + ["[NO_OBJ]"] * (qmax - num_objs)
            video_objs_with_adj_selected = video_objs_with_adj + [["[NO_OBJ]"]] * (qmax - num_objs)
            # 使用预计算的 video_objs_ids 并补足 [NO_OBJ] 的token ids
            if video_objs_ids:
                # [NO_OBJ] 的token id是 49412
                video_objs_ids_selected = video_objs_ids + [[49412]] * (qmax - num_objs)
            else:
                video_objs_ids_selected = self._convert_objs_to_ids_fast(video_objs_selected)
        else:
            # 正好qmax个
            video_objs_selected = video_objs
            video_objs_with_adj_selected = video_objs_with_adj
            # 直接使用预计算的 video_objs_ids（优化性能）
            if video_objs_ids:
                video_objs_ids_selected = video_objs_ids
            else:
                video_objs_ids_selected = self._convert_objs_to_ids_fast(video_objs_selected)
        
        # 2. 为每个对象随机选择一个词组
        video_phrases_selected = []
        for obj, adj_phrases in zip(video_objs_selected, video_objs_with_adj_selected):
            if not adj_phrases or obj == "[NO_OBJ]":
                # 没有词组或是[NO_OBJ]，直接使用对象词本身
                video_phrases_selected.append(obj)
            else:
                # 随机选择一个词组
                selected_phrase = rng.choice(adj_phrases)
                video_phrases_selected.append(selected_phrase)
        
        # 3. video_objs_ids_selected 已经在上面的逻辑中处理好了，不需要再计算
        
        # 4. 生成video_phrases_ids_selected（tokenize词组）
        encoded = self.tokenizer(
            video_phrases_selected,
                padding='max_length',
                max_length=77,
                truncation=True,
                return_tensors='pt'
            )
        video_phrases_ids_selected = encoded['input_ids']  # [qmax, 77]
        video_phrases_ids_selected_mask = encoded['attention_mask']  # [qmax, 77]
        
        # 5. 生成video_objs_ids_selected_mask（标注真实对象位置）
        video_objs_ids_selected_mask = torch.zeros(qmax, dtype=torch.long)
        for i, obj in enumerate(video_objs_selected):
            if obj != "[NO_OBJ]":
                video_objs_ids_selected_mask[i] = 1
        
        return (video_objs_selected, video_objs_ids_selected, 
                video_phrases_selected, video_phrases_ids_selected,
                video_objs_ids_selected_mask, video_phrases_ids_selected_mask)
    
    def __getitem__(self, idx):
        """
        获取一个样本
        
        Returns (23个):
            [0] vid_feat_tensor: 视频特征 [T, D] (Tensor)
            [1] vid_mask_tensor: 视频mask [T] (Tensor)
            
            Caption i（当前句子）：
            [2] caption_i_ids: caption_i token ids [77] (Tensor)
            [3] caption_i_mask: caption_i mask [77] (Tensor)
            [4] caption_i_label: caption_i 文本 (str)
            [5] sent_i_id: 句子i的ID (str)
            
            Caption j（对比句子）：
            [6] caption_j_ids: caption_j token ids [77] (Tensor)
            [7] caption_j_mask: caption_j mask [77] (Tensor)
            [8] caption_j_label: caption_j 文本 (str)
            [9] sent_j_id: 句子j的ID (str)
            
            质量评分：
            [10] capscore_i: caption_i的CIDEr评分 (float)
            [11] capscore_j: caption_j的CIDEr评分 (float)
            
            [12] vid: 视频ID (str)
            
            对象标注（原始）：
            [13] video_objs: 视频对象词列表 (List[str])
            [14] video_objs_idx: 多标签向量 [49413] (Tensor, float32)
            [15] video_objs_ids: 对象token ids (List[List[int]])
            [16] video_objs_with_adj: 带修饰的短语 (List[List[str]])
            
            对象标注（选择后，固定Qmax=64）：
            [17] video_objs_selected: 对象词 (List[str]) [64]
            [18] video_objs_ids_selected: 对象token ids (List[List[int]]) [64]
            [19] video_phrases_selected: 词组 (List[str]) [64]
            [20] video_phrases_ids_selected: 词组token ids (Tensor) [64, 77]
            [21] video_objs_ids_selected_mask: 真实对象mask (Tensor) [64]
            [22] video_phrases_ids_selected_mask: 词组attention mask (Tensor) [64, 77]
        
        对比学习说明：
            - 如果use_contrastive=True且有anno_cider：
              caption_j从同video的其他句子中随机选择
              capscore_i, capscore_j为对应的CIDEr评分
            
            - 如果use_contrastive=False或没有anno_cider：
              caption_j = caption_i（相同）
              capscore_i = capscore_j = 0.0
        
        使用示例：
        ```python
        # 获取样本（23个返回值）
        (vid_feat, vid_mask,
         caption_i_ids, caption_i_mask, caption_i_label, sent_i_id,
         caption_j_ids, caption_j_mask, caption_j_label, sent_j_id,
         capscore_i, capscore_j, vid,
         video_objs, video_objs_idx, video_objs_ids, video_objs_with_adj,
         video_objs_selected, video_objs_ids_selected,
         video_phrases_selected, video_phrases_ids_selected,
         video_objs_ids_selected_mask, video_phrases_ids_selected_mask) = dataset[0]
        
        # 对比学习
        if capscore_i > capscore_j:
            # caption_i质量更高
            pass
        ```
        """
        # 原始数据（继承自父类）
        vid, caption_i, sent_i_id = self.data_list[idx]
        
        # 获取视频帧特征和 mask（lazy加载，每次使用时转换）
        feat, vid_mask = self.features[vid]
        vid_feat_tensor = torch.tensor(feat, dtype=torch.float32)
        vid_mask_tensor = torch.tensor(vid_mask, dtype=torch.long)
        
        # 对 caption_i 文本进行分词编码（使用扩展后的 tokenizer）
        encoded_i = self.tokenizer(
            caption_i,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        caption_i_ids = encoded_i['input_ids'].squeeze(0)
        caption_i_mask = encoded_i['attention_mask'].squeeze(0)
        caption_i_label = caption_i
        
        # === 对比学习：选择caption_j（同video的另一个句子）===
        caption_j_ids = None
        caption_j_mask = None
        caption_j_label = None
        sent_j_id = None
        capscore_i = 0.0
        capscore_j = 0.0
        
        if self.use_contrastive and self.anno_cider is not None:
            # 获取同video的所有句子
            video_sentences = self.captions_data.get(vid, [])
            
            if len(video_sentences) > 1:
                # 随机选择一个不同的句子作为caption_j
                import random
                other_sentences = [(cap, sid) for cap, sid in video_sentences 
                                  if str(sid) != str(sent_i_id)]
                
                if other_sentences:
                    caption_j, sent_j_id = random.choice(other_sentences)
                    sent_j_id = str(sent_j_id)
                    
                    # Tokenize caption_j
                    encoded_j = self.tokenizer(
                        caption_j,
                        max_length=self.max_len,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                    caption_j_ids = encoded_j['input_ids'].squeeze(0)
                    caption_j_mask = encoded_j['attention_mask'].squeeze(0)
                    caption_j_label = caption_j
                    
                    # 获取评分
                    video_scores = self.anno_cider.get(vid, {})
                    capscore_i = video_scores.get(str(sent_i_id), 0.0)
                    capscore_j = video_scores.get(sent_j_id, 0.0)
        
        # 如果没有选择caption_j，使用相同的caption_i（placeholder）
        if caption_j_ids is None:
            caption_j_ids = caption_i_ids.clone()
            caption_j_mask = caption_i_mask.clone()
            caption_j_label = caption_i_label
            sent_j_id = sent_i_id
        
        # === 新增：从预加载的缓存中获取 anno_objs 数据 ===
        anno_data = self._load_anno_obj(vid)
        
        # 1. video_objs: 视频级别的对象词列表
        video_objs = anno_data.get("video_objs", [])
        
        # 2. video_objs_idx: 多标签向量（转换为 tensor）
        video_objs_idx_list = anno_data.get("video_objs_idx", [])
        if video_objs_idx_list:
            video_objs_idx = torch.tensor(video_objs_idx_list, dtype=torch.float32)
        else:
            # 如果没有，创建全零向量（词表大小需要动态获取，这里先用0）
            video_objs_idx = torch.tensor([], dtype=torch.float32)
        
        # 3. video_objs_with_adj: 带修饰的对象短语列表
        video_objs_with_adj = anno_data.get("video_objs_with_adj", [])
        
        # 4. 从缓存获取预计算的 video_objs_ids（优化性能，避免重复tokenize）
        video_objs_ids = self.video_objs_ids_cache.get(vid, None)
        # 如果缓存中没有（理论上不应该发生），则实时计算
        if video_objs_ids is None:
            video_objs_ids = self._convert_objs_to_ids(video_objs)
            self.video_objs_ids_cache[vid] = video_objs_ids
        
        # 5. 选择并padding对象到Qmax=64（传递预计算的video_objs_ids以优化性能）
        (video_objs_selected, video_objs_ids_selected,
         video_phrases_selected, video_phrases_ids_selected,
         video_objs_ids_selected_mask, video_phrases_ids_selected_mask) = self._select_and_pad_objs(
            video_objs, video_objs_with_adj, video_objs_ids=video_objs_ids, qmax=64
        )
        
        # 6. 加载预提取特征（如果启用）
        phrase_embs_precomputed = None
        phrase_mask_precomputed = None
        sentence_embs_i_precomputed = None
        sentence_mask_i_precomputed = None
        sentence_embs_j_precomputed = None
        sentence_mask_j_precomputed = None
        
        if self.use_precomputed:
            # 加载 phrase 特征（按视频存储，不含 sent_id）
            phrase_feat_file = self.precomputed_phrase_dir / f"{vid}.npy"
            if phrase_feat_file.exists():
                try:
                    feat_data = np.load(phrase_feat_file, allow_pickle=True).item()
                    phrase_embs_precomputed = torch.from_numpy(feat_data['phrase_embs']).float()  # [Qmax, 77, 512]
                    phrase_mask_precomputed = torch.from_numpy(feat_data['phrase_word_mask']).long()  # [Qmax, 77]
                    self._phrase_loaded_count += 1
                except Exception as e:
                    # 如果加载失败，打印警告但继续（使用在线编码）
                    self._phrase_missing_count += 1
                    if not hasattr(self, '_phrase_load_warned'):
                        print(f"\n⚠️ 警告：加载 phrase 特征失败: {phrase_feat_file}")
                        print(f"  错误: {e}")
                        print(f"  将使用在线编码（慢）")
                        self._phrase_load_warned = True
            else:
                # 文件不存在，记录一次警告
                self._phrase_missing_count += 1
                if self._phrase_missing_count <= 3:  # 只打印前3个缺失的文件
                    print(f"\n⚠️ 警告：预提取 phrase 特征文件不存在: {phrase_feat_file}")
                    print(f"  将使用在线编码（慢）")
                    if self._phrase_missing_count == 3:
                        print(f"  ...（后续缺失文件不再打印）")
                        print(f"  请检查路径是否正确，或运行 extract_msrvtt_CLIPPhraseEncoder_feats.py 提取特征")
            
            # 加载 sentence i 特征
            if self.precomputed_sentence_dir:
                sent_i_file = self.precomputed_sentence_dir / f"{sent_i_id}.npy"
                if sent_i_file.exists():
                    try:
                        feat_data = np.load(sent_i_file, allow_pickle=True).item()
                        sentence_embs_i_precomputed = torch.from_numpy(feat_data['sentence_embs']).float()  # [77, 512]
                        sentence_mask_i_precomputed = torch.from_numpy(feat_data['sentence_word_mask']).long()  # [77]
                        self._sentence_loaded_count += 1
                    except Exception as e:
                        self._sentence_missing_count += 1
                        if not hasattr(self, '_sentence_i_load_warned'):
                            print(f"\n⚠️ 警告：加载 sentence_i 特征失败: {sent_i_file}")
                            print(f"  错误: {e}")
                            self._sentence_i_load_warned = True
                else:
                    self._sentence_missing_count += 1
                    if self._sentence_missing_count <= 3:
                        print(f"\n⚠️ 警告：预提取 sentence_i 特征文件不存在: {sent_i_file}")
                        print(f"  将使用在线编码（慢）")
                        if self._sentence_missing_count == 3:
                            print(f"  ...（后续缺失文件不再打印）")
                
                # 加载 sentence j 特征（如果有）
                if caption_j_ids is not None and sent_j_id is not None:
                    sent_j_file = self.precomputed_sentence_dir / f"{sent_j_id}.npy"
                    if sent_j_file.exists():
                        try:
                            feat_data = np.load(sent_j_file, allow_pickle=True).item()
                            sentence_embs_j_precomputed = torch.from_numpy(feat_data['sentence_embs']).float()  # [77, 512]
                            sentence_mask_j_precomputed = torch.from_numpy(feat_data['sentence_word_mask']).long()  # [77]
                        except Exception as e:
                            if not hasattr(self, '_sentence_j_load_warned'):
                                print(f"\n⚠️ 警告：加载 sentence_j 特征失败: {sent_j_file}")
                                print(f"  错误: {e}")
                                self._sentence_j_load_warned = True
        
        # 返回原始数据 + 新增的对象标注数据 + 对比学习数据 + 预提取特征
        return (
            vid_feat_tensor,              # [0] [T, D] 视频特征
            vid_mask_tensor,              # [1] [T] 视频mask
            # === Caption i（当前句子） ===
            caption_i_ids,                # [2] [77] caption_i token ids
            caption_i_mask,               # [3] [77] caption_i mask
            caption_i_label,              # [4] str caption_i文本
            sent_i_id,                    # [5] str 句子i的ID
            # === Caption j（对比句子） ===
            caption_j_ids,                # [6] [77] caption_j token ids
            caption_j_mask,               # [7] [77] caption_j mask
            caption_j_label,              # [8] str caption_j文本
            sent_j_id,                    # [9] str 句子j的ID
            # === 质量评分 ===
            capscore_i,                   # [10] float caption_i的质量分数
            capscore_j,                   # [11] float caption_j的质量分数
            # === 视频ID ===
            vid,                          # [12] str 视频ID
            # === 对象标注（原始） ===
            video_objs,                   # [13] List[str] 视频对象词（原始，长度N）
            video_objs_idx,               # [14] Tensor[vocab_size] 多标签向量
            video_objs_ids,               # [15] List[List[int]] 对象token ids（原始，长度N）
            video_objs_with_adj,          # [16] List[List[str]] 带修饰的短语（原始，长度N）
            # === 对象标注（选择后，固定Qmax=64） ===
            video_objs_selected,          # [17] List[str] [Qmax=64]
            video_objs_ids_selected,      # [18] List[List[int]] [Qmax=64]
            video_phrases_selected,       # [19] List[str] [Qmax=64]
            video_phrases_ids_selected,   # [20] Tensor[Qmax=64, 77]
            video_objs_ids_selected_mask, # [21] Tensor[Qmax=64] 真实对象mask
            video_phrases_ids_selected_mask, # [22] Tensor[Qmax=64, 77] 词组attention mask
            # === 预提取特征（可选，用于加速训练） ===
            phrase_embs_precomputed,      # [23] Tensor[Qmax, 77, D] or None
            phrase_mask_precomputed,      # [24] Tensor[Qmax, 77] or None
            sentence_embs_i_precomputed,  # [25] Tensor[77, D] or None
            sentence_mask_i_precomputed,  # [26] Tensor[77] or None
            sentence_embs_j_precomputed,  # [27] Tensor[77, D] or None
            sentence_mask_j_precomputed   # [28] Tensor[77] or None
        )


# ============================================================================
# 测试函数：MSRVTT_FeaturesDataset_AddClass
# ============================================================================
def test_addclass_dataset():
    """测试新增的数据集类"""
    print("=" * 80)
    print("测试 MSRVTT_FeaturesDataset_AddClass")
    print("=" * 80)
    
    # 配置路径（从 dataloaders/ 子目录运行）
    features_path = '../../datasets/MSRVTT/feats/ViT-B-32_k_split_ks12_features.pickle'
    json_path = '../../datasets/MSRVTT/MSRVTT_data.json'
    anno_objs_dir = '../../datasets/MSRVTT/WordsExtraction/anno_objs'
    updated_tokenizer_dir = '../models/clip_tokenizer/models--openai--clip-vit-base-patch32/updated_with_obj_tokens'
    anno_cider_path = '../../datasets/MSRVTT/anno_msrvtt_cider.json'  # CIDEr评分
    
    # 验证路径
    print(f"检查文件路径：")
    print(f"  features_path: {'✅' if os.path.exists(features_path) else '❌'} {features_path}")
    print(f"  json_path: {'✅' if os.path.exists(json_path) else '❌'} {json_path}")
    print(f"  anno_objs_dir: {'✅' if os.path.exists(anno_objs_dir) else '❌'} {anno_objs_dir}")
    print(f"  updated_tokenizer_dir: {'✅' if os.path.exists(updated_tokenizer_dir) else '❌'} {updated_tokenizer_dir}")
    print()
    
    # 创建数据集
    print("\n[创建数据集]")
    use_cider = os.path.exists(anno_cider_path)
    
    dataset = MSRVTT_FeaturesDataset_AddClass(
        features_path=features_path,
        json_path=json_path,
        anno_objs_dir=anno_objs_dir,
        updated_tokenizer_dir=updated_tokenizer_dir,
        anno_cider_path=anno_cider_path if use_cider else None,
        split='train',
        tkz_type='clip',
        use_contrastive=use_cider  # 如果有CIDEr评分则启用对比学习
    )
    
    print(f"\n数据集大小: {len(dataset)}")
    
    # 找到 video55 对应的样本索引
    print("\n[查找 video55 的样本]")
    video55_idx = None
    for i, (v, c, s) in enumerate(dataset.data_list):
        if v == 'video55':
            video55_idx = i
            print(f"  找到 video55 的第一个样本: 索引 {i}")
            print(f"  caption: {c}")
            print(f"  sen_id: {s}")
            break
    
    if video55_idx is None:
        print(f"  ❌ 未找到 video55（可能不在当前split）")
        print(f"  使用 video0 作为测试样本")
        video55_idx = 0
    
    # 测试 video55 样本
    print(f"\n[测试样本 {video55_idx}]")
    sample = dataset[video55_idx]
    
    print(f"返回值数量: {len(sample)}")
    
    # 解包（23个返回值）
    (vid_feat, vid_mask,
     caption_i_ids, caption_i_mask, caption_i_label, sent_i_id,
     caption_j_ids, caption_j_mask, caption_j_label, sent_j_id,
     capscore_i, capscore_j, vid,
     video_objs, video_objs_idx, video_objs_ids, video_objs_with_adj,
     video_objs_selected, video_objs_ids_selected,
     video_phrases_selected, video_phrases_ids_selected,
     video_objs_ids_selected_mask, video_phrases_ids_selected_mask) = sample
    
    print(f"\n原始返回值：")
    print(f"  vid_feat shape: {vid_feat.shape}")
    print(f"  vid_mask shape: {vid_mask.shape}")
    print(f"  video_id: {vid}")
    
    print(f"\nCaption i（当前句子）：")
    print(f"  caption_i_ids shape: {caption_i_ids.shape}")
    print(f"  caption_i_mask shape: {caption_i_mask.shape}")
    print(f"  caption_i_label: {caption_i_label}")
    print(f"  sent_i_id: {sent_i_id}")
    print(f"  capscore_i: {capscore_i:.4f}")
    
    print(f"\nCaption j（对比句子）：")
    print(f"  caption_j_ids shape: {caption_j_ids.shape}")
    print(f"  caption_j_mask shape: {caption_j_mask.shape}")
    print(f"  caption_j_label: {caption_j_label}")
    print(f"  sent_j_id: {sent_j_id}")
    print(f"  capscore_j: {capscore_j:.4f}")
    print(f"  是否为同一句子: {sent_i_id == sent_j_id}")
    
    print(f"\n新增返回值（对象标注）：")
    print(f"  video_objs: {video_objs[:5]}...")
    print(f"  video_objs_idx shape: {video_objs_idx.shape}")
    print(f"  video_objs_idx dtype: {video_objs_idx.dtype}")
    print(f"  video_objs_idx 中为1的数量: {int(video_objs_idx.sum())}")
    print(f"  video_objs_ids类型: List[List[int]]，长度={len(video_objs_ids)}")
    print(f"  video_objs_with_adj 数量: {len(video_objs_with_adj)}")
    if video_objs_with_adj:
        print(f"  video_objs_with_adj[0]: {video_objs_with_adj[0]}")
    
    # 验证video_objs和video_objs_ids的对应关系（基于BPE边界）
    print(f"\n[验证] video_objs ↔ video_objs_ids 对应关系（BPE单词边界）：")
    for i, (obj, obj_token_ids) in enumerate(zip(video_objs[:5], video_objs_ids[:5])):
        # 使用convert_ids_to_tokens直接获取token字符串（包含</w>）
        token_strings = dataset.tokenizer.convert_ids_to_tokens(obj_token_ids)
        decoded_full = dataset.tokenizer.decode(obj_token_ids, skip_special_tokens=True)
        
        if len(obj_token_ids) == 1:
            print(f"  [{i}] '{obj}' -> token_id={obj_token_ids[0]} -> token='{token_strings[0]}' -> decoded='{decoded_full}'")
        else:
            print(f"  [{i}] '{obj}' -> token_ids={obj_token_ids}")
            print(f"      各token: {token_strings} -> 合并='{decoded_full}'")
            # 验证</w>边界
            has_end_marker = [('✓' if '</w>' in t else '✗') for t in token_strings]
            print(f"      </w>标记: {has_end_marker} (✓=有边界)")
    
    # 验证对应关系
    print(f"\n[验证] video_objs 和 video_objs_with_adj 对应关系：")
    for i, (obj, adj_list) in enumerate(zip(video_objs[:3], video_objs_with_adj[:3])):
        print(f"  [{i}] '{obj}' -> {adj_list}")
    
    # 新增：验证选择后的对象（固定64个）
    print(f"\n新增返回值（选择后的对象，固定Qmax=64）：")
    print(f"  video_objs原始数量: {len(video_objs)}")
    print(f"  video_objs_selected长度: {len(video_objs_selected)}")
    print(f"  video_objs_ids_selected长度: {len(video_objs_ids_selected)}")
    print(f"  video_phrases_selected长度: {len(video_phrases_selected)}")
    print(f"  video_phrases_ids_selected shape: {video_phrases_ids_selected.shape}")
    
    # 显示前5个
    print(f"\n前5个对象及其选择的词组：")
    for i in range(min(5, len(video_objs_selected))):
        obj = video_objs_selected[i]
        phrase = video_phrases_selected[i]
        obj_token_ids = video_objs_ids_selected[i]
        phrase_ids = video_phrases_ids_selected[i]
        
        print(f"  [{i}] 对象: '{obj}'")
        print(f"      对象token_ids: {obj_token_ids}")
        print(f"      选择的词组: '{phrase}'")
        print(f"      词组ids[:10]: {phrase_ids[:10].tolist()}")
        print()
    
    # 显示最后5个（应该是[NO_OBJ]如果N<64）
    print(f"最后5个对象（如果原始N<64应该是[NO_OBJ]）：")
    for i in range(59, 64):
        obj = video_objs_selected[i]
        phrase = video_phrases_selected[i]
        obj_mask = video_objs_ids_selected_mask[i].item()
        print(f"  [{i}] 对象: '{obj}' -> 词组: '{phrase}' -> mask: {obj_mask}")
    
    # 新增：验证两个 mask
    print(f"\n新增的 mask 验证：")
    print(f"  video_objs_ids_selected_mask shape: {video_objs_ids_selected_mask.shape}")
    print(f"  video_objs_ids_selected_mask dtype: {video_objs_ids_selected_mask.dtype}")
    print(f"  video_objs_ids_selected_mask 中为1的数量: {video_objs_ids_selected_mask.sum().item()}")
    print(f"  预期：应该等于真实对象数量（不包括[NO_OBJ]）")
    
    print(f"\n  video_phrases_ids_selected_mask shape: {video_phrases_ids_selected_mask.shape}")
    print(f"  video_phrases_ids_selected_mask dtype: {video_phrases_ids_selected_mask.dtype}")
    
    # 显示前3个词组的mask（查看CLS到SEP的范围）
    print(f"\n  前3个词组的attention mask：")
    for i in range(min(3, len(video_phrases_selected))):
        phrase = video_phrases_selected[i]
        mask = video_phrases_ids_selected_mask[i]
        mask_sum = mask.sum().item()
        # 找到第一个0的位置（SEP之后）
        nonzero_indices = torch.nonzero(mask).squeeze()
        if nonzero_indices.numel() > 0:
            last_valid = nonzero_indices[-1].item() if nonzero_indices.dim() > 0 else nonzero_indices.item()
            print(f"    [{i}] '{phrase}'")
            print(f"        mask有效长度: {mask_sum} tokens (从CLS到SEP)")
            print(f"        最后一个有效位置: {last_valid}")
            print(f"        mask前10个: {mask[:10].tolist()}")
        else:
            print(f"    [{i}] '{phrase}' -> mask全0（异常）")
    
    print("\n" + "=" * 80)
    print("✅ 测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    # 测试新增的数据集类
    test_addclass_dataset()