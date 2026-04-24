# -*- coding: utf-8 -*-
"""
dataset_msvd_raw.py — MSVD 数据集的原始视频加载器
用于提取 CLIP visual features
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import random
from rawvideo_util import RawVideoExtractor


class MSVD_Raw_Dataset(Dataset):
    """
    MSVD 数据集加载器，用于特征提取。
    
    MSVD 视频文件命名格式: {youtube_id}_{start}_{end}.avi
    例如: 00jrXRMlZOY_0_10.avi
    
    Params:
        videos_dir: YouTubeClips 视频文件夹路径
        max_frames: 最大抽取帧数，默认 12
        image_resolution: 图像分辨率，默认 224
        frame_order: 0=正常顺序; 1=逆序; 2=随机顺序
        slice_framepos: 0=从头采样; 1=从尾采样; 2=均匀采样
        transform_type: 预处理类型
        extract_type: 采样方法 ('fps' or 'k_split')
    """
    
    def __init__(
            self,
            videos_dir,
            max_frames=12,
            feature_framerate=1,
            image_resolution=224,
            frame_order=0,
            slice_framepos=2,  # 默认均匀采样
            transform_type=0,
            extract_type='k_split',
            extract_loc='mid'
    ):
        super(MSVD_Raw_Dataset, self).__init__()
        self.videos_dir = videos_dir
        self.max_frames = max_frames
        self.feature_framerate = feature_framerate
        self.transform_type = transform_type
        self.sample_type = extract_type
        self.extract_loc = extract_loc
        
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]
        
        # 初始化视频提取器
        self.rawVideoExtractor = RawVideoExtractor(
            framerate=feature_framerate,
            size=image_resolution,
            transform_type=self.transform_type,
            k_segments=self.max_frames,
            sample_type=self.sample_type,
            extract_loc=self.extract_loc
        )
        
        # 扫描视频目录，获取所有视频文件
        self.video_list = self._scan_videos()
        print(f"[MSVD_Raw_Dataset] Found {len(self.video_list)} videos in {videos_dir}")
    
    def _scan_videos(self):
        """扫描视频目录，返回 (video_id, video_path) 列表"""
        video_list = []
        supported_ext = ('.avi', '.mp4', '.webm', '.mkv')
        
        for fname in sorted(os.listdir(self.videos_dir)):
            if fname.lower().endswith(supported_ext):
                # video_id 去掉扩展名
                video_id = os.path.splitext(fname)[0]
                video_path = os.path.join(self.videos_dir, fname)
                video_list.append((video_id, video_path))
        
        return video_list
    
    def __len__(self):
        return len(self.video_list)
    
    def _get_rawvideo(self, video_id, video_path):
        """提取单个视频的帧"""
        video_mask = np.zeros((1, self.max_frames), dtype=np.int64)
        max_video_length = 0
        
        if not os.path.exists(video_path):
            print(f"[Warning] Video not found: {video_path}")
            # 返回空的 video 和 mask
            video = np.zeros((1, self.max_frames, 1, 3, 224, 224), dtype=np.float32)
            return video, video_mask
        
        # 获取原始视频数据
        raw_video_data, shapes = self.rawVideoExtractor.get_video_data(video_path)
        raw_video_data = raw_video_data['video']
        
        # 初始化 video tensor
        video = np.zeros((1, self.max_frames, 1, 3, shapes[2], shapes[3]), dtype=np.float32)
        
        if len(raw_video_data.shape) > 3:
            raw_video_data_clip = raw_video_data
            # 处理原始数据 -> [frames, 1, 3, H, W]
            raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
            
            if self.max_frames < raw_video_slice.shape[0]:
                # 需要采样
                if self.slice_framepos == 0:  # 从头采样
                    video_slice = raw_video_slice[:self.max_frames, ...]
                elif self.slice_framepos == 1:  # 从尾采样
                    video_slice = raw_video_slice[-self.max_frames:, ...]
                else:  # 均匀采样
                    sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                    video_slice = raw_video_slice[sample_indx, ...]
            else:
                video_slice = raw_video_slice
            
            # 处理帧顺序
            video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)
            
            slice_len = video_slice.shape[0]
            max_video_length = slice_len
            
            if slice_len >= 1:
                video[0][:slice_len, ...] = video_slice
        else:
            print(f"[Warning] Video data error: {video_path}, video_id: {video_id}")
        
        # 设置 mask
        video_mask[0][:max_video_length] = 1
        
        return video, video_mask
    
    def __getitem__(self, idx):
        video_id, video_path = self.video_list[idx]
        video, video_mask = self._get_rawvideo(video_id, video_path)
        return video_id, video, video_mask
