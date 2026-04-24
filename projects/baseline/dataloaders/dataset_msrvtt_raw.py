from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import cv2
import random
from rawvideo_util import RawVideoExtractor
from PIL import Image

# Based on https://github.com/ArrowLuo/CLIP4Clip

# 这个主要是用来提取visual features的
class MSRVTT_Raw_Dataset(Dataset):
    """Implementation of the dataloader for MSRVTT. Mainly used in the feature extraction process.
        Params:
            csv_path: Path to the msrvtt.csv file.
                '''
                    train_list.csv
                    video0
                    video1
                    ...
                    video9999
                '''
            videos_path: Path to the video files.
            max_words: Max word length retained. Any more than the value will be truncated. Default: 73
            feature_framerate: sampling rate in second. Default: 1.0

            max_frames: Max frame sampled. Any more than the value will be ignored. Default: 20
            image_resolution: Processed image's width and height, in pixel. If param transform_type = 0 and
                the original image is greater than this value, it will be resized and center cropped. Default: 224
            frame_order: 0: ordinary order; 1: reverse order; 2: random order. Default: 0
            slice_framepos: 0: sample from the first frames; 1: sample from the last frames;
                2: sample uniformly. Default: 0
            transform_type:
                0: default transformation;
                1: transformation for objects, iou, temporal, action;
                2: transformation for i3d;. Default: 0
            extract_type: Type of frame extraction method ('fps' or 'k_split'). Default: 'fps'
            k_segments: Number of segments for 'k_split' method. Default: 1
            extract_loc: Frame extraction location within each segment ('start', 'mid', 'end') for 'k_split' method. Default: 'start'
    """
    def __init__(
            self,
            csv_path, # cav的路径，一般用于划分训练测试集这么
            videos_dir, # 视频文件夹package
            max_words=77, # 最大长度单词数
            feature_framerate=1, # 采样率，1s内的采样帧数 for fps extract type
            max_frames=20, # 最大抽取帧数，也就是说视频不足20s的话只会采样到十几帧
            image_resolution=224, # 图像分辨率
            frame_order=0, # 默认正常顺序
            slice_framepos=0, # for fps extract type
            transform_type =0,# 预处理类别

            extract_type='k_split', # 采样方法类型 k_split or fps
            k_segments=20, # 分段数
            extract_loc='start' # 抽帧位置
    ):
        super(MSRVTT_Raw_Dataset, self).__init__()
        self.data = pd.read_csv(csv_path) #读取csv
        self.videos_dir = videos_dir
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.transform_type = transform_type

        self.sample_type = extract_type
        self.k_segments = k_segments
        self.extract_loc = extract_loc
       
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        # uniformly应该就是从中间抽取
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        # 根据extract_type选择合适的RawVideoExtractor实例

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate,
                                                   size=image_resolution,
                                                   transform_type=self.transform_type,

                                                   k_segments=self.max_frames,
                                                   sample_type=self.sample_type,
                                                   extract_loc=self.extract_loc)


        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return len(self.data) #返回有多少个视频数据

    def _get_rawvideo(self, choice_video_ids):# 私有方法，用于类内
        # 这个choice_video_ids大概长这样['video1','video3','video4']
        # 比如说选了3个video
        # 那么video_mask就是有3个video的mask叠加在一起的，每个,video_mask里面有20个帧的位置
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.int64)
        # 最大视频长度？
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
#         video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
#                           self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)

        for i, video_id in enumerate(choice_video_ids):
            # Individual for YoucokII dataset, due to it video format
            # videos_path是所有视频raw文件夹，videos_path下放了10000个video
            # video_path就是单个视频的path
            video_path = os.path.join(self.videos_dir, "{}.mp4".format(video_id))
            if os.path.exists(video_path) is False:
                #当程序检测到指定路径的 .mp4 视频文件不存在时，自动尝试将文件后缀替换为 .webm，以寻找可能存在的另一种格式的视频文件。
                # 鲁棒性，如果mp4格式的文件没有就去寻找webm格式的文件
                video_path = video_path.replace(".mp4", ".webm")

            # 这个得到的raw_video_data默认是无重叠，不切块的
            # 那么它video_data的shape就是 [frames, h, w, c]
            # 又因为它是一个{'video',video_data_tensor}的字典
            raw_video_data, shapes = self.rawVideoExtractor.get_video_data(video_path)
            # 获取原始视频数据
            raw_video_data = raw_video_data['video']
                          # Pair x L x T x 3 x H x W
            # 	T为时序建模预留的维度（可能用于3D卷积或Transformer的位置编码）
            #   Pair为文本视频对的个数
            #   L为最大帧数
            video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                              shapes[2], shapes[3]), dtype=np.float32)
            if len(raw_video_data.shape) > 3:
                # [frames, h, w, c]
                raw_video_data_clip = raw_video_data
                # L x T x H x W x C
                # [frames, 1, h, w, c]
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                # print('raw_video_slice.shape', raw_video_slice.shape)
                # raw_video_slice.shapetorch.Size([12, 1, 3, 224, 224])
                # raw_video_slice.shapetorch.Size([22, 1, 3, 224, 224])

                if self.max_frames < raw_video_slice.shape[0]:
                    # 如果采样的最大帧数，小于原视频的帧数，要么前后截断采样，要么uniformly均匀采样
                    # 如果最大frames < raw_videp_slice.shape[0]也就是
                    if self.slice_framepos == 0: # 从头开始采样，
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1: # 从视频的尾部开始采样
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:# 整个视频上均匀采样
                        #[0,n-1]的位置采样max_frames
                        sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    video_slice = raw_video_slice
                # 然后就是制定采样顺序，0就是正常顺序，1就是逆序，2就是随机顺序
                #  [frames, 1, h, w, c]
                video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                slice_len = video_slice.shape[0] # 原采样的视频帧数
                # 第i个视频的最大长度，max_video_length[i]表示第i个视频的最大长度
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:# 视频帧数小于1？这是什么情况才会遇到的？
                    pass
                else:
                    # 第i个视频，切前slice_len个帧
                    # 由于第i个视频只有slice_len个帧，所以只切到这，又因为要装下所有video，所有第二个维度本身存储要保持最大的
                    video[i][:slice_len, ...] = video_slice
                    # 所以video最后的shape是
                    # [video_nums, max_frames, 1, h, w, c]
            else:# 最后的video shape一定是大于3的
                print("video path: {} error. video id: {}".format(video_path, video_id))
        # 处理玩video, 闲现在就要来处理每个video的video_mask
        # value = 1的位置表示又帧的位置
        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length
            # 1表示正常的位置，0表示没有帧或者被掩蔽的帧

        return video, video_mask

    def __getitem__(self, idx):
        # self.data是那个pand's的表,idx应该是多个int索引
        video_id = self.data['video_id'].values[idx]
        choice_video_ids = [video_id]

        video, video_mask = self._get_rawvideo(choice_video_ids)
        # video_mask的形状是[video_nums,max_frames]
        return video_id,video, video_mask #由于返回的矩阵是[video_nums, max_frames, 1, c, h, w]
                                            # 但是并不一定所有视频都是同样的帧数
                                        # 有的视频可能只有前面几帧有视频帧，所以对应的video_mask也只有有帧的位置才为1
                                        # video_id是'video0','video510'等这种形式
                                        # video是tensor