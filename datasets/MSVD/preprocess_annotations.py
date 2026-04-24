# -*- coding: utf-8 -*-
"""
preprocess_annotations.py — 为 MSVD annotations.txt 添加句子ID

输入格式 (annotations.txt):
    video_id caption
    -4wsuPCjDBc_5_15 a squirrel is eating a peanut

输出格式 (annotations_preprocessed.txt):
    video_id sen_id caption
    -4wsuPCjDBc_5_15 0 a squirrel is eating a peanut
    -4wsuPCjDBc_5_15 1 a chipmunk is eating
    ...

每个视频的 sen_id 从 0 开始递增
"""

import os
from collections import defaultdict

def preprocess_annotations(input_path, output_path):
    """
    读取 annotations.txt，为每个视频的 caption 添加句子ID
    """
    # 1. 读取所有行并按 video_id 分组
    video_captions = defaultdict(list)
    
    print(f"[INFO] 读取: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 第一个空格分隔 video_id 和 caption
            parts = line.split(' ', 1)
            if len(parts) < 2:
                print(f"[WARN] 跳过格式错误的行: {line[:50]}...")
                continue
            video_id, caption = parts
            video_captions[video_id].append(caption)
    
    print(f"[INFO] 共 {len(video_captions)} 个视频")
    
    # 统计每个视频的 caption 数量
    caption_counts = [len(caps) for caps in video_captions.values()]
    print(f"[INFO] Caption 数量统计:")
    print(f"       总 caption 数: {sum(caption_counts)}")
    print(f"       最少: {min(caption_counts)}, 最多: {max(caption_counts)}, 平均: {sum(caption_counts)/len(caption_counts):.1f}")
    
    # 2. 写入新文件，添加 sen_id
    print(f"[INFO] 写入: {output_path}")
    total_lines = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        # 按 video_id 排序保持一致性
        for video_id in sorted(video_captions.keys()):
            captions = video_captions[video_id]
            for sen_id, caption in enumerate(captions):
                # 格式: video_id sen_id caption
                f.write(f"{video_id} {sen_id} {caption}\n")
                total_lines += 1
    
    print(f"[OK] 完成! 共写入 {total_lines} 行")
    return total_lines


if __name__ == "__main__":
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_file = os.path.join(script_dir, "annotations.txt")
    output_file = os.path.join(script_dir, "annotations_preprocessed.txt")
    
    preprocess_annotations(input_file, output_file)
