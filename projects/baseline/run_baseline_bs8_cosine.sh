#!/usr/bin/env bash
set -e

TS=$(date +%Y%m%d_%H%M%S)
LOG="/mnt/sda/Disk_D/zhangwei/projects/VC/project/baseline/nohup_logs/baseline_msvd_bs8_gpu0_${TS}.log"

GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
nohup env CUDA_VISIBLE_DEVICES="${GPU_ID}" /mnt/sda/Disk_D/zhangwei/anaconda3/envs/vcr/bin/python \
  /mnt/sda/Disk_D/zhangwei/projects/VC/project/baseline/train_base_mean_monitored.py \
  --dataset_type msvd \
  --clip_global_vision_feats_path /mnt/sda/Disk_D/zhangwei/projects/VC/datasets/MSVD/feats/clip4clip_vitb32_k_split_ks12_features.pickle \
  --annotations_path /mnt/sda/Disk_D/zhangwei/projects/VC/datasets/MSVD/annotations_preprocessed.txt \
  --out_dir /mnt/sda/Disk_D/zhangwei/projects/VC/project/baseline/runs/bs8 \
  --batch_size 8 \
  --epochs 20 \
  --scheduler cosine \
  --warmup_epochs 5 \
  --beam_size 5 \
  --beam_alpha 0.7 \
  --lr 5e-4 \
  --min_lr 1e-6 \
  > "${LOG}" 2>&1 &

echo "launched baseline bs8 cosine -> ${LOG}"
