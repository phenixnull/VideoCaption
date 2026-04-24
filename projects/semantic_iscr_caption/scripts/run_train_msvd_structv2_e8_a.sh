#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" "$ROOT/train_structured_refine_monitored.py" \
  --dataset_type msvd \
  --model_type structured_local_best \
  --clip_global_vision_feats_path "$ROOT/../../datasets/MSVD/feats/clip4clip_vitb32_k_split_ks12_features.pickle" \
  --annotations_path "$ROOT/../../datasets/MSVD/annotations_preprocessed.txt" \
  --structured_gt_path "$ROOT/annotations/msvd_structured_train_api.json" \
  --init_caption_ckpt "${INIT_CAPTION_CKPT:-/mnt/sda/Disk_D/zhangwei/projects/VC/project/structured_caption/runs/local_noleak_api_e8_s6_tinylr/msvd_structured_api_e8_s6_tinylr_20260302_234617/checkpoints/epoch_007.pt}" \
  --out_dir "$ROOT/runs/local_noleak_api_structv2_e8_a" \
  --batch_size 48 \
  --epochs 8 \
  --scheduler cosine \
  --warmup_epochs 1 \
  --beam_size 5 \
  --beam_alpha 0.7 \
  --lambda_entity 0.05 \
  --lambda_action 0.05 \
  --lambda_phrase 0.0 \
  --struct_condition 1 \
  --struct_condition_scale 0.25 \
  --lr 2e-5 \
  --min_lr 1e-6

