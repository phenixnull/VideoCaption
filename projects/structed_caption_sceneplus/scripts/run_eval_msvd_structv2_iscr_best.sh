#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CKPT="${1:-$ROOT/weights/epoch_003.pt}"
OUT_JSON="${2:-$ROOT/runs/iscr_eval_best.json}"

mkdir -p "$(dirname "$OUT_JSON")"

"$PYTHON_BIN" "$ROOT/eval_structured_iscr_rerank.py" \
  --checkpoint "$CKPT" \
  --split test \
  --dataset_type msvd \
  --clip_global_vision_feats_path "$ROOT/../../datasets/MSVD/feats/clip4clip_vitb32_k_split_ks12_features.pickle" \
  --annotations_path "$ROOT/../../datasets/MSVD/annotations_preprocessed.txt" \
  --structured_gt_path "$ROOT/annotations/msvd_structured_train_api.json" \
  --batch_size 48 \
  --num_workers 8 \
  --beam_size 4 \
  --beam_alpha 0.7 \
  --iscr_rerank 1 \
  --iscr_rerank_alpha 1.0 \
  --iscr_rerank_lambda_cov 0.9 \
  --iscr_rerank_lambda_hall 0.15 \
  --iscr_prior_topk 64 \
  --iscr_rerank_topk 20 \
  --iscr_evidence_threshold 0.0 \
  --output_json "$OUT_JSON"

