#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:-C}"
GPU_ID="${2:-${CUDA_VISIBLE_DEVICES:-2}}"
TS_RAW="${3:-$(date +%Y%m%d_%H%M%S)}"
TS="$(printf '%s' "${TS_RAW}" | tr -d '\r\n')"
VARIANT_LOWER="$(echo "${VARIANT}" | tr '[:upper:]' '[:lower:]')"
PHRASE_DECODER_ENABLE_WAS_SET="${PHRASE_DECODER_ENABLE+x}"
PHRASE_CONDITION_ENABLE_WAS_SET="${PHRASE_CONDITION_ENABLE+x}"
LAMBDA_PHRASE_WAS_SET="${LAMBDA_PHRASE+x}"
LAMBDA_PHRASE_GEN_WAS_SET="${LAMBDA_PHRASE_GEN+x}"
PHRASE_TARGET_MODE_WAS_SET="${PHRASE_TARGET_MODE+x}"
PHRASE_SLOT_SCHEMA_WAS_SET="${PHRASE_SLOT_SCHEMA+x}"
MAX_PHRASE_SLOTS_WAS_SET="${MAX_PHRASE_SLOTS+x}"
PHRASE_SLOT_MAX_LEN_WAS_SET="${PHRASE_SLOT_MAX_LEN+x}"
PHRASE_SLOT_PLANNER_ENABLE_WAS_SET="${PHRASE_SLOT_PLANNER_ENABLE+x}"
PHRASE_SLOT_PLANNER_FLOW_ENABLE_WAS_SET="${PHRASE_SLOT_PLANNER_FLOW_ENABLE+x}"
PHRASE_SLOT_PLANNER_FLOW_SCALE_WAS_SET="${PHRASE_SLOT_PLANNER_FLOW_SCALE+x}"
PHRASE_SLOT_PLANNER_FLOW_SLOT_TYPES_WAS_SET="${PHRASE_SLOT_PLANNER_FLOW_SLOT_TYPES+x}"
PHRASE_SLOT_GUIDANCE_ENABLE_WAS_SET="${PHRASE_SLOT_GUIDANCE_ENABLE+x}"
PHRASE_SLOT_ROLE_ANCHOR_ENABLE_WAS_SET="${PHRASE_SLOT_ROLE_ANCHOR_ENABLE+x}"
PHRASE_SLOT_ROLE_ANCHOR_TOPK_WAS_SET="${PHRASE_SLOT_ROLE_ANCHOR_TOPK+x}"
PHRASE_SLOT_ROLE_ANCHOR_SCALE_WAS_SET="${PHRASE_SLOT_ROLE_ANCHOR_SCALE+x}"
PHRASE_SLOT_ROLE_ANCHOR_SLOT_TYPES_WAS_SET="${PHRASE_SLOT_ROLE_ANCHOR_SLOT_TYPES+x}"
PHRASE_SLOT_PRESENCE_ENABLE_WAS_SET="${PHRASE_SLOT_PRESENCE_ENABLE+x}"
PHRASE_SLOT_PRESENCE_SUPPORT_ENABLE_WAS_SET="${PHRASE_SLOT_PRESENCE_SUPPORT_ENABLE+x}"
PHRASE_SLOT_PRESENCE_EVIDENCE_ENABLE_WAS_SET="${PHRASE_SLOT_PRESENCE_EVIDENCE_ENABLE+x}"
PHRASE_SLOT_PRESENCE_CONTEXT_SLOT_TYPES_WAS_SET="${PHRASE_SLOT_PRESENCE_CONTEXT_SLOT_TYPES+x}"
PHRASE_SLOT_ACTIVE_SLOT_TYPES_WAS_SET="${PHRASE_SLOT_ACTIVE_SLOT_TYPES+x}"
PHRASE_SLOT_MULTIREF_ENABLE_WAS_SET="${PHRASE_SLOT_MULTIREF_ENABLE+x}"
PHRASE_SLOT_MULTIREF_MAX_REFS_WAS_SET="${PHRASE_SLOT_MULTIREF_MAX_REFS+x}"
ALLOWED_GPU_CSV="${ALLOWED_GPU_CSV:-2,3}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-48}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_TRAIN_STEPS_PER_EPOCH="${MAX_TRAIN_STEPS_PER_EPOCH:-0}"
PHRASE_DECODER_ENABLE="${PHRASE_DECODER_ENABLE:-0}"
PHRASE_CONDITION_ENABLE="${PHRASE_CONDITION_ENABLE:-0}"
LAMBDA_PHRASE="${LAMBDA_PHRASE:-0.0}"
LAMBDA_PHRASE_GEN="${LAMBDA_PHRASE_GEN:-0.0}"
PHRASE_MEMORY_MODE="${PHRASE_MEMORY_MODE:-pooled}"
PHRASE_TARGET_MODE="${PHRASE_TARGET_MODE:-flat}"
PHRASE_SLOT_SCHEMA="${PHRASE_SLOT_SCHEMA:-raw}"
PHRASE_INCLUDE_ATTR_UNITS="${PHRASE_INCLUDE_ATTR_UNITS:-0}"
PHRASE_INCLUDE_SCENE_UNITS="${PHRASE_INCLUDE_SCENE_UNITS:-0}"
PHRASE_ATTR_SCENE_UNITS_START_EPOCH="${PHRASE_ATTR_SCENE_UNITS_START_EPOCH:-0}"
PHRASE_ATTR_SCENE_UNITS_END_EPOCH="${PHRASE_ATTR_SCENE_UNITS_END_EPOCH:-0}"
PHRASE_INCLUDE_VIDEO_PHRASE_UNITS="${PHRASE_INCLUDE_VIDEO_PHRASE_UNITS:-0}"
PHRASE_INCLUDE_VIDEO_ATTR_UNITS="${PHRASE_INCLUDE_VIDEO_ATTR_UNITS:-0}"
PHRASE_INCLUDE_VIDEO_SCENE_UNITS="${PHRASE_INCLUDE_VIDEO_SCENE_UNITS:-0}"
PHRASE_VIDEO_PHRASE_MIN_SUPPORT="${PHRASE_VIDEO_PHRASE_MIN_SUPPORT:-2}"
PHRASE_VIDEO_PHRASE_MAX_UNITS="${PHRASE_VIDEO_PHRASE_MAX_UNITS:-4}"
MAX_PHRASE_SLOTS="${MAX_PHRASE_SLOTS:-4}"
PHRASE_SLOT_MAX_LEN="${PHRASE_SLOT_MAX_LEN:-24}"
PHRASE_SLOT_PLANNER_ENABLE="${PHRASE_SLOT_PLANNER_ENABLE:-0}"
PHRASE_SLOT_PLANNER_FLOW_ENABLE="${PHRASE_SLOT_PLANNER_FLOW_ENABLE:-0}"
PHRASE_SLOT_PLANNER_FLOW_SCALE="${PHRASE_SLOT_PLANNER_FLOW_SCALE:-0.20}"
PHRASE_SLOT_PLANNER_FLOW_SLOT_TYPES="${PHRASE_SLOT_PLANNER_FLOW_SLOT_TYPES:-}"
PHRASE_SLOT_GUIDANCE_ENABLE="${PHRASE_SLOT_GUIDANCE_ENABLE:-0}"
PHRASE_SLOT_ROLE_ANCHOR_ENABLE="${PHRASE_SLOT_ROLE_ANCHOR_ENABLE:-0}"
PHRASE_SLOT_ROLE_ANCHOR_TOPK="${PHRASE_SLOT_ROLE_ANCHOR_TOPK:-4}"
PHRASE_SLOT_ROLE_ANCHOR_SCALE="${PHRASE_SLOT_ROLE_ANCHOR_SCALE:-1.0}"
PHRASE_SLOT_ROLE_ANCHOR_SLOT_TYPES="${PHRASE_SLOT_ROLE_ANCHOR_SLOT_TYPES:-}"
PHRASE_SLOT_PRESENCE_ENABLE="${PHRASE_SLOT_PRESENCE_ENABLE:-0}"
PHRASE_SLOT_PRESENCE_SUPPORT_ENABLE="${PHRASE_SLOT_PRESENCE_SUPPORT_ENABLE:-0}"
PHRASE_SLOT_PRESENCE_EVIDENCE_ENABLE="${PHRASE_SLOT_PRESENCE_EVIDENCE_ENABLE:-0}"
PHRASE_SLOT_PRESENCE_CONTEXT_SLOT_TYPES="${PHRASE_SLOT_PRESENCE_CONTEXT_SLOT_TYPES:-}"
PHRASE_SLOT_PRESENCE_THRESHOLD="${PHRASE_SLOT_PRESENCE_THRESHOLD:-0.5}"
PHRASE_SLOT_ACTIVE_SLOT_TYPES="${PHRASE_SLOT_ACTIVE_SLOT_TYPES:-}"
PHRASE_SLOT_PRESENCE_CALIBRATION_MODE="${PHRASE_SLOT_PRESENCE_CALIBRATION_MODE:-none}"
PHRASE_SLOT_PRESENCE_CALIBRATION_SLOT_TYPES="${PHRASE_SLOT_PRESENCE_CALIBRATION_SLOT_TYPES:-}"
PHRASE_SLOT_PRESENCE_THRESHOLD_MIN="${PHRASE_SLOT_PRESENCE_THRESHOLD_MIN:-0.35}"
PHRASE_SLOT_PRESENCE_THRESHOLD_MAX="${PHRASE_SLOT_PRESENCE_THRESHOLD_MAX:-0.65}"
PHRASE_SLOT_REWEIGHT_ENABLE="${PHRASE_SLOT_REWEIGHT_ENABLE:-0}"
PHRASE_SLOT_REWEIGHT_POWER="${PHRASE_SLOT_REWEIGHT_POWER:-0.5}"
PHRASE_SLOT_REWEIGHT_MIN="${PHRASE_SLOT_REWEIGHT_MIN:-1.0}"
PHRASE_SLOT_REWEIGHT_MAX="${PHRASE_SLOT_REWEIGHT_MAX:-4.0}"
PHRASE_SLOT_MULTIREF_ENABLE="${PHRASE_SLOT_MULTIREF_ENABLE:-0}"
PHRASE_SLOT_MULTIREF_MAX_REFS="${PHRASE_SLOT_MULTIREF_MAX_REFS:-0}"
PHRASE_SLOT_MULTIREF_REDUCE="${PHRASE_SLOT_MULTIREF_REDUCE:-mean}"
PHRASE_SLOT_MULTIREF_GAIN="${PHRASE_SLOT_MULTIREF_GAIN:-1.0}"
PHRASE_SLOT_MULTIREF_CHUNK_SIZE="${PHRASE_SLOT_MULTIREF_CHUNK_SIZE:-4}"
PHRASE_SLOT_PRED_MULTIREF_REDUCE="${PHRASE_SLOT_PRED_MULTIREF_REDUCE:-inherit}"
PHRASE_SLOT_PRED_MULTIREF_SOFTMIN_TEMP="${PHRASE_SLOT_PRED_MULTIREF_SOFTMIN_TEMP:-1.0}"
PHRASE_SLOT_FAMILY_EXPAND_MODE="${PHRASE_SLOT_FAMILY_EXPAND_MODE:-none}"
PHRASE_SLOT_FAMILY_SAMPLE_MODE="${PHRASE_SLOT_FAMILY_SAMPLE_MODE:-first}"
PHRASE_SLOT_FAMILY_SAMPLE_SEED="${PHRASE_SLOT_FAMILY_SAMPLE_SEED:-${SEED}}"
PRIOR_HEAD_TYPE="${PRIOR_HEAD_TYPE:-simple}"
PRIOR_HEAD_NUM_HEADS="${PRIOR_HEAD_NUM_HEADS:-8}"
PRIOR_HEAD_HIDDEN_DIM="${PRIOR_HEAD_HIDDEN_DIM:-2048}"
PRIOR_HEAD_NUM_BLOCKS="${PRIOR_HEAD_NUM_BLOCKS:-4}"
STRUCT_CONDITION_QUERY_BRIDGE_ENABLE="${STRUCT_CONDITION_QUERY_BRIDGE_ENABLE:-0}"
STRUCT_CONDITION_QUERY_BRIDGE_NUM_QUERIES="${STRUCT_CONDITION_QUERY_BRIDGE_NUM_QUERIES:-4}"
STRUCT_CONDITION_QUERY_BRIDGE_SCALE="${STRUCT_CONDITION_QUERY_BRIDGE_SCALE:-0.15}"
STRUCT_CONDITION_QUERY_BRIDGE_MEMORY_ENABLE="${STRUCT_CONDITION_QUERY_BRIDGE_MEMORY_ENABLE:-0}"
STRUCT_CONDITION_QUERY_BRIDGE_MEMORY_SCALE="${STRUCT_CONDITION_QUERY_BRIDGE_MEMORY_SCALE:-0.15}"
STRUCT_CONDITION_QUERY_BRIDGE_HIDDEN_ENABLE="${STRUCT_CONDITION_QUERY_BRIDGE_HIDDEN_ENABLE:-0}"
STRUCT_CONDITION_QUERY_BRIDGE_HIDDEN_SCALE="${STRUCT_CONDITION_QUERY_BRIDGE_HIDDEN_SCALE:-0.15}"
PHRASE_CONDITION_SLOT_AWARE_ENABLE="${PHRASE_CONDITION_SLOT_AWARE_ENABLE:-0}"
PHRASE_CONDITION_SLOT_SELECTIVE_ENABLE="${PHRASE_CONDITION_SLOT_SELECTIVE_ENABLE:-0}"
PHRASE_CONDITION_AUX_SCALE="${PHRASE_CONDITION_AUX_SCALE:-0.15}"
PHRASE_CONDITION_SLOT_RESIDUAL_ENABLE="${PHRASE_CONDITION_SLOT_RESIDUAL_ENABLE:-0}"
PHRASE_CONDITION_SLOT_RESIDUAL_SCALE="${PHRASE_CONDITION_SLOT_RESIDUAL_SCALE:-0.15}"
PHRASE_CONDITION_SLOT_RESIDUAL_SLOT_TYPES="${PHRASE_CONDITION_SLOT_RESIDUAL_SLOT_TYPES:-}"
PHRASE_CONDITION_FAMILY_BRIDGE_ENABLE="${PHRASE_CONDITION_FAMILY_BRIDGE_ENABLE:-0}"
PHRASE_CONDITION_FAMILY_BRIDGE_SCALE="${PHRASE_CONDITION_FAMILY_BRIDGE_SCALE:-0.20}"
PHRASE_CONDITION_CANDIDATE_BIAS_ENABLE="${PHRASE_CONDITION_CANDIDATE_BIAS_ENABLE:-0}"
PHRASE_CONDITION_CANDIDATE_BIAS_SCALE="${PHRASE_CONDITION_CANDIDATE_BIAS_SCALE:-0.10}"
PHRASE_CONDITION_CANDIDATE_TOPK="${PHRASE_CONDITION_CANDIDATE_TOPK:-12}"
PHRASE_CONDITION_CANDIDATE_SLOT_TYPES="${PHRASE_CONDITION_CANDIDATE_SLOT_TYPES:-}"
PHRASE_CONDITION_QUERY_BRIDGE_ENABLE="${PHRASE_CONDITION_QUERY_BRIDGE_ENABLE:-0}"
PHRASE_CONDITION_QUERY_BRIDGE_NUM_QUERIES="${PHRASE_CONDITION_QUERY_BRIDGE_NUM_QUERIES:-4}"
PHRASE_CONDITION_QUERY_BRIDGE_SCALE="${PHRASE_CONDITION_QUERY_BRIDGE_SCALE:-0.15}"
PHRASE_CONDITION_CORE_SLOT_TYPES="${PHRASE_CONDITION_CORE_SLOT_TYPES:-}"
PHRASE_CONDITION_AUX_SLOT_TYPES="${PHRASE_CONDITION_AUX_SLOT_TYPES:-}"
PHRASE_CONDITION_TRAIN_USE_PREDICTED="${PHRASE_CONDITION_TRAIN_USE_PREDICTED:-0}"
PHRASE_CONDITION_PRED_DETACH="${PHRASE_CONDITION_PRED_DETACH:-1}"
PHRASE_CONDITION_PRED_DETACH_UNTIL_EPOCH="${PHRASE_CONDITION_PRED_DETACH_UNTIL_EPOCH:-0}"
PHRASE_CONDITION_TEACHER_SOURCE="${PHRASE_CONDITION_TEACHER_SOURCE:-single_ref}"
PHRASE_SLOT_DECODE_ANCHOR_ENABLE="${PHRASE_SLOT_DECODE_ANCHOR_ENABLE:-0}"
PHRASE_SLOT_DECODE_ANCHOR_TOPK="${PHRASE_SLOT_DECODE_ANCHOR_TOPK:-8}"
PHRASE_SLOT_DECODE_ANCHOR_SCALE="${PHRASE_SLOT_DECODE_ANCHOR_SCALE:-1.0}"
PHRASE_SLOT_DECODE_ANCHOR_EARLY_SCALE="${PHRASE_SLOT_DECODE_ANCHOR_EARLY_SCALE:-1.25}"
PHRASE_SLOT_DECODE_ANCHOR_FAMILY_SCALE="${PHRASE_SLOT_DECODE_ANCHOR_FAMILY_SCALE:-0.75}"
PHRASE_SLOT_DECODE_ANCHOR_FAMILY_TOPK="${PHRASE_SLOT_DECODE_ANCHOR_FAMILY_TOPK:-64}"
PHRASE_SLOT_DECODE_ANCHOR_FAMILY_MIN_COUNT="${PHRASE_SLOT_DECODE_ANCHOR_FAMILY_MIN_COUNT:-2}"
PHRASE_SLOT_DECODE_ANCHOR_STOPWORD_PENALTY="${PHRASE_SLOT_DECODE_ANCHOR_STOPWORD_PENALTY:-0.75}"
PHRASE_SLOT_DECODE_ANCHOR_STOPWORD_STEPS="${PHRASE_SLOT_DECODE_ANCHOR_STOPWORD_STEPS:-2}"
PHRASE_SLOT_DECODE_ANCHOR_DEBUG_TOPK="${PHRASE_SLOT_DECODE_ANCHOR_DEBUG_TOPK:-8}"
LAMBDA_PHRASE_SLOT_PRESENCE="${LAMBDA_PHRASE_SLOT_PRESENCE:-0.0}"
LAMBDA_PHRASE_SLOT_DIV="${LAMBDA_PHRASE_SLOT_DIV:-0.0}"
LAMBDA_PHRASE_REF_SLOT_ALIGN="${LAMBDA_PHRASE_REF_SLOT_ALIGN:-0.0}"
LAMBDA_PHRASE_REF_BRIDGE="${LAMBDA_PHRASE_REF_BRIDGE:-0.0}"
LAMBDA_PHRASE_BRIDGE="${LAMBDA_PHRASE_BRIDGE:-0.0}"
LAMBDA_PHRASE_PRED_GEN="${LAMBDA_PHRASE_PRED_GEN:-0.0}"
PHRASE_PRED_GEN_START_EPOCH="${PHRASE_PRED_GEN_START_EPOCH:-1}"
LAMBDA_PHRASE_SLOT_SOURCE_ALIGN="${LAMBDA_PHRASE_SLOT_SOURCE_ALIGN:-0.0}"
LAMBDA_PHRASE_SLOT_SOURCE_COMP="${LAMBDA_PHRASE_SLOT_SOURCE_COMP:-0.0}"
PHRASE_SLOT_SOURCE_COMP_MARGIN="${PHRASE_SLOT_SOURCE_COMP_MARGIN:-0.05}"
LAMBDA_ATTR="${LAMBDA_ATTR:-0.0}"
LAMBDA_SCENE="${LAMBDA_SCENE:-0.0}"
PRIOR_CAPTION_POS_WEIGHT="${PRIOR_CAPTION_POS_WEIGHT:-1.0}"
PRIOR_VIDEO_ONLY_POS_WEIGHT="${PRIOR_VIDEO_ONLY_POS_WEIGHT:-1.0}"
LAMBDA_CE="${LAMBDA_CE:-1.0}"
LAMBDA_ENTITY="${LAMBDA_ENTITY:-0.05}"
LAMBDA_ACTION="${LAMBDA_ACTION:-0.05}"
PRIOR_LOSS_TYPE="${PRIOR_LOSS_TYPE:-bce}"
PRIOR_ASL_GAMMA_NEG="${PRIOR_ASL_GAMMA_NEG:-4.0}"
PRIOR_ASL_GAMMA_POS="${PRIOR_ASL_GAMMA_POS:-1.0}"
PRIOR_ASL_CLIP="${PRIOR_ASL_CLIP:-0.05}"
PRIOR_ASL_EPS="${PRIOR_ASL_EPS:-1e-8}"
ENTITY_PRIOR_CAPTION_POS_WEIGHT="${ENTITY_PRIOR_CAPTION_POS_WEIGHT:-}"
ENTITY_PRIOR_VIDEO_ONLY_POS_WEIGHT="${ENTITY_PRIOR_VIDEO_ONLY_POS_WEIGHT:-}"
ACTION_PRIOR_CAPTION_POS_WEIGHT="${ACTION_PRIOR_CAPTION_POS_WEIGHT:-}"
ACTION_PRIOR_VIDEO_ONLY_POS_WEIGHT="${ACTION_PRIOR_VIDEO_ONLY_POS_WEIGHT:-}"
ATTR_PRIOR_CAPTION_POS_WEIGHT="${ATTR_PRIOR_CAPTION_POS_WEIGHT:-}"
ATTR_PRIOR_VIDEO_ONLY_POS_WEIGHT="${ATTR_PRIOR_VIDEO_ONLY_POS_WEIGHT:-}"
SCENE_PRIOR_CAPTION_POS_WEIGHT="${SCENE_PRIOR_CAPTION_POS_WEIGHT:-}"
SCENE_PRIOR_VIDEO_ONLY_POS_WEIGHT="${SCENE_PRIOR_VIDEO_ONLY_POS_WEIGHT:-}"
PRIOR_HEAD_NUM_CLUSTERS="${PRIOR_HEAD_NUM_CLUSTERS:-16}"
PRIOR_HEAD_EXPANSION="${PRIOR_HEAD_EXPANSION:-2}"
PRIOR_HEAD_GROUPS="${PRIOR_HEAD_GROUPS:-8}"
INIT_SKIP_STRUCTURED_VOCAB_MODULES="${INIT_SKIP_STRUCTURED_VOCAB_MODULES:-0}"
AUX_VISUAL_ENABLE="${AUX_VISUAL_ENABLE:-0}"
AUX_RAW_GLOBAL_ENABLE="${AUX_RAW_GLOBAL_ENABLE:-0}"
AUX_RAW_GLOBAL_FEATS_PATH="${AUX_RAW_GLOBAL_FEATS_PATH:-/mnt/sda/Disk_D/zhangwei/projects/VC/datasets/MSVD/feats/ViT-B-32_k_split_ks12_features.pickle}"
AUX_PATCH_ENABLE="${AUX_PATCH_ENABLE:-0}"
AUX_PATCH_ROOT="${AUX_PATCH_ROOT:-/mnt/sda/Disk_D/zhangwei/projects/VC/datasets/MSVD/feats/clip_patch_ks12splits_vitb32}"
AUX_PATCH_BLOCK="${AUX_PATCH_BLOCK:-6}"
AUX_VISUAL_RAW_GLOBAL_DIM="${AUX_VISUAL_RAW_GLOBAL_DIM:-512}"
AUX_VISUAL_PATCH_DIM="${AUX_VISUAL_PATCH_DIM:-768}"
AUX_VISUAL_PRIOR_SCALE="${AUX_VISUAL_PRIOR_SCALE:-0.15}"
AUX_VISUAL_STRUCT_SCALE="${AUX_VISUAL_STRUCT_SCALE:-0.10}"
AUX_VISUAL_MEMORY_SCALE="${AUX_VISUAL_MEMORY_SCALE:-0.10}"
PHRASE_PROGRESS_ENABLE="${PHRASE_PROGRESS_ENABLE:-0}"
PHRASE_PROGRESS_MEMORY_SCALE="${PHRASE_PROGRESS_MEMORY_SCALE:-0.10}"
PHRASE_PROGRESS_SOURCE_SCALE="${PHRASE_PROGRESS_SOURCE_SCALE:-0.10}"
GPU_BIND_MAX_POLLS="${GPU_BIND_MAX_POLLS:-15}"
GPU_BIND_POLL_SEC="${GPU_BIND_POLL_SEC:-2}"

is_allowed_gpu() {
  local gpu_id="$1"
  case ",${ALLOWED_GPU_CSV}," in
    *,"${gpu_id}",*) return 0 ;;
    *) return 1 ;;
  esac
}

trim_csv_field() {
  echo "$1" | xargs
}

declare -A GPU_UUID_TO_INDEX=()
while IFS=',' read -r idx uuid; do
  idx="$(trim_csv_field "$idx")"
  uuid="$(trim_csv_field "$uuid")"
  if [[ -n "$idx" && -n "$uuid" ]]; then
    GPU_UUID_TO_INDEX["$uuid"]="$idx"
  fi
done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits)

capture_gpu_snapshot() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | while IFS=',' read -r idx mem util; do
    idx="$(trim_csv_field "$idx")"
    mem="$(trim_csv_field "$mem")"
    util="$(trim_csv_field "$util")"
    echo "${idx}:${mem}:${util}"
  done
}

gpu_snapshot_line() {
  local gpu_id="$1"
  local snapshot="$2"
  echo "$snapshot" | grep "^${gpu_id}:"
}

resolve_pid_gpu_indices() {
  local target_pid="$1"
  local matches=()
  while IFS=',' read -r uuid pid _used_mem _proc_name; do
    uuid="$(trim_csv_field "$uuid")"
    pid="$(trim_csv_field "$pid")"
    if [[ "$pid" != "$target_pid" ]]; then
      continue
    fi
    local idx="${GPU_UUID_TO_INDEX[$uuid]:-}"
    if [[ -n "$idx" ]]; then
      matches+=("$idx")
    fi
  done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader,nounits 2>/dev/null || true)
  if [[ "${#matches[@]}" -eq 0 ]]; then
    return 0
  fi
  printf '%s\n' "${matches[@]}" | sort -u | paste -sd',' -
}

if ! is_allowed_gpu "$GPU_ID"; then
  echo "[GPU_GUARD] forbidden GPU_ID=${GPU_ID}. allowed=${ALLOWED_GPU_CSV}. GPU0/GPU1 are strictly forbidden." >&2
  exit 1
fi

GPU_SNAPSHOT_BEFORE="$(capture_gpu_snapshot)"
echo "[GPU_GUARD][precheck-1] requested_gpu=${GPU_ID} allowed=${ALLOWED_GPU_CSV}"
echo "[GPU_GUARD][precheck-2] before_launch_gpus"
echo "$GPU_SNAPSHOT_BEFORE"

SERVER_ROOT="${SERVER_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON_BIN="/mnt/sda/Disk_D/zhangwei/anaconda3/envs/vcr/bin/python"
TRAIN_PY="${SERVER_ROOT}/train_structured_refine_monitored.py"

FEATS_PATH="/mnt/sda/Disk_D/zhangwei/projects/VC/datasets/MSVD/feats/clip4clip_vitb32_k_split_ks12_features.pickle"
ANNOTATIONS_PATH="/mnt/sda/Disk_D/zhangwei/projects/VC/datasets/MSVD/annotations_preprocessed.txt"
STRUCTURED_GT_PATH="${STRUCTURED_GT_PATH:-${SERVER_ROOT}/annotations/msvd_structured_train_api.json}"
ANCHOR_CKPT="${SERVER_ROOT}/runs/local_noleak_api_structv2_e8_a/msvd_structured_api_structv2_e8_a_20260303_010342/checkpoints/epoch_003.pt"
TRAINING_STAGE="${TRAINING_STAGE:-joint}"
INIT_MODEL_CKPT="${INIT_MODEL_CKPT:-}"
INIT_CAPTION_CKPT="${INIT_CAPTION_CKPT:-${ANCHOR_CKPT}}"
INIT_PHRASE_INTERFACE_INHERITED=()

maybe_assign_if_unset() {
  local var_name="$1"
  local value="$2"
  local was_set_var="$3"
  if [[ -z "${!was_set_var:-}" ]]; then
    printf -v "${var_name}" '%s' "${value}"
    INIT_PHRASE_INTERFACE_INHERITED+=("${var_name}=${value}")
  fi
}

maybe_inherit_stage3_phrase_interface() {
  local ckpt_path="$1"
  if [[ -z "${ckpt_path}" ]]; then
    return 0
  fi
  local run_dir
  run_dir="$(dirname "$(dirname "${ckpt_path}")")"
  local args_json="${run_dir}/args.json"
  if [[ ! -f "${args_json}" ]]; then
    return 0
  fi

  local payload
  payload="$("${PYTHON_BIN}" - "${args_json}" <<'PY'
import json
import sys

args_path = sys.argv[1]
with open(args_path, "r", encoding="utf-8") as f:
    args = json.load(f)

mapping = (
    ("PHRASE_TARGET_MODE", "phrase_target_mode"),
    ("PHRASE_SLOT_SCHEMA", "phrase_slot_schema"),
    ("MAX_PHRASE_SLOTS", "max_phrase_slots"),
    ("PHRASE_SLOT_MAX_LEN", "phrase_slot_max_len"),
    ("PHRASE_SLOT_PLANNER_ENABLE", "phrase_slot_planner_enable"),
    ("PHRASE_SLOT_PLANNER_FLOW_ENABLE", "phrase_slot_planner_flow_enable"),
    ("PHRASE_SLOT_PLANNER_FLOW_SCALE", "phrase_slot_planner_flow_scale"),
    ("PHRASE_SLOT_PLANNER_FLOW_SLOT_TYPES", "phrase_slot_planner_flow_slot_types"),
    ("PHRASE_SLOT_GUIDANCE_ENABLE", "phrase_slot_guidance_enable"),
    ("PHRASE_SLOT_ROLE_ANCHOR_ENABLE", "phrase_slot_role_anchor_enable"),
    ("PHRASE_SLOT_ROLE_ANCHOR_TOPK", "phrase_slot_role_anchor_topk"),
    ("PHRASE_SLOT_ROLE_ANCHOR_SCALE", "phrase_slot_role_anchor_scale"),
    ("PHRASE_SLOT_ROLE_ANCHOR_SLOT_TYPES", "phrase_slot_role_anchor_slot_types"),
    ("PHRASE_SLOT_PRESENCE_ENABLE", "phrase_slot_presence_enable"),
    ("PHRASE_SLOT_PRESENCE_SUPPORT_ENABLE", "phrase_slot_presence_support_enable"),
    ("PHRASE_SLOT_PRESENCE_EVIDENCE_ENABLE", "phrase_slot_presence_evidence_enable"),
    ("PHRASE_SLOT_PRESENCE_CONTEXT_SLOT_TYPES", "phrase_slot_presence_context_slot_types"),
    ("PHRASE_SLOT_ACTIVE_SLOT_TYPES", "phrase_slot_active_slot_types"),
    ("PHRASE_SLOT_MULTIREF_ENABLE", "phrase_slot_multiref_enable"),
    ("PHRASE_SLOT_MULTIREF_MAX_REFS", "phrase_slot_multiref_max_refs"),
)

for env_name, json_key in mapping:
    value = args.get(json_key, None)
    if value is None:
        continue
    if isinstance(value, bool):
        value = int(value)
    print(f"{env_name}={value}")
PY
)" || return 0

  while IFS='=' read -r key value; do
    [[ -n "${key}" ]] || continue
    case "${key}" in
      PHRASE_TARGET_MODE) maybe_assign_if_unset "${key}" "${value}" "PHRASE_TARGET_MODE_WAS_SET" ;;
      PHRASE_SLOT_SCHEMA) maybe_assign_if_unset "${key}" "${value}" "PHRASE_SLOT_SCHEMA_WAS_SET" ;;
      MAX_PHRASE_SLOTS) maybe_assign_if_unset "${key}" "${value}" "MAX_PHRASE_SLOTS_WAS_SET" ;;
      PHRASE_SLOT_MAX_LEN) maybe_assign_if_unset "${key}" "${value}" "PHRASE_SLOT_MAX_LEN_WAS_SET" ;;
      PHRASE_SLOT_PLANNER_ENABLE) maybe_assign_if_unset "${key}" "${value}" "PHRASE_SLOT_PLANNER_ENABLE_WAS_SET" ;;
      PHRASE_SLOT_PLANNER_FLOW_ENABLE) maybe_assign_if_unset "${key}" "${value}" "PHRASE_SLOT_PLANNER_FLOW_ENABLE_WAS_SET" ;;
      PHRASE_SLOT_PLANNER_FLOW_SCALE) maybe_assign_if_unset "${key}" "${value}" "PHRASE_SLOT_PLANNER_FLOW_SCALE_WAS_SET" ;;
      PHRASE_SLOT_PLANNER_FLOW_SLOT_TYPES) maybe_assign_if_unset "${key}" "${value}" "PHRASE_SLOT_PLANNER_FLOW_SLOT_TYPES_WAS_SET" ;;
      PHRASE_SLOT_GUIDANCE_ENABLE) maybe_assign_if_unset "${key}" "${value}" "PHRASE_SLOT_GUIDANCE_ENABLE_WAS_SET" ;;
      PHRASE_SLOT_ROLE_ANCHOR_ENABLE) maybe_assign_if_unset "${key}" "${value}" "PHRASE_SLOT_ROLE_ANCHOR_ENABLE_WAS_SET" ;;
      PHRASE_SLOT_ROLE_ANCHOR_TOPK) maybe_assign_if_unset "${key}" "${value}" "PHRASE_SLOT_ROLE_ANCHOR_TOPK_WAS_SET" ;;
      PHRASE_SLOT_ROLE_ANCHOR_SCALE) maybe_assign_if_unset "${key}" "${value}" "PHRASE_SLOT_ROLE_ANCHOR_SCALE_WAS_SET" ;;
      PHRASE_SLOT_ROLE_ANCHOR_SLOT_TYPES) maybe_assign_if_unset "${key}" "${value}" "PHRASE_SLOT_ROLE_ANCHOR_SLOT_TYPES_WAS_SET" ;;
      PHRASE_SLOT_PRESENCE_ENABLE) maybe_assign_if_unset "${key}" "${value}" "PHRASE_SLOT_PRESENCE_ENABLE_WAS_SET" ;;
      PHRASE_SLOT_PRESENCE_SUPPORT_ENABLE) maybe_assign_if_unset "${key}" "${value}" "PHRASE_SLOT_PRESENCE_SUPPORT_ENABLE_WAS_SET" ;;
      PHRASE_SLOT_PRESENCE_EVIDENCE_ENABLE) maybe_assign_if_unset "${key}" "${value}" "PHRASE_SLOT_PRESENCE_EVIDENCE_ENABLE_WAS_SET" ;;
      PHRASE_SLOT_PRESENCE_CONTEXT_SLOT_TYPES) maybe_assign_if_unset "${key}" "${value}" "PHRASE_SLOT_PRESENCE_CONTEXT_SLOT_TYPES_WAS_SET" ;;
      PHRASE_SLOT_ACTIVE_SLOT_TYPES) maybe_assign_if_unset "${key}" "${value}" "PHRASE_SLOT_ACTIVE_SLOT_TYPES_WAS_SET" ;;
      PHRASE_SLOT_MULTIREF_ENABLE) maybe_assign_if_unset "${key}" "${value}" "PHRASE_SLOT_MULTIREF_ENABLE_WAS_SET" ;;
      PHRASE_SLOT_MULTIREF_MAX_REFS) maybe_assign_if_unset "${key}" "${value}" "PHRASE_SLOT_MULTIREF_MAX_REFS_WAS_SET" ;;
    esac
  done <<< "${payload}"
}

if [[ "${TRAINING_STAGE}" == "stage3_sentence" && -n "${INIT_MODEL_CKPT}" ]]; then
  maybe_inherit_stage3_phrase_interface "${INIT_MODEL_CKPT}"
fi

RUN_ROOT="${SERVER_ROOT}/runs/phrase_mainline"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-}"
RUN_NAME="msvd_phrase_${VARIANT_LOWER}_${TS}"
if [[ -n "${RUN_NAME_SUFFIX}" ]]; then
  RUN_NAME="${RUN_NAME}_${RUN_NAME_SUFFIX}"
fi
RUN_DIR="${RUN_ROOT}/${RUN_NAME}"
LOG_PATH="${RUN_DIR}/train.log"
PID_PATH="${RUN_DIR}/train.pid"

MODEL_TAG="structured_phrase_b"

case "${VARIANT}" in
  B|b)
    MODEL_TAG="structured_phrase_b"
    ;;
  C|c)
    if [[ -z "${PHRASE_DECODER_ENABLE_WAS_SET}" ]]; then
      PHRASE_DECODER_ENABLE=1
    fi
    if [[ -z "${LAMBDA_PHRASE_WAS_SET}" ]]; then
      LAMBDA_PHRASE=0.02
    fi
    if [[ -z "${LAMBDA_PHRASE_GEN_WAS_SET}" ]]; then
      LAMBDA_PHRASE_GEN=0.20
    fi
    MODEL_TAG="structured_phrase_c"
    ;;
  D|d)
    if [[ -z "${PHRASE_DECODER_ENABLE_WAS_SET}" ]]; then
      PHRASE_DECODER_ENABLE=1
    fi
    if [[ -z "${PHRASE_CONDITION_ENABLE_WAS_SET}" ]]; then
      PHRASE_CONDITION_ENABLE=1
    fi
    if [[ -z "${LAMBDA_PHRASE_WAS_SET}" ]]; then
      LAMBDA_PHRASE=0.02
    fi
    if [[ -z "${LAMBDA_PHRASE_GEN_WAS_SET}" ]]; then
      LAMBDA_PHRASE_GEN=0.20
    fi
    MODEL_TAG="structured_phrase_d"
    ;;
  *)
    echo "Unsupported variant: ${VARIANT}. Use B, C, or D." >&2
    exit 1
    ;;
esac

mkdir -p "${RUN_DIR}"

EXTRA_PRIOR_ARGS=()
if [[ -n "${ENTITY_PRIOR_CAPTION_POS_WEIGHT}" ]]; then
  EXTRA_PRIOR_ARGS+=(--entity_prior_caption_pos_weight "${ENTITY_PRIOR_CAPTION_POS_WEIGHT}")
fi
if [[ -n "${ENTITY_PRIOR_VIDEO_ONLY_POS_WEIGHT}" ]]; then
  EXTRA_PRIOR_ARGS+=(--entity_prior_video_only_pos_weight "${ENTITY_PRIOR_VIDEO_ONLY_POS_WEIGHT}")
fi
if [[ -n "${ACTION_PRIOR_CAPTION_POS_WEIGHT}" ]]; then
  EXTRA_PRIOR_ARGS+=(--action_prior_caption_pos_weight "${ACTION_PRIOR_CAPTION_POS_WEIGHT}")
fi
if [[ -n "${ACTION_PRIOR_VIDEO_ONLY_POS_WEIGHT}" ]]; then
  EXTRA_PRIOR_ARGS+=(--action_prior_video_only_pos_weight "${ACTION_PRIOR_VIDEO_ONLY_POS_WEIGHT}")
fi
if [[ -n "${ATTR_PRIOR_CAPTION_POS_WEIGHT}" ]]; then
  EXTRA_PRIOR_ARGS+=(--attr_prior_caption_pos_weight "${ATTR_PRIOR_CAPTION_POS_WEIGHT}")
fi
if [[ -n "${ATTR_PRIOR_VIDEO_ONLY_POS_WEIGHT}" ]]; then
  EXTRA_PRIOR_ARGS+=(--attr_prior_video_only_pos_weight "${ATTR_PRIOR_VIDEO_ONLY_POS_WEIGHT}")
fi
if [[ -n "${SCENE_PRIOR_CAPTION_POS_WEIGHT}" ]]; then
  EXTRA_PRIOR_ARGS+=(--scene_prior_caption_pos_weight "${SCENE_PRIOR_CAPTION_POS_WEIGHT}")
fi
if [[ -n "${SCENE_PRIOR_VIDEO_ONLY_POS_WEIGHT}" ]]; then
  EXTRA_PRIOR_ARGS+=(--scene_prior_video_only_pos_weight "${SCENE_PRIOR_VIDEO_ONLY_POS_WEIGHT}")
fi

EXTRA_INIT_ARGS=(--training_stage "${TRAINING_STAGE}")
if [[ -n "${INIT_MODEL_CKPT}" ]]; then
  EXTRA_INIT_ARGS+=(--init_model_ckpt "${INIT_MODEL_CKPT}")
elif [[ -n "${INIT_CAPTION_CKPT}" ]]; then
  EXTRA_INIT_ARGS+=(--init_caption_ckpt "${INIT_CAPTION_CKPT}")
fi

nohup env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${TRAIN_PY}" \
  --dataset_type msvd \
  --model_type "${MODEL_TAG}" \
  --run_name "${RUN_NAME}" \
  --clip_global_vision_feats_path "${FEATS_PATH}" \
  --annotations_path "${ANNOTATIONS_PATH}" \
  --structured_gt_path "${STRUCTURED_GT_PATH}" \
  --out_dir "${RUN_ROOT}" \
  --seed "${SEED}" \
  --batch_size "${BATCH_SIZE}" \
  --accum_steps "${ACCUM_STEPS}" \
  --num_workers "${NUM_WORKERS}" \
  --max_train_steps_per_epoch "${MAX_TRAIN_STEPS_PER_EPOCH}" \
  --epochs "${EPOCHS}" \
  --scheduler cosine \
  --warmup_epochs 1 \
  --beam_size 5 \
  --beam_alpha 0.7 \
  --aux_visual_enable "${AUX_VISUAL_ENABLE}" \
  --aux_raw_global_enable "${AUX_RAW_GLOBAL_ENABLE}" \
  --aux_raw_global_feats_path "${AUX_RAW_GLOBAL_FEATS_PATH}" \
  --aux_patch_enable "${AUX_PATCH_ENABLE}" \
  --aux_patch_root "${AUX_PATCH_ROOT}" \
  --aux_patch_block "${AUX_PATCH_BLOCK}" \
  --aux_visual_raw_global_dim "${AUX_VISUAL_RAW_GLOBAL_DIM}" \
  --aux_visual_patch_dim "${AUX_VISUAL_PATCH_DIM}" \
  --aux_visual_prior_scale "${AUX_VISUAL_PRIOR_SCALE}" \
  --aux_visual_struct_scale "${AUX_VISUAL_STRUCT_SCALE}" \
  --aux_visual_memory_scale "${AUX_VISUAL_MEMORY_SCALE}" \
  --phrase_progress_enable "${PHRASE_PROGRESS_ENABLE}" \
  --phrase_progress_memory_scale "${PHRASE_PROGRESS_MEMORY_SCALE}" \
  --phrase_progress_source_scale "${PHRASE_PROGRESS_SOURCE_SCALE}" \
  --lr 2e-5 \
  --min_lr 1e-6 \
  --lambda_ce "${LAMBDA_CE}" \
  --lambda_entity "${LAMBDA_ENTITY}" \
  --lambda_action "${LAMBDA_ACTION}" \
  --lambda_attr "${LAMBDA_ATTR}" \
  --lambda_scene "${LAMBDA_SCENE}" \
  --prior_caption_pos_weight "${PRIOR_CAPTION_POS_WEIGHT}" \
  --prior_video_only_pos_weight "${PRIOR_VIDEO_ONLY_POS_WEIGHT}" \
  --prior_loss_type "${PRIOR_LOSS_TYPE}" \
  --prior_asl_gamma_neg "${PRIOR_ASL_GAMMA_NEG}" \
  --prior_asl_gamma_pos "${PRIOR_ASL_GAMMA_POS}" \
  --prior_asl_clip "${PRIOR_ASL_CLIP}" \
  --prior_asl_eps "${PRIOR_ASL_EPS}" \
  "${EXTRA_INIT_ARGS[@]}" \
  "${EXTRA_PRIOR_ARGS[@]}" \
  --lambda_phrase "${LAMBDA_PHRASE}" \
  --lambda_phrase_gen "${LAMBDA_PHRASE_GEN}" \
  --lambda_phrase_pred_gen "${LAMBDA_PHRASE_PRED_GEN}" \
  --phrase_pred_gen_start_epoch "${PHRASE_PRED_GEN_START_EPOCH}" \
  --prior_dropout 0.1 \
  --struct_condition 1 \
  --struct_condition_scale 0.25 \
  --struct_condition_query_bridge_enable "${STRUCT_CONDITION_QUERY_BRIDGE_ENABLE}" \
  --struct_condition_query_bridge_num_queries "${STRUCT_CONDITION_QUERY_BRIDGE_NUM_QUERIES}" \
  --struct_condition_query_bridge_scale "${STRUCT_CONDITION_QUERY_BRIDGE_SCALE}" \
  --struct_condition_query_bridge_memory_enable "${STRUCT_CONDITION_QUERY_BRIDGE_MEMORY_ENABLE}" \
  --struct_condition_query_bridge_memory_scale "${STRUCT_CONDITION_QUERY_BRIDGE_MEMORY_SCALE}" \
  --struct_condition_query_bridge_hidden_enable "${STRUCT_CONDITION_QUERY_BRIDGE_HIDDEN_ENABLE}" \
  --struct_condition_query_bridge_hidden_scale "${STRUCT_CONDITION_QUERY_BRIDGE_HIDDEN_SCALE}" \
  --phrase_decoder_enable "${PHRASE_DECODER_ENABLE}" \
  --phrase_condition_enable "${PHRASE_CONDITION_ENABLE}" \
  --phrase_condition_slot_aware_enable "${PHRASE_CONDITION_SLOT_AWARE_ENABLE}" \
  --phrase_condition_slot_selective_enable "${PHRASE_CONDITION_SLOT_SELECTIVE_ENABLE}" \
  --phrase_condition_slot_residual_enable "${PHRASE_CONDITION_SLOT_RESIDUAL_ENABLE}" \
  --phrase_condition_train_use_predicted "${PHRASE_CONDITION_TRAIN_USE_PREDICTED}" \
  --phrase_condition_pred_detach "${PHRASE_CONDITION_PRED_DETACH}" \
  --phrase_condition_pred_detach_until_epoch "${PHRASE_CONDITION_PRED_DETACH_UNTIL_EPOCH}" \
  --phrase_condition_teacher_source "${PHRASE_CONDITION_TEACHER_SOURCE}" \
  --phrase_decoder_layers 2 \
  --phrase_condition_scale 0.25 \
  --phrase_condition_aux_scale "${PHRASE_CONDITION_AUX_SCALE}" \
  --phrase_condition_slot_residual_scale "${PHRASE_CONDITION_SLOT_RESIDUAL_SCALE}" \
  --phrase_condition_family_bridge_enable "${PHRASE_CONDITION_FAMILY_BRIDGE_ENABLE}" \
  --phrase_condition_family_bridge_scale "${PHRASE_CONDITION_FAMILY_BRIDGE_SCALE}" \
  --phrase_condition_core_slot_types "${PHRASE_CONDITION_CORE_SLOT_TYPES}" \
  --phrase_condition_aux_slot_types "${PHRASE_CONDITION_AUX_SLOT_TYPES}" \
  --phrase_condition_query_bridge_enable "${PHRASE_CONDITION_QUERY_BRIDGE_ENABLE}" \
  --phrase_condition_query_bridge_num_queries "${PHRASE_CONDITION_QUERY_BRIDGE_NUM_QUERIES}" \
  --phrase_condition_query_bridge_scale "${PHRASE_CONDITION_QUERY_BRIDGE_SCALE}" \
  --phrase_condition_slot_residual_slot_types "${PHRASE_CONDITION_SLOT_RESIDUAL_SLOT_TYPES}" \
  --phrase_condition_candidate_bias_enable "${PHRASE_CONDITION_CANDIDATE_BIAS_ENABLE}" \
  --phrase_condition_candidate_bias_scale "${PHRASE_CONDITION_CANDIDATE_BIAS_SCALE}" \
  --phrase_condition_candidate_topk "${PHRASE_CONDITION_CANDIDATE_TOPK}" \
  --phrase_condition_candidate_slot_types "${PHRASE_CONDITION_CANDIDATE_SLOT_TYPES}" \
  --phrase_gen_max_len 32 \
  --phrase_memory_mode "${PHRASE_MEMORY_MODE}" \
  --phrase_target_mode "${PHRASE_TARGET_MODE}" \
  --phrase_slot_schema "${PHRASE_SLOT_SCHEMA}" \
  --phrase_include_attr_units "${PHRASE_INCLUDE_ATTR_UNITS}" \
  --phrase_include_scene_units "${PHRASE_INCLUDE_SCENE_UNITS}" \
  --phrase_attr_scene_units_start_epoch "${PHRASE_ATTR_SCENE_UNITS_START_EPOCH}" \
  --phrase_attr_scene_units_end_epoch "${PHRASE_ATTR_SCENE_UNITS_END_EPOCH}" \
  --phrase_include_video_phrase_units "${PHRASE_INCLUDE_VIDEO_PHRASE_UNITS}" \
  --phrase_include_video_attr_units "${PHRASE_INCLUDE_VIDEO_ATTR_UNITS}" \
  --phrase_include_video_scene_units "${PHRASE_INCLUDE_VIDEO_SCENE_UNITS}" \
  --phrase_video_phrase_min_support "${PHRASE_VIDEO_PHRASE_MIN_SUPPORT}" \
  --phrase_video_phrase_max_units "${PHRASE_VIDEO_PHRASE_MAX_UNITS}" \
  --max_phrase_slots "${MAX_PHRASE_SLOTS}" \
  --phrase_slot_max_len "${PHRASE_SLOT_MAX_LEN}" \
  --phrase_slot_planner_enable "${PHRASE_SLOT_PLANNER_ENABLE}" \
  --phrase_slot_planner_flow_enable "${PHRASE_SLOT_PLANNER_FLOW_ENABLE}" \
  --phrase_slot_planner_flow_scale "${PHRASE_SLOT_PLANNER_FLOW_SCALE}" \
  --phrase_slot_planner_flow_slot_types "${PHRASE_SLOT_PLANNER_FLOW_SLOT_TYPES}" \
  --phrase_slot_guidance_enable "${PHRASE_SLOT_GUIDANCE_ENABLE}" \
  --phrase_slot_role_anchor_enable "${PHRASE_SLOT_ROLE_ANCHOR_ENABLE}" \
  --phrase_slot_role_anchor_topk "${PHRASE_SLOT_ROLE_ANCHOR_TOPK}" \
  --phrase_slot_role_anchor_scale "${PHRASE_SLOT_ROLE_ANCHOR_SCALE}" \
  --phrase_slot_role_anchor_slot_types "${PHRASE_SLOT_ROLE_ANCHOR_SLOT_TYPES}" \
  --phrase_slot_decode_anchor_enable "${PHRASE_SLOT_DECODE_ANCHOR_ENABLE}" \
  --phrase_slot_decode_anchor_topk "${PHRASE_SLOT_DECODE_ANCHOR_TOPK}" \
  --phrase_slot_decode_anchor_scale "${PHRASE_SLOT_DECODE_ANCHOR_SCALE}" \
  --phrase_slot_decode_anchor_early_scale "${PHRASE_SLOT_DECODE_ANCHOR_EARLY_SCALE}" \
  --phrase_slot_decode_anchor_family_scale "${PHRASE_SLOT_DECODE_ANCHOR_FAMILY_SCALE}" \
  --phrase_slot_decode_anchor_family_topk "${PHRASE_SLOT_DECODE_ANCHOR_FAMILY_TOPK}" \
  --phrase_slot_decode_anchor_family_min_count "${PHRASE_SLOT_DECODE_ANCHOR_FAMILY_MIN_COUNT}" \
  --phrase_slot_decode_anchor_stopword_penalty "${PHRASE_SLOT_DECODE_ANCHOR_STOPWORD_PENALTY}" \
  --phrase_slot_decode_anchor_stopword_steps "${PHRASE_SLOT_DECODE_ANCHOR_STOPWORD_STEPS}" \
  --phrase_slot_decode_anchor_debug_topk "${PHRASE_SLOT_DECODE_ANCHOR_DEBUG_TOPK}" \
  --phrase_slot_presence_enable "${PHRASE_SLOT_PRESENCE_ENABLE}" \
  --phrase_slot_presence_support_enable "${PHRASE_SLOT_PRESENCE_SUPPORT_ENABLE}" \
  --phrase_slot_presence_evidence_enable "${PHRASE_SLOT_PRESENCE_EVIDENCE_ENABLE}" \
  --phrase_slot_presence_context_slot_types "${PHRASE_SLOT_PRESENCE_CONTEXT_SLOT_TYPES}" \
  --phrase_slot_presence_threshold "${PHRASE_SLOT_PRESENCE_THRESHOLD}" \
  --phrase_slot_active_slot_types "${PHRASE_SLOT_ACTIVE_SLOT_TYPES}" \
  --phrase_slot_presence_calibration_mode "${PHRASE_SLOT_PRESENCE_CALIBRATION_MODE}" \
  --phrase_slot_presence_calibration_slot_types "${PHRASE_SLOT_PRESENCE_CALIBRATION_SLOT_TYPES}" \
  --phrase_slot_presence_threshold_min "${PHRASE_SLOT_PRESENCE_THRESHOLD_MIN}" \
  --phrase_slot_presence_threshold_max "${PHRASE_SLOT_PRESENCE_THRESHOLD_MAX}" \
  --phrase_slot_reweight_enable "${PHRASE_SLOT_REWEIGHT_ENABLE}" \
  --phrase_slot_reweight_power "${PHRASE_SLOT_REWEIGHT_POWER}" \
  --phrase_slot_reweight_min "${PHRASE_SLOT_REWEIGHT_MIN}" \
  --phrase_slot_reweight_max "${PHRASE_SLOT_REWEIGHT_MAX}" \
  --phrase_slot_multiref_enable "${PHRASE_SLOT_MULTIREF_ENABLE}" \
  --phrase_slot_multiref_max_refs "${PHRASE_SLOT_MULTIREF_MAX_REFS}" \
  --phrase_slot_multiref_reduce "${PHRASE_SLOT_MULTIREF_REDUCE}" \
  --phrase_slot_multiref_gain "${PHRASE_SLOT_MULTIREF_GAIN}" \
  --phrase_slot_multiref_chunk_size "${PHRASE_SLOT_MULTIREF_CHUNK_SIZE}" \
  --phrase_slot_pred_multiref_reduce "${PHRASE_SLOT_PRED_MULTIREF_REDUCE}" \
  --phrase_slot_pred_multiref_softmin_temp "${PHRASE_SLOT_PRED_MULTIREF_SOFTMIN_TEMP}" \
  --phrase_slot_family_expand_mode "${PHRASE_SLOT_FAMILY_EXPAND_MODE}" \
  --phrase_slot_family_sample_mode "${PHRASE_SLOT_FAMILY_SAMPLE_MODE}" \
  --phrase_slot_family_sample_seed "${PHRASE_SLOT_FAMILY_SAMPLE_SEED}" \
  --prior_head_type "${PRIOR_HEAD_TYPE}" \
  --prior_head_num_heads "${PRIOR_HEAD_NUM_HEADS}" \
  --prior_head_hidden_dim "${PRIOR_HEAD_HIDDEN_DIM}" \
  --prior_head_num_blocks "${PRIOR_HEAD_NUM_BLOCKS}" \
  --prior_head_num_clusters "${PRIOR_HEAD_NUM_CLUSTERS}" \
  --prior_head_expansion "${PRIOR_HEAD_EXPANSION}" \
  --prior_head_groups "${PRIOR_HEAD_GROUPS}" \
  --phrase_max_len 77 \
  --phrase_fallback_to_caption 0 \
  --init_skip_structured_vocab_modules "${INIT_SKIP_STRUCTURED_VOCAB_MODULES}" \
  --lambda_phrase_slot_presence "${LAMBDA_PHRASE_SLOT_PRESENCE}" \
  --lambda_phrase_slot_div "${LAMBDA_PHRASE_SLOT_DIV}" \
  --lambda_phrase_ref_slot_align "${LAMBDA_PHRASE_REF_SLOT_ALIGN}" \
  --lambda_phrase_ref_bridge "${LAMBDA_PHRASE_REF_BRIDGE}" \
  --lambda_phrase_bridge "${LAMBDA_PHRASE_BRIDGE}" \
  --lambda_phrase_slot_source_align "${LAMBDA_PHRASE_SLOT_SOURCE_ALIGN}" \
  --lambda_phrase_slot_source_comp "${LAMBDA_PHRASE_SLOT_SOURCE_COMP}" \
  --phrase_slot_source_comp_margin "${PHRASE_SLOT_SOURCE_COMP_MARGIN}" \
  > "${LOG_PATH}" 2>&1 &

echo $! > "${PID_PATH}"

PID="$(cat "${PID_PATH}")"
BOUND_GPU_IDS=""
for _attempt in $(seq 1 "${GPU_BIND_MAX_POLLS}"); do
  if ! kill -0 "${PID}" 2>/dev/null; then
    echo "[GPU_GUARD] launched pid ${PID} exited before GPU binding verification." >&2
    exit 1
  fi
  BOUND_GPU_IDS="$(resolve_pid_gpu_indices "${PID}")"
  if [[ -n "${BOUND_GPU_IDS}" ]]; then
    break
  fi
  sleep "${GPU_BIND_POLL_SEC}"
done

GPU_SNAPSHOT_AFTER="$(capture_gpu_snapshot)"
echo "[GPU_GUARD][precheck-3] after_launch_gpus"
echo "$GPU_SNAPSHOT_AFTER"

if [[ -z "${BOUND_GPU_IDS}" ]]; then
  echo "[GPU_GUARD] failed to observe pid ${PID} on any GPU within verification window; killing it to avoid ambiguous placement." >&2
  kill "${PID}" 2>/dev/null || true
  exit 1
fi

if [[ "${BOUND_GPU_IDS}" != "${GPU_ID}" ]]; then
  echo "[GPU_GUARD] pid ${PID} bound to GPU(s) ${BOUND_GPU_IDS}, expected only ${GPU_ID}; killing it." >&2
  kill "${PID}" 2>/dev/null || true
  exit 1
fi

for forbidden_gpu in 0 1; do
  if [[ "${BOUND_GPU_IDS}" == *"${forbidden_gpu}"* ]]; then
    echo "[GPU_GUARD] pid ${PID} touched forbidden GPU${forbidden_gpu}; killing it." >&2
    kill "${PID}" 2>/dev/null || true
    exit 1
  fi
done

echo "variant=${VARIANT}"
echo "run_name=${RUN_NAME}"
echo "run_dir=${RUN_DIR}"
echo "log_path=${LOG_PATH}"
echo "pid=${PID}"
echo "gpu_guard_allowed=${ALLOWED_GPU_CSV}"
echo "gpu_guard_bound=${BOUND_GPU_IDS}"
echo "gpu_bind_max_polls=${GPU_BIND_MAX_POLLS}"
echo "gpu_bind_poll_sec=${GPU_BIND_POLL_SEC}"
echo "epochs=${EPOCHS}"
echo "seed=${SEED}"
echo "batch_size=${BATCH_SIZE}"
echo "accum_steps=${ACCUM_STEPS}"
echo "num_workers=${NUM_WORKERS}"
echo "max_train_steps_per_epoch=${MAX_TRAIN_STEPS_PER_EPOCH}"
echo "structured_gt_path=${STRUCTURED_GT_PATH}"
echo "phrase_memory_mode=${PHRASE_MEMORY_MODE}"
echo "phrase_target_mode=${PHRASE_TARGET_MODE}"
echo "phrase_slot_schema=${PHRASE_SLOT_SCHEMA}"
echo "phrase_include_attr_units=${PHRASE_INCLUDE_ATTR_UNITS}"
echo "phrase_include_scene_units=${PHRASE_INCLUDE_SCENE_UNITS}"
echo "phrase_attr_scene_units_start_epoch=${PHRASE_ATTR_SCENE_UNITS_START_EPOCH}"
echo "phrase_attr_scene_units_end_epoch=${PHRASE_ATTR_SCENE_UNITS_END_EPOCH}"
echo "phrase_include_video_phrase_units=${PHRASE_INCLUDE_VIDEO_PHRASE_UNITS}"
echo "phrase_include_video_attr_units=${PHRASE_INCLUDE_VIDEO_ATTR_UNITS}"
echo "phrase_include_video_scene_units=${PHRASE_INCLUDE_VIDEO_SCENE_UNITS}"
echo "phrase_video_phrase_min_support=${PHRASE_VIDEO_PHRASE_MIN_SUPPORT}"
echo "phrase_video_phrase_max_units=${PHRASE_VIDEO_PHRASE_MAX_UNITS}"
echo "max_phrase_slots=${MAX_PHRASE_SLOTS}"
echo "phrase_slot_max_len=${PHRASE_SLOT_MAX_LEN}"
echo "phrase_slot_planner_enable=${PHRASE_SLOT_PLANNER_ENABLE}"
echo "phrase_slot_planner_flow_enable=${PHRASE_SLOT_PLANNER_FLOW_ENABLE}"
echo "phrase_slot_planner_flow_scale=${PHRASE_SLOT_PLANNER_FLOW_SCALE}"
echo "phrase_slot_planner_flow_slot_types=${PHRASE_SLOT_PLANNER_FLOW_SLOT_TYPES}"
echo "phrase_slot_guidance_enable=${PHRASE_SLOT_GUIDANCE_ENABLE}"
echo "phrase_slot_role_anchor_enable=${PHRASE_SLOT_ROLE_ANCHOR_ENABLE}"
echo "phrase_slot_role_anchor_topk=${PHRASE_SLOT_ROLE_ANCHOR_TOPK}"
echo "phrase_slot_role_anchor_scale=${PHRASE_SLOT_ROLE_ANCHOR_SCALE}"
echo "phrase_slot_role_anchor_slot_types=${PHRASE_SLOT_ROLE_ANCHOR_SLOT_TYPES}"
echo "phrase_slot_decode_anchor_enable=${PHRASE_SLOT_DECODE_ANCHOR_ENABLE}"
echo "phrase_slot_decode_anchor_topk=${PHRASE_SLOT_DECODE_ANCHOR_TOPK}"
echo "phrase_slot_decode_anchor_scale=${PHRASE_SLOT_DECODE_ANCHOR_SCALE}"
echo "phrase_slot_decode_anchor_early_scale=${PHRASE_SLOT_DECODE_ANCHOR_EARLY_SCALE}"
echo "phrase_slot_decode_anchor_family_scale=${PHRASE_SLOT_DECODE_ANCHOR_FAMILY_SCALE}"
echo "phrase_slot_decode_anchor_family_topk=${PHRASE_SLOT_DECODE_ANCHOR_FAMILY_TOPK}"
echo "phrase_slot_decode_anchor_family_min_count=${PHRASE_SLOT_DECODE_ANCHOR_FAMILY_MIN_COUNT}"
echo "phrase_slot_decode_anchor_stopword_penalty=${PHRASE_SLOT_DECODE_ANCHOR_STOPWORD_PENALTY}"
echo "phrase_slot_decode_anchor_stopword_steps=${PHRASE_SLOT_DECODE_ANCHOR_STOPWORD_STEPS}"
echo "phrase_slot_decode_anchor_debug_topk=${PHRASE_SLOT_DECODE_ANCHOR_DEBUG_TOPK}"
echo "phrase_slot_presence_enable=${PHRASE_SLOT_PRESENCE_ENABLE}"
echo "phrase_slot_presence_support_enable=${PHRASE_SLOT_PRESENCE_SUPPORT_ENABLE}"
echo "phrase_slot_presence_evidence_enable=${PHRASE_SLOT_PRESENCE_EVIDENCE_ENABLE}"
echo "phrase_slot_presence_context_slot_types=${PHRASE_SLOT_PRESENCE_CONTEXT_SLOT_TYPES}"
echo "phrase_slot_presence_threshold=${PHRASE_SLOT_PRESENCE_THRESHOLD}"
echo "phrase_slot_active_slot_types=${PHRASE_SLOT_ACTIVE_SLOT_TYPES}"
echo "phrase_slot_presence_calibration_mode=${PHRASE_SLOT_PRESENCE_CALIBRATION_MODE}"
echo "phrase_slot_presence_calibration_slot_types=${PHRASE_SLOT_PRESENCE_CALIBRATION_SLOT_TYPES}"
echo "phrase_slot_presence_threshold_min=${PHRASE_SLOT_PRESENCE_THRESHOLD_MIN}"
echo "phrase_slot_presence_threshold_max=${PHRASE_SLOT_PRESENCE_THRESHOLD_MAX}"
echo "phrase_slot_reweight_enable=${PHRASE_SLOT_REWEIGHT_ENABLE}"
echo "phrase_slot_reweight_power=${PHRASE_SLOT_REWEIGHT_POWER}"
echo "phrase_slot_reweight_min=${PHRASE_SLOT_REWEIGHT_MIN}"
echo "phrase_slot_reweight_max=${PHRASE_SLOT_REWEIGHT_MAX}"
echo "phrase_slot_multiref_enable=${PHRASE_SLOT_MULTIREF_ENABLE}"
echo "phrase_slot_multiref_max_refs=${PHRASE_SLOT_MULTIREF_MAX_REFS}"
echo "phrase_slot_multiref_reduce=${PHRASE_SLOT_MULTIREF_REDUCE}"
echo "phrase_slot_multiref_gain=${PHRASE_SLOT_MULTIREF_GAIN}"
echo "phrase_slot_multiref_chunk_size=${PHRASE_SLOT_MULTIREF_CHUNK_SIZE}"
echo "phrase_slot_pred_multiref_reduce=${PHRASE_SLOT_PRED_MULTIREF_REDUCE}"
echo "phrase_slot_pred_multiref_softmin_temp=${PHRASE_SLOT_PRED_MULTIREF_SOFTMIN_TEMP}"
echo "phrase_slot_family_expand_mode=${PHRASE_SLOT_FAMILY_EXPAND_MODE}"
echo "phrase_slot_family_sample_mode=${PHRASE_SLOT_FAMILY_SAMPLE_MODE}"
echo "phrase_slot_family_sample_seed=${PHRASE_SLOT_FAMILY_SAMPLE_SEED}"
echo "prior_head_type=${PRIOR_HEAD_TYPE}"
echo "prior_head_num_heads=${PRIOR_HEAD_NUM_HEADS}"
echo "prior_head_hidden_dim=${PRIOR_HEAD_HIDDEN_DIM}"
echo "prior_head_num_blocks=${PRIOR_HEAD_NUM_BLOCKS}"
echo "init_skip_structured_vocab_modules=${INIT_SKIP_STRUCTURED_VOCAB_MODULES}"
echo "phrase_condition_slot_aware_enable=${PHRASE_CONDITION_SLOT_AWARE_ENABLE}"
echo "phrase_condition_slot_selective_enable=${PHRASE_CONDITION_SLOT_SELECTIVE_ENABLE}"
echo "phrase_condition_aux_scale=${PHRASE_CONDITION_AUX_SCALE}"
echo "phrase_condition_slot_residual_enable=${PHRASE_CONDITION_SLOT_RESIDUAL_ENABLE}"
echo "phrase_condition_slot_residual_scale=${PHRASE_CONDITION_SLOT_RESIDUAL_SCALE}"
echo "phrase_condition_slot_residual_slot_types=${PHRASE_CONDITION_SLOT_RESIDUAL_SLOT_TYPES}"
echo "phrase_condition_family_bridge_enable=${PHRASE_CONDITION_FAMILY_BRIDGE_ENABLE}"
echo "phrase_condition_family_bridge_scale=${PHRASE_CONDITION_FAMILY_BRIDGE_SCALE}"
echo "phrase_condition_candidate_bias_enable=${PHRASE_CONDITION_CANDIDATE_BIAS_ENABLE}"
echo "phrase_condition_candidate_bias_scale=${PHRASE_CONDITION_CANDIDATE_BIAS_SCALE}"
echo "phrase_condition_candidate_topk=${PHRASE_CONDITION_CANDIDATE_TOPK}"
echo "phrase_condition_candidate_slot_types=${PHRASE_CONDITION_CANDIDATE_SLOT_TYPES}"
echo "phrase_condition_core_slot_types=${PHRASE_CONDITION_CORE_SLOT_TYPES}"
echo "phrase_condition_aux_slot_types=${PHRASE_CONDITION_AUX_SLOT_TYPES}"
echo "phrase_condition_train_use_predicted=${PHRASE_CONDITION_TRAIN_USE_PREDICTED}"
echo "phrase_condition_pred_detach=${PHRASE_CONDITION_PRED_DETACH}"
echo "phrase_condition_pred_detach_until_epoch=${PHRASE_CONDITION_PRED_DETACH_UNTIL_EPOCH}"
echo "phrase_condition_teacher_source=${PHRASE_CONDITION_TEACHER_SOURCE}"
echo "training_stage=${TRAINING_STAGE}"
if [[ ${#INIT_PHRASE_INTERFACE_INHERITED[@]} -gt 0 ]]; then
  echo "init_phrase_interface_inherited=${INIT_PHRASE_INTERFACE_INHERITED[*]}"
else
  echo "init_phrase_interface_inherited="
fi
echo "struct_condition_query_bridge_enable=${STRUCT_CONDITION_QUERY_BRIDGE_ENABLE}"
echo "struct_condition_query_bridge_num_queries=${STRUCT_CONDITION_QUERY_BRIDGE_NUM_QUERIES}"
echo "struct_condition_query_bridge_scale=${STRUCT_CONDITION_QUERY_BRIDGE_SCALE}"
echo "struct_condition_query_bridge_memory_enable=${STRUCT_CONDITION_QUERY_BRIDGE_MEMORY_ENABLE}"
echo "struct_condition_query_bridge_memory_scale=${STRUCT_CONDITION_QUERY_BRIDGE_MEMORY_SCALE}"
echo "struct_condition_query_bridge_hidden_enable=${STRUCT_CONDITION_QUERY_BRIDGE_HIDDEN_ENABLE}"
echo "struct_condition_query_bridge_hidden_scale=${STRUCT_CONDITION_QUERY_BRIDGE_HIDDEN_SCALE}"
echo "phrase_condition_query_bridge_enable=${PHRASE_CONDITION_QUERY_BRIDGE_ENABLE}"
echo "phrase_condition_query_bridge_num_queries=${PHRASE_CONDITION_QUERY_BRIDGE_NUM_QUERIES}"
echo "phrase_condition_query_bridge_scale=${PHRASE_CONDITION_QUERY_BRIDGE_SCALE}"
echo "lambda_phrase_slot_presence=${LAMBDA_PHRASE_SLOT_PRESENCE}"
echo "lambda_phrase_slot_div=${LAMBDA_PHRASE_SLOT_DIV}"
echo "lambda_phrase_ref_slot_align=${LAMBDA_PHRASE_REF_SLOT_ALIGN}"
echo "lambda_phrase_ref_bridge=${LAMBDA_PHRASE_REF_BRIDGE}"
echo "lambda_phrase_bridge=${LAMBDA_PHRASE_BRIDGE}"
echo "lambda_phrase_pred_gen=${LAMBDA_PHRASE_PRED_GEN}"
echo "phrase_pred_gen_start_epoch=${PHRASE_PRED_GEN_START_EPOCH}"
echo "lambda_phrase_slot_source_align=${LAMBDA_PHRASE_SLOT_SOURCE_ALIGN}"
echo "lambda_phrase_slot_source_comp=${LAMBDA_PHRASE_SLOT_SOURCE_COMP}"
echo "phrase_slot_source_comp_margin=${PHRASE_SLOT_SOURCE_COMP_MARGIN}"
echo "lambda_attr=${LAMBDA_ATTR}"
echo "lambda_scene=${LAMBDA_SCENE}"
echo "prior_caption_pos_weight=${PRIOR_CAPTION_POS_WEIGHT}"
echo "prior_video_only_pos_weight=${PRIOR_VIDEO_ONLY_POS_WEIGHT}"
echo "prior_loss_type=${PRIOR_LOSS_TYPE}"
echo "prior_asl_gamma_neg=${PRIOR_ASL_GAMMA_NEG}"
echo "prior_asl_gamma_pos=${PRIOR_ASL_GAMMA_POS}"
echo "prior_asl_clip=${PRIOR_ASL_CLIP}"
echo "prior_asl_eps=${PRIOR_ASL_EPS}"
echo "entity_prior_caption_pos_weight=${ENTITY_PRIOR_CAPTION_POS_WEIGHT}"
echo "entity_prior_video_only_pos_weight=${ENTITY_PRIOR_VIDEO_ONLY_POS_WEIGHT}"
echo "action_prior_caption_pos_weight=${ACTION_PRIOR_CAPTION_POS_WEIGHT}"
echo "action_prior_video_only_pos_weight=${ACTION_PRIOR_VIDEO_ONLY_POS_WEIGHT}"
echo "attr_prior_caption_pos_weight=${ATTR_PRIOR_CAPTION_POS_WEIGHT}"
echo "attr_prior_video_only_pos_weight=${ATTR_PRIOR_VIDEO_ONLY_POS_WEIGHT}"
echo "scene_prior_caption_pos_weight=${SCENE_PRIOR_CAPTION_POS_WEIGHT}"
echo "scene_prior_video_only_pos_weight=${SCENE_PRIOR_VIDEO_ONLY_POS_WEIGHT}"
echo "prior_head_num_clusters=${PRIOR_HEAD_NUM_CLUSTERS}"
echo "prior_head_expansion=${PRIOR_HEAD_EXPANSION}"
echo "prior_head_groups=${PRIOR_HEAD_GROUPS}"
echo "aux_visual_enable=${AUX_VISUAL_ENABLE}"
echo "aux_raw_global_enable=${AUX_RAW_GLOBAL_ENABLE}"
echo "aux_raw_global_feats_path=${AUX_RAW_GLOBAL_FEATS_PATH}"
echo "aux_patch_enable=${AUX_PATCH_ENABLE}"
echo "aux_patch_root=${AUX_PATCH_ROOT}"
echo "aux_patch_block=${AUX_PATCH_BLOCK}"
echo "aux_visual_raw_global_dim=${AUX_VISUAL_RAW_GLOBAL_DIM}"
echo "aux_visual_patch_dim=${AUX_VISUAL_PATCH_DIM}"
echo "aux_visual_prior_scale=${AUX_VISUAL_PRIOR_SCALE}"
echo "aux_visual_struct_scale=${AUX_VISUAL_STRUCT_SCALE}"
echo "aux_visual_memory_scale=${AUX_VISUAL_MEMORY_SCALE}"
echo "phrase_progress_enable=${PHRASE_PROGRESS_ENABLE}"
echo "phrase_progress_memory_scale=${PHRASE_PROGRESS_MEMORY_SCALE}"
echo "phrase_progress_source_scale=${PHRASE_PROGRESS_SOURCE_SCALE}"
