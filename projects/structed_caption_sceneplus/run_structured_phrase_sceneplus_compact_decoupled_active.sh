#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-${CUDA_VISIBLE_DEVICES:-2}}"
TS_RAW="${2:-$(date +%Y%m%d_%H%M%S)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-sceneplus_compact_scenehint_decoupled_active_s10_1e}"
export PHRASE_SLOT_PRESENCE_THRESHOLD="${PHRASE_SLOT_PRESENCE_THRESHOLD:-0.35}"
export PHRASE_SLOT_ACTIVE_SLOT_TYPES="${PHRASE_SLOT_ACTIVE_SLOT_TYPES:-subject_action,object_passive,subject_entity,object_entity,subject_modifier,object_modifier,relation_detail,instrument_detail,scene_context}"

bash "${SCRIPT_DIR}/run_structured_phrase_sceneplus_compact_decoupled.sh" "${GPU_ID}" "${TS_RAW}"
