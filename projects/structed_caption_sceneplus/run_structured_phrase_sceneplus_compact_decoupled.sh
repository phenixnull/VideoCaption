#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-${CUDA_VISIBLE_DEVICES:-2}}"
TS_RAW="${2:-$(date +%Y%m%d_%H%M%S)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EPOCHS="${EPOCHS:-1}"
export RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-sceneplus_compact_scenehint_decoupled_s10_1e}"

export PHRASE_CONDITION_SLOT_RESIDUAL_SLOT_TYPES="${PHRASE_CONDITION_SLOT_RESIDUAL_SLOT_TYPES:-subject_entity,subject_action,object_entity,object_passive,subject_modifier,object_modifier,relation_detail}"
export PHRASE_CONDITION_CANDIDATE_SLOT_TYPES="${PHRASE_CONDITION_CANDIDATE_SLOT_TYPES:-subject_entity,object_entity,subject_modifier,object_modifier}"

bash "${SCRIPT_DIR}/run_structured_phrase_sceneplus_compact_scenehint.sh" "${GPU_ID}" "${TS_RAW}"
