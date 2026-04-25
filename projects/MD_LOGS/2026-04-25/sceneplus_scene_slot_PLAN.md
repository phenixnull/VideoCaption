# Sceneplus Scene Slot Experiment Plan

## Selected Idea
Add one conservative scene/context phrase slot to the existing `structured_phrase_d + typed_rich_roleaware` slot phrase decoder by introducing `typed_rich_roleaware_sceneplus`. The original 9 role-aware slots stay ordered the same; extra slots repeat `scene_context` first and are supervised only by caption-local scene units with video-level scene union disabled.

## User Requirements
- Preserve non-collapsed intermediate phrase interpretability.
- Improve story-friendly middle-stage output with scene/context detail, not only subject/object/verb.
- Target metrics: CIDEr around 120, BLEU-4 63+, ROUGE-L 78+, METEOR 41-42+.
- Keep the no-leak boundary: no validation/test text or inference-time target text.

## Baseline Contract
- Reference run: `msvd_phrase_d_20260320_112700_softcandprogressctrl_3e`.
- Reference metrics: CIDEr 122.8176, BLEU-4 64.5279, METEOR 42.0146, ROUGE-L 78.5895.
- Dataset/split/evaluator stay MSVD test with the existing training and evaluation scripts.

## Literature-Driven Rationale
- Semantic attribute/concept injection is useful when treated as high-level guidance, not as a replacement caption.
- Scene graph and meta-concept video captioning work motivates adding contextual semantic tokens for objects, predicates, and scene context.
- Gating/selective conditioning is preferred so auxiliary concepts can improve interpretability without dominating final sentence decoding.

## Code Change Map
- Active branch directory: `projects/structed_caption_sceneplus`.
- Original directory `projects/structed_caption` is kept clean and should not be edited, deleted, overwritten, or used as an output root for this line.
- `projects/structed_caption_sceneplus/dataloaders/dataset_structured_caption.py`: add `typed_rich_roleaware_sceneplus` schema and extra-slot priority with repeated `scene_context`.
- `projects/structed_caption_sceneplus/train_structured_refine_monitored.py`: allow the new schema in CLI choices.
- `projects/structed_caption_sceneplus/run_structured_phrase_sceneplus_scene_slots.sh`: launcher for a 3-epoch sceneplus gate.
- `projects/structed_caption_sceneplus/run_structured_phrase_mainline.sh`: resolve `SERVER_ROOT` from the sceneplus branch directory so runs write under the new branch.

## Run Contract
- `run_id`: `sceneplus_s10_sceneunits_3e`.
- Type: auxiliary/dev gate that can be promoted if metrics and phrase exports pass.
- Changed factors: `phrase_slot_schema=typed_rich_roleaware_sceneplus`, `max_phrase_slots=10`, `phrase_include_scene_units=1`, `phrase_include_video_scene_units=0`, selective scene auxiliary conditioning.
- Safety gate: keep final caption conditioning core on `subject_action,subject_entity,object_entity,object_passive`; scene slots enter as low-scale aux context and phrase/presence supervision.
- Stop condition: run fails, metrics below CIDEr 120 or BLEU-4 63 or ROUGE-L 78 or METEOR 41, or exported phrase slots collapse into empty/repeated generic text.

## Expected Outputs
- Server branch directory: `/mnt/sda/Disk_D/zhangwei/projects/VC/project/structured_caption_sceneplus`.
- Server run directory under `/mnt/sda/Disk_D/zhangwei/projects/VC/project/structured_caption_sceneplus/runs/phrase_mainline/`.
- `args.json`, `train.log`, checkpoints, `test_metrics.jsonl`.
- Phrase export JSONL containing active `scene_context` slots.

## Revision Log
- 2026-04-25: Initial plan after code audit and web scouting.
- 2026-04-25: Added compact scenehint and decoupled active-scene variants. Selected `msvd_phrase_d_20260425_121903_sceneplus_compact_scenehint_decoupled_active_s10_1e` as best sceneplus run: CIDEr 120.3973, BLEU-4 63.2704, METEOR 41.8541, ROUGE-L 78.3260, with clean scene slots in 99/670 test records. A follow-up 3-epoch diagnostic (`msvd_phrase_d_20260425_124916_sceneplus_compact_scenehint_decoupled_active_s10_3e`) did not improve; its best training-test row was epoch 2: CIDEr 119.7485, BLEU-4 62.8426, METEOR 41.3746, ROUGE-L 77.6984.
