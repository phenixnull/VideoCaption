# Data And Shape Trace

## Base visual input

Source: `dataloaders/dataset_msvd_feats.py`

- feature pickle item: `(feat, vid_mask)`
- `feat`: `[T, D]`, expected MSVD CLIP feature tensor, typically `[12, 512]`
- `vid_mask`: `[T]`

Returned base sample:

- `vid_feat_tensor`: `[T, D]`
- `vid_mask_tensor`: `[T]`
- `caption_ids`: `[77]`
- `caption_mask`: `[77]`

## Structured wrapper output

Source: `dataloaders/dataset_structured_caption.py::__getitem__`

Returned tuple extends the base sample with:

- `entity_target`: `[entity_dim]`
- `action_target`: `[action_dim]`
- `attr_target`: `[attribute_dim]`
- `scene_target`: `[scene_dim]`
- `caption_entity_target`: `[entity_dim]`
- `caption_action_target`: `[action_dim]`
- `caption_attr_target`: `[attribute_dim]`
- `caption_scene_target`: `[scene_dim]`
- `attr_known_mask`: `[]`
- `scene_known_mask`: `[]`
- `phrase_ids`: `[phrase_max_len]`
- `phrase_mask`: `[phrase_max_len]`
- `phrase_slot_ids`: `[max_phrase_slots, phrase_slot_max_len]`
- `phrase_slot_mask`: `[max_phrase_slots, phrase_slot_max_len]`
- `phrase_slot_valid`: `[max_phrase_slots]`

Optional multiref payload:

- `phrase_slot_ref_ids`: `[max_phrase_slots, max_refs, phrase_slot_max_len]`
- `phrase_slot_ref_mask`: `[max_phrase_slots, max_refs, phrase_slot_max_len]`
- `phrase_slot_ref_valid`: `[max_phrase_slots, max_refs]`

## Model input / output

Backbone source: `models.py::CaptionModel_Base.forward`

- `video_feats`: `[B, T, D]`
- `vid_mask`: `[B, T]`
- `captions`: `[B, 77]`
- `caption_mask`: `[B, 77]`
- backbone logits: `[B, 76, vocab_size]`

Structured model source: `models_structured.py::StructuredCaptionModel.forward`

Core outputs in `aux` include:

- `entity_logits`: `[B, entity_dim]`
- `action_logits`: `[B, action_dim]`
- optional `attribute_logits`: `[B, attribute_dim]`
- optional `scene_logits`: `[B, scene_dim]`
- optional phrase decoder outputs:
  - `phrase_decoder_logits`
  - `phrase_slot_logits`
  - `phrase_slot_presence_logits`

## Eval-time beam candidates

Source: `eval_structured_iscr_rerank.py::beam_search_candidates_batch`

- per sample beam list: `List[(token_ids, normalized_logprob)]`
- rerank score:
  - `alpha * base_score + lambda_cov * coverage - lambda_hall * hallucination`

