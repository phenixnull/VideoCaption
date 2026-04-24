# Model Inventory

This clean repo keeps one experiment family only: `Structured Decoder v2 + ISCR rerank`.

## Files kept

- `train_structured_refine_monitored.py`
  - training entrypoint for the structured decoder anchor checkpoint
- `eval_structured_iscr_rerank.py`
  - beam decode + ISCR rerank evaluation
- `models_structured.py`
  - `StructuredCaptionModel`
- `models.py`
  - `CaptionModel_Base` and decoder backbone
- `structured_prior_heads.py`
  - structured prior prediction heads
- `phrase_lexical_anchors.py`
  - lexical anchor helper used by eval / training utilities
- `dataloaders/dataset_msvd_feats.py`
  - base MSVD feature dataset
- `dataloaders/dataset_structured_caption.py`
  - structured supervision wrapper
- `dataloaders/dataset_visual_evidence_wrapper.py`
  - optional evidence wrapper retained because the training helpers reference it

## Files intentionally excluded

- phrase-mainline launchers
- analysis rendering scripts
- gemini stage generation pipelines
- report-only utilities
- unrelated ablation runners

## Reference metrics

- anchor checkpoint best epoch:
  - `CIDEr 121.5153`
  - `BLEU-4 63.3693`
- best rerank:
  - `CIDEr 122.2919`
  - `BLEU-4 62.8434`

