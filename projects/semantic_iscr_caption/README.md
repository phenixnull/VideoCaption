# semantic_iscr_caption

Clean extraction of the MSVD `Structured Decoder v2 + ISCR rerank` line from `projects/structed_caption`.

This repo keeps only the code and assets needed for the no-leak semantic ISCR caption pipeline:

- training: `train_structured_refine_monitored.py`
- evaluation + rerank: `eval_structured_iscr_rerank.py`
- model: `models_structured.py`, `models.py`, `structured_prior_heads.py`
- dataset loading: `dataloaders/`
- tokenizer / CLIP init assets: `models/clip_tokenizer`, `models/clip_models/ViT-B-32.pt`

## Selected experiment

Anchor training run:

- run id: `msvd_structured_api_structv2_e8_a_20260303_010342`
- best checkpoint: `epoch_003.pt`
- training-best metrics: `CIDEr 121.5153`, `BLEU-4 63.3693`, `METEOR 41.6117`, `ROUGE-L 78.0057`

Best ISCR rerank result on that checkpoint:

- decode: `beam=4`, `beam_alpha=0.7`
- rerank: `cov=0.9`, `hall=0.15`, `prior_topk=64`, `rerank_topk=20`
- metrics: `CIDEr 122.2919`, `BLEU-4 62.8434`, `METEOR 41.7040`, `ROUGE-L 78.2536`

## Layout

- `configs/`: extracted reference hyperparameters
- `docs/`: model notes and tensor shape trace
- `scripts/`: reproducible train/eval launchers
- `weights/`: intended location for extracted checkpoints in this clean repo

## Expected data inputs

- MSVD visual features: `datasets/MSVD/feats/clip4clip_vitb32_k_split_ks12_features.pickle`
- MSVD captions: `datasets/MSVD/annotations_preprocessed.txt`
- structured train annotations: `annotations/msvd_structured_train_api.json`

## Notes

- This repo is intentionally scoped to the MSVD semantic ISCR line. It does not carry the phrase-mainline or unrelated ablation families.
- The bundled CLIP weight is used only for embedding initialization; the video input is pre-extracted feature tensors.

