# structed_caption_sceneplus

Clean scene-slot branch for the MSVD `structured_phrase_d` caption line.

This branch keeps the compact semantic vocabulary path and adds scene/context phrase slots without letting scene phrases condition the final caption decoder.

## Main Experiment

- launcher: `run_structured_phrase_sceneplus_compact_decoupled_active.sh`
- structured GT builder: `build_compact_scenehint_gt.py`
- schema: `typed_rich_roleaware_sceneplus`
- slots: 10 total, with an extra `scene_context` repeat slot
- scene conditioning: decoupled from caption candidate/residual conditioning
- best observed MSVD test metrics: `CIDEr 120.3973`, `BLEU-4 63.2704`, `METEOR 41.8541`, `ROUGE-L 78.3260`

## Reproduce

Build compact scenehint annotations:

```bash
python build_compact_scenehint_gt.py \
  --compact_gt /mnt/sda/Disk_D/zhangwei/projects/VC/project/structured_caption/annotations/msvd_structured_train_api.json \
  --rich_gt /mnt/sda/Disk_D/zhangwei/projects/VC/project/structured_caption/annotations/msvd_structured_phrasegram_v2_20260405_200801_typedrichroleaware_all.json \
  --output_path annotations/msvd_structured_train_api_scenehint.json \
  --max_scene_units 3
```

Train/evaluate the active scene-slot gate:

```bash
ALLOWED_GPU_CSV=2 GPU_BIND_MAX_POLLS=120 GPU_BIND_POLL_SEC=2 \
  bash run_structured_phrase_sceneplus_compact_decoupled_active.sh 2
```

## Layout

- `configs/`: extracted reference hyperparameters
- `dataloaders/`: structured caption dataset and slot target construction
- `docs/`: model notes
- `scripts/`: compact train/eval helpers
- `weights/`: intended location for extracted checkpoints in this clean repo

## Expected data inputs

- MSVD visual features: `datasets/MSVD/feats/clip4clip_vitb32_k_split_ks12_features.pickle`
- MSVD captions: `datasets/MSVD/annotations_preprocessed.txt`
- compact structured annotations: `annotations/msvd_structured_train_api.json`
- generated scenehint annotations: `annotations/msvd_structured_train_api_scenehint.json`

## Notes

- This repo is intentionally scoped to the scene-slot experiment and does not carry server `runs/` outputs.
- The bundled CLIP weight is used only for embedding initialization; the video input is pre-extracted feature tensors.

