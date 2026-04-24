# ActivityNet Captions Dataset Notes

Date prepared: 2026-04-24

## What Was Downloaded

Source: [official DenseVid ActivityNet Captions page](https://cs.stanford.edu/people/ranjaykrishna/densevid/).

```text
ActivityNet/
  annotations/
    captions.zip
    readme.txt
    train.json        # 10,009 videos, 37,421 captioned segments
    val_1.json        # 4,917 videos, 17,505 captioned segments
    val_2.json        # 4,885 videos, 17,031 captioned segments
    train_ids.json
    val_ids.json
    test_ids.json     # ids only; public test captions are not included
  features/           # empty; put C3D/TSN/TSP/CLIP feature pickles here later
  raw/                # empty; put videos here later
  samples/            # smoke test creates tiny_activitynet_features.pickle
  scripts/
    dataset_activitynet_captions.py
    download_activitynet_annotations.ps1
    smoke_test_activitynet.py
```

The official JSON schema is:

```json
{
  "v_QOlSCBRmfWY": {
    "duration": 82.73,
    "timestamps": [[0.83, 19.86], [17.37, 60.81]],
    "sentences": ["A young woman ...", "The girl dances ..."]
  }
}
```

## Split Convention

The real ActivityNet test captions are not public.  Open-source captioning
repositories commonly evaluate with both validation annotation files.  This
loader uses:

| Loader split | File | Meaning |
| --- | --- | --- |
| `train` | `annotations/train.json` | training captions |
| `val` / `val_1` | `annotations/val_1.json` | validation reference set 1 |
| `test` / `val_2` | `annotations/val_2.json` | public test proxy / validation reference set 2 |

`annotations/test_ids.json` is retained for completeness, but it has ids only
and is not used by the caption loader.

## Loader Behavior

`scripts/dataset_activitynet_captions.py` exposes `ActivityNetCaptionsDataset`.

Default `sample_mode="segments"` returns one item per timestamp/caption pair.
`sample_mode="video"` returns one item per video while preserving all timestamps
and sentences.

Returned items are dictionaries with:

```text
video_id, duration, start, end, timestamp, caption, segment_index,
timestamps, sentences, feature, feature_mask, raw_video_path
```

Features are optional.  If supplied, `features_path` should point to a pickle
with either:

```python
{video_id: (features, mask)}
{video_id: {"features": features, "mask": mask}}
```

Both `v_xxx` and `xxx` feature keys are accepted.

## Smoke Test

The smoke test samples train/val/test segments, creates
`samples/tiny_activitynet_features.pickle`, then verifies that PyTorch
`DataLoader` can stack `(B, 16, 512)` feature tensors.

```powershell
& "D:\Users\Administrator\anaconda3\envs\vc_local\python.exe" datasets\ActivityNet\scripts\smoke_test_activitynet.py
```

Run again without rewriting the tiny fixture:

```powershell
& "D:\Users\Administrator\anaconda3\envs\vc_local\python.exe" datasets\ActivityNet\scripts\smoke_test_activitynet.py --skip-create-features
```

## Frame Extraction

No frames were extracted in this setup pass.  `ActivityNet/raw/` is empty.
When videos or pre-extracted features are available, put full-video features in:

```text
ActivityNet/features/ViT-B-32_k_split_ks16_features.pickle
```

with pickle values:

```python
{
    "v_QOlSCBRmfWY": (features[16, 512], mask[16]),
}
```

For dense-captioning models, you may instead keep repository-specific feature
folders such as C3D/TSN/TSP.  The loader still returns timestamps, so segment
cropping can be added downstream without changing annotation parsing.

## Reference Summary

- BMT converts ActivityNet Captions into TSV rows:
  `video_id, caption, start, end, duration, phase, idx`, then loads I3D/VGGish
  features for each segment.
- PDVC keeps the official JSON shape and reads `timestamps` plus `sentences`
  directly, with C3D/TSN/TSP feature folders configured separately.
- This project adapter keeps the official JSON shape, returns segment metadata,
  and accepts the local baseline pickle format so it can be used before choosing
  a final feature backend.

