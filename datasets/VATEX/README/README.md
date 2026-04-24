# VATEX Dataset Notes

Date prepared: 2026-04-24

## What Was Downloaded

Source: [official VATEX download page](https://eric-xw.github.io/vatex-website/download.html).

```text
VATEX/
  annotations/
    vatex_training_v1.0.json                 # train, 25,991 clips, 259,910 English captions
    vatex_validation_v1.0.json               # val, 3,000 clips, 30,000 English captions
    vatex_public_test_english_v1.1.json      # public test, 6,000 clips, 60,000 English captions
    vatex_private_test_without_annotations.json
  features/                                  # empty; put CLIP/I3D feature pickles here later
  raw/                                       # empty; put downloaded clips/videos here later
  samples/                                   # smoke test creates tiny_vatex_features.pickle
  scripts/
    dataset_vatex.py
    download_vatex_annotations.ps1
    smoke_test_vatex.py
```

VATEX clip ids encode the YouTube id and time window:

```text
Ptf_2VRj-V0_000122_000132
youtube id = Ptf_2VRj-V0
start = 122s
end = 132s
```

The private test file only contains `videoID`, so the loader returns
`caption=None` for that split.

## Loader Behavior

`scripts/dataset_vatex.py` exposes `VATEXCaptionDataset`.

Default split mapping:

| Split | File |
| --- | --- |
| `train` | `annotations/vatex_training_v1.0.json` |
| `val` / `validation` | `annotations/vatex_validation_v1.0.json` |
| `test` / `public_test` | `annotations/vatex_public_test_english_v1.1.json` |
| `private_test` | `annotations/vatex_private_test_without_annotations.json` |

Default `caption_mode="auto"` uses all captions for `train`, the first caption
for `val/test`, and one metadata-only record for private test.  Returned items
are dictionaries with:

```text
clip_id, video_id, youtube_url, start, end, caption, captions,
chinese_captions, feature, feature_mask, raw_video_path
```

Features are optional.  If supplied, `features_path` should point to a pickle
with either:

```python
{clip_id: (features, mask)}
{video_id: (features, mask)}
{clip_id: {"features": features, "mask": mask}}
```

## Smoke Test

The smoke test does not need real videos.  It samples train/val/test records,
creates `samples/tiny_vatex_features.pickle`, then verifies that PyTorch
`DataLoader` can stack `(B, 12, 512)` feature tensors.

```powershell
& "D:\Users\Administrator\anaconda3\envs\vc_local\python.exe" datasets\VATEX\scripts\smoke_test_vatex.py
```

Run again without rewriting the tiny fixture:

```powershell
& "D:\Users\Administrator\anaconda3\envs\vc_local\python.exe" datasets\VATEX\scripts\smoke_test_vatex.py --skip-create-features
```

## Frame Extraction

No frames were extracted in this setup pass.  When raw videos are available,
store them in `VATEX/raw/` as either `<clip_id>.mp4` or `<video_id>.mp4`.
For this project, the most compatible feature output is:

```text
VATEX/features/ViT-B-32_k_split_ks12_features.pickle
```

with pickle values:

```python
{
    "Ptf_2VRj-V0_000122_000132": (features[12, 512], mask[12]),
}
```

## Reference Summary

- Official VATEX JSON is clip-level and stores `videoID`, `enCap`, and `chCap`.
- Local baseline loaders expect pre-extracted visual features plus caption text.
- Hugging Face VATEX mirrors the same annotation fields and is a useful schema
  sanity check.
- For training, using all 10 English captions per clip matches common caption
  training practice.  For validation/test, using one item per clip avoids
  duplicating video features during generation/evaluation.

