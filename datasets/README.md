# Dataset Setup Log: VATEX and ActivityNet Captions

Date: 2026-04-24

This folder now contains lightweight, loader-ready annotation copies for VATEX
and ActivityNet Captions.  Full raw videos and full pretrained feature packs
were not downloaded, because both datasets are large and the immediate goal was
to make train/val/test dataloaders verifiably usable.

## Directory Layout

```text
datasets/
  VATEX/
    annotations/
      vatex_training_v1.0.json
      vatex_validation_v1.0.json
      vatex_public_test_english_v1.1.json
      vatex_private_test_without_annotations.json
    features/
    raw/
    samples/
    scripts/
      dataset_vatex.py
      download_vatex_annotations.ps1
      smoke_test_vatex.py
    README/README.md
  ActivityNet/
    annotations/
      captions.zip
      readme.txt
      train.json
      train_ids.json
      val_1.json
      val_2.json
      val_ids.json
      test_ids.json
    features/
    raw/
    samples/
    scripts/
      dataset_activitynet_captions.py
      download_activitynet_annotations.ps1
      smoke_test_activitynet.py
    README/README.md
```

## Downloaded Files

VATEX source: [official VATEX download page](https://eric-xw.github.io/vatex-website/download.html).

| Relative path | Bytes | Records | English captions |
| --- | ---: | ---: | ---: |
| `VATEX/annotations/vatex_training_v1.0.json` | 57,319,458 | 25,991 | 259,910 |
| `VATEX/annotations/vatex_validation_v1.0.json` | 6,598,992 | 3,000 | 30,000 |
| `VATEX/annotations/vatex_public_test_english_v1.1.json` | 4,933,553 | 6,000 | 60,000 |
| `VATEX/annotations/vatex_private_test_without_annotations.json` | 263,676 | 6,278 | 0 |

ActivityNet Captions source: [official DenseVid page](https://cs.stanford.edu/people/ranjaykrishna/densevid/).  The same JSON files are mirrored at [Hugging Face](https://huggingface.co/datasets/friedrichor/ActivityNet_Captions/tree/main/raw_data) if the Stanford zip is unavailable.

| Relative path | Bytes | Videos | Segments |
| --- | ---: | ---: | ---: |
| `ActivityNet/annotations/train.json` | 4,041,288 | 10,009 | 37,421 |
| `ActivityNet/annotations/val_1.json` | 1,910,790 | 4,917 | 17,505 |
| `ActivityNet/annotations/val_2.json` | 1,727,037 | 4,885 | 17,031 |
| `ActivityNet/annotations/train_ids.json` | 170,408 | 10,024 | 0 |
| `ActivityNet/annotations/val_ids.json` | 83,742 | 4,926 | 0 |
| `ActivityNet/annotations/test_ids.json` | 85,748 | 5,044 | 0 |

## Environment Notes

Baseline reference: `projects/baseline/readme/README01.md` recommends a
`vcr` conda environment with Python 3.8.20 and torch.  For this loader-only
check, only `torch` and `numpy` are required.  The existing local conda env
`vc_local` already has GPU torch `2.11.0+cu126`, and the smoke tests were run
with it while preparing the files.

Minimal commands:

```powershell
conda create --name vcr python=3.8.20 -y
conda activate vcr
python -m pip install --upgrade pip
pip install numpy tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

If that PyTorch index is slow, use the local proxy:

```powershell
$env:HTTP_PROXY="http://127.0.0.1:6789"
$env:HTTPS_PROXY="http://127.0.0.1:6789"
```

## Dataloader References Checked

- Local baseline: `projects/baseline/dataloaders/dataset_msrvtt_feats.py`,
  `dataset_msvd_feats.py`, and `rawvideo_util.py`.  These use the project
  pickle convention `{video_id: (features, mask)}` and return tensors plus
  caption metadata.
- [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip): common video-caption
  retrieval loader style and the original source of several MSRVTT/MSVD
  conventions mirrored by the baseline.
- [BMT](https://github.com/v-iashin/BMT): ActivityNet Captions is converted
  into TSV rows with `video_id`, `caption`, `start`, `end`, `duration`, then
  paired with I3D/VGGish feature stacks.
- [PDVC](https://github.com/ttengwang/PDVC): keeps ActivityNet Captions JSON
  fields `timestamps` and `sentences`, supports C3D/TSN/TSP feature folders,
  and evaluates against both `val_1.json` and `val_2.json`.
- [Hugging Face VATEX](https://huggingface.co/datasets/HuggingFaceM4/vatex):
  confirms the official VATEX JSON schema centered on `videoID`, `enCap`, and
  `chCap`.

The implemented loaders keep the baseline-friendly feature pickle path while
also preserving fields used by dense-captioning repositories, especially
ActivityNet timestamps.

## Frame Extraction Status

No frame extraction was performed in this setup pass.  `raw/` is intentionally
empty for both datasets.  Future extraction can reuse the baseline convention:
sample K frames per video or clip, encode with CLIP, and save
`features/<name>.pickle` as `{id: (features[T,D], mask[T])}`.  The smoke tests
generate tiny synthetic feature pickles under `samples/` only to verify loader
plumbing.

## Smoke Test Commands

```powershell
& "D:\Users\Administrator\anaconda3\envs\vc_local\python.exe" datasets\VATEX\scripts\smoke_test_vatex.py
& "D:\Users\Administrator\anaconda3\envs\vc_local\python.exe" datasets\ActivityNet\scripts\smoke_test_activitynet.py
```

After creating `vcr`, the same commands can be run with:

```powershell
conda run -n vcr python datasets\VATEX\scripts\smoke_test_vatex.py
conda run -n vcr python datasets\ActivityNet\scripts\smoke_test_activitynet.py
```

