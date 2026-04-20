# Repository Guidelines

## Project Structure & Module Organization
This workspace is organized by dataset and experiment root, not as a single packaged app. `datasets/MSVD/` holds captions, split files, feature assets, and preprocessing helpers. `projects/baseline/` is the CLIP4Clip-style anchor. Active research variants live in `projects/structed_caption/`, `projects/qb3s_querybridge/`, and `projects/structured_from_scratch/`, each with root-level `train_*.py`, `eval_*.py`, `build_*.py`, and `run_*.sh` entrypoints plus supporting folders such as `dataloaders/`, `tests/`, `analysis/`, `annotations/`, and `runs/`. `projects/MultiSemanticHead/` is the clean reusable package, with source under `src/`, examples under `examples/`, and tests under `tests/`.

## Build, Test, and Development Commands
Commands are project-scoped. Common local checks:

```powershell
python -m pytest projects/MultiSemanticHead/tests/test_contract.py
python -m pytest projects/structed_caption/tests -v
python projects/structed_caption/build_structured_gt_local.py --dataset_type msvd --annotations_path datasets/MSVD/annotations_preprocessed.txt --output_path projects/structed_caption/annotations/msvd_structured_train.json
python projects/structed_caption/remote_ssh.py --cmd "cd /mnt/sda/Disk_D/zhangwei/projects/VC/project/structured_caption && bash run_structured_phrase_mainline.sh D 2"
```

Use local Python for parsing, unit tests, and documentation updates. Use `remote_ssh.py` for training, evaluation, and GPU inspection on the server.

## Coding Style & Naming Conventions
Use Python and shell with 4-space indentation in Python. Prefer explicit imports, type hints on new code, and small helper functions over monolithic scripts. Follow existing names: `snake_case` for files, functions, variables, and CLI flags; `PascalCase` for classes; `test_<feature>.py` for tests; `run_<experiment>.sh` for launchers. Keep new experiments isolated behind new flags or launchers instead of silently changing baseline behavior.

## Testing Guidelines
`pytest` is the standard test runner. Keep tests deterministic and offline; mock API/network calls instead of hitting live services. For model or preprocessing changes, add a focused regression test near the touched project and pair it with a short remote smoke run when metrics or checkpoints are involved. Record caption metrics consistently: `BLEU-4`, `CIDEr`, `METEOR`, and `ROUGE-L`.

## Commit & Pull Request Guidelines
This staging mirror does not include a top-level `.git` history, so use short imperative commit messages with a scope prefix, for example `structed_caption: add phrase export regression test`. PRs should state the touched project, dataset split (`MSVD val/test`), exact command or launcher, produced artifact paths, and metric deltas. Include screenshots only for report or HTML output changes.

## Security & Configuration Tips
Do not commit `.keys`, PATs, SSH credentials, or large generated artifacts from `runs/`, `nohup_logs/`, or `temp/`. Keep dataset paths relative to the workspace where possible, and treat server-only execution details as runtime configuration rather than source code.
