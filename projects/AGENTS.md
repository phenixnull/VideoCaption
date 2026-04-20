# Repository Guidelines

## Workspace Contract
- This directory `D:\Users\Administrator\Desktop\2025-CommercialOrder\OnGoingOrders\VideoCaption\projects` is a local code-only staging checkout. It intentionally does not contain the full server-side weights, datasets, or extracted features.
- The authoritative repository lives on the server at `/mnt/sda/Disk_D/zhangwei/projects/VC/` and is exposed locally through the SFTP mapping `Z:\VC`.
- Resume interrupted work from the evidence in `TaskSessionContinue.txt` before broad re-exploration.
- Path mapping must stay explicit:
  - local `baseline` -> remote `Z:\VC\project\baseline`
  - local `structed_caption` -> remote `Z:\VC\project\structured_caption`
  - local root `projects` -> remote `Z:\VC\project`

## Edit And Sync Policy
- Make code and documentation changes in this local staging tree first.
- Workflow is fixed: read, edit, and write locally first, then sync the touched file(s) to the server copy. Do not use the server repo as the primary editing surface.
- After every meaningful change, sync the touched file(s) to the server copy immediately. Do not leave local-only edits pending.
- Before each sync, state which file(s) will be pushed and the exact remote target path; after each sync, confirm the result.
- Prefer narrow file-level sync over blind whole-tree overwrite.
- Do not persist passwords, API keys, or other secrets in repo files, logs, or AGENTS notes.

## Environment Policy
- Use the server conda environment `vcr` for all real training, evaluation, checkpoint-dependent tests, GPU monitoring, and any run that needs the true server assets.
- The server `vcr` environment is the source of truth for runtime and dependency judgments on real runs; do not mark training, evaluation, or checkpoint-dependent work as blocked only because the local staging environment lacks packages.
- SSH target for server execution: `zhangwei@172.18.232.201:8800`.
- The login credential is managed outside the repository and must not be stored here.
- Prefer `structed_caption\remote_ssh.py` for remote execution because it already defaults to the correct host, port, user, and conda environment.
- The local environment `D:\Users\Administrator\anaconda3\envs\vcr` is only for lightweight tasks such as API extraction, parsing, static checks, and small smoke tests that do not require server-only data, features, or checkpoints.

## Execution Rules
- Training and evaluation must run from the server repository, not from this local clone.
- Any task that depends on datasets, weights, features, prior runs, or GPU state must be checked against the server-side repo under `Z:\VC` or `/mnt/sda/Disk_D/zhangwei/projects/VC/`.
- Keep GPU usage explicit and conservative; do not opportunistically expand to extra GPUs.

## Baseline Protection Rules
- The known `120+ CIDEr` legacy mainlines are protected. Do not silently mutate their launcher defaults, default config semantics, or evaluation path just to support a new experiment.
- New structured-caption rectification ideas must be isolated behind a dedicated launcher or dedicated config name. Prefer creating a new `run_*.sh` entry rather than repurposing an existing baseline script.
- Shared Python modules may add gated support for new experiments, but the default behavior must remain baseline-compatible when the new flags are not enabled.
- When introducing an experiment-specific flag bundle, record the dedicated launcher/config name and explicitly state that legacy baselines should keep those flags off.

## Reporting Contract
- When reporting progress, include: local edit path, remote sync target, whether sync is pending or completed, and which environment was used.
- For server runs, also record the remote working directory, launch command, PID or process handle when available, log path, and GPU assignment.
- Any headline progress claim, "current best" claim, "best evidence" claim, or target-hit claim for the video-caption project must use the relevant `test` split metrics only.
- Do not use validation metrics, dev metrics, or unlabeled metrics as the main evidence for milestone or target reporting.
- If a metric is not from the required `test` split, label the split explicitly and do not present it as proof that the target has been reached or is close.
