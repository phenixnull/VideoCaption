#!/usr/bin/env python3
"""Stable SSH wrapper for the VC project.

This script avoids fragile PowerShell inline quoting by driving the existing
OpenSSH client from Python. It supports two common modes:

1. `--cmd "<remote shell command>"` for one-shot remote commands.
2. `--stdin-script` or `--script-file <path>` for multiline bash scripts.

Examples:
  python remote_ssh.py --cmd "nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader"
  python remote_ssh.py --stdin-script < local_script.sh
  python remote_ssh.py --script-file Z:\\VC\\tmp_remote_check.sh
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_HOST = "172.18.232.201"
DEFAULT_PORT = 8800
DEFAULT_USER = "zhangwei"
DEFAULT_CONDA_ENV = "vcr"
DEFAULT_CONDA_SH = "/mnt/sda/Disk_D/zhangwei/anaconda3/etc/profile.d/conda.sh"
DEFAULT_SSH_EXE = r"C:\Windows\System32\OpenSSH\ssh.exe"


def normalize_script(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.endswith("\n"):
        text += "\n"
    return text


def build_bootstrap(args: argparse.Namespace) -> list[str]:
    lines: list[str] = []
    if not args.no_conda:
        lines.append(f"source {shlex.quote(args.conda_sh)}")
        lines.append(
            f"conda activate {shlex.quote(args.conda_env)} >/dev/null 2>&1"
        )
    if args.cwd:
        lines.append(f"cd {shlex.quote(args.cwd)}")
    return lines


def build_remote_cmd(args: argparse.Namespace) -> str:
    pieces = build_bootstrap(args)
    pieces.append(args.cmd)
    joined = " && ".join(pieces)
    return f"bash -lc {shlex.quote(joined)}"


def build_remote_script(args: argparse.Namespace, body: str) -> tuple[str, str]:
    bootstrap = build_bootstrap(args)
    content = "\n".join(bootstrap + [body]) if bootstrap else body
    return "tr -d '\\r' | bash -s", normalize_script(content)


def build_ssh_base(args: argparse.Namespace) -> list[str]:
    ssh_exe = args.ssh_exe
    if os.name != "nt" and ssh_exe == DEFAULT_SSH_EXE:
        ssh_exe = "ssh"
    return [
        ssh_exe,
        "-F",
        "NUL" if os.name == "nt" else "/dev/null",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=NUL" if os.name == "nt" else "/dev/null",
        "-p",
        str(args.port),
        f"{args.user}@{args.host}",
    ]


def load_script_text(args: argparse.Namespace) -> str:
    if args.script_file:
        return Path(args.script_file).read_text(encoding="utf-8")
    if args.stdin_script:
        data = sys.stdin.read()
        if not data:
            raise SystemExit("--stdin-script was set but stdin is empty.")
        return data
    raise SystemExit("No script input was provided.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project SSH wrapper for remote VC execution.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--ssh-exe", default=DEFAULT_SSH_EXE)
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENV)
    parser.add_argument("--conda-sh", default=DEFAULT_CONDA_SH)
    parser.add_argument("--cwd", help="Optional remote working directory.")
    parser.add_argument("--no-conda", action="store_true")
    parser.add_argument("--cmd", help="One-shot remote shell command.")
    parser.add_argument("--script-file", help="Path to a local bash script.")
    parser.add_argument(
        "--stdin-script",
        action="store_true",
        help="Read a multiline bash script from stdin.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress printing the resolved remote mode to stderr.",
    )
    args = parser.parse_args()

    modes = [bool(args.cmd), bool(args.script_file), bool(args.stdin_script)]
    if sum(modes) != 1:
        parser.error("Choose exactly one of --cmd, --script-file, or --stdin-script.")
    return args


def main() -> int:
    args = parse_args()
    ssh_base = build_ssh_base(args)

    if args.cmd:
        remote = build_remote_cmd(args)
        if not args.quiet:
            print(f"[remote_ssh] mode=cmd host={args.host} port={args.port}", file=sys.stderr)
        proc = subprocess.run(ssh_base + [remote], text=True)
        return proc.returncode

    script_text = load_script_text(args)
    remote, payload = build_remote_script(args, script_text)
    if not args.quiet:
        mode = "script-file" if args.script_file else "stdin-script"
        print(f"[remote_ssh] mode={mode} host={args.host} port={args.port}", file=sys.stderr)
    proc = subprocess.run(ssh_base + [remote], input=payload, text=True)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
