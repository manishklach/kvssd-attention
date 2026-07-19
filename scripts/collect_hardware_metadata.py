"""Collect reproducible, best-effort hardware qualification metadata as JSON."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


def _optional_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _timeout_text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value or ""


def _torch_runtime() -> dict[str, object] | None:
    try:
        import torch
    except ImportError:
        return None
    device_available = torch.cuda.is_available()
    return {
        "version": torch.__version__,
        "cuda": torch.version.cuda,
        "rocm": torch.version.hip,
        "device_available": device_available,
        "device_name": torch.cuda.get_device_name() if device_available else None,
    }


def _command(*command: str) -> dict[str, object]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"command": list(command), "available": False}
    try:
        completed = subprocess.run(
            [executable, *command[1:]], capture_output=True, text=True, check=False, timeout=30
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "command": list(command),
            "available": True,
            "timed_out": True,
            "stdout": _timeout_text(exc.stdout),
            "stderr": _timeout_text(exc.stderr),
        }
    return {
        "command": list(command),
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    args = parser.parse_args()

    storage_root = args.storage_root.expanduser().resolve()
    storage_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "platform": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "system": platform.platform(),
            "machine": platform.machine(),
            "kernel": platform.release(),
        },
        "runner": {
            "name": os.getenv("RUNNER_NAME"),
            "labels": os.getenv("RUNNER_LABELS"),
            "repository": os.getenv("GITHUB_REPOSITORY"),
            "workflow": os.getenv("GITHUB_WORKFLOW"),
            "run_id": os.getenv("GITHUB_RUN_ID"),
            "run_attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
            "sha": os.getenv("GITHUB_SHA"),
        },
        "storage_root": str(storage_root),
        "python_packages": {
            name: _optional_version(name) for name in ("torch", "triton", "kvssd-attention")
        },
        "torch_runtime": _torch_runtime(),
        "commands": {
            "findmnt": _command(
                "findmnt",
                "--target",
                str(storage_root),
                "--json",
                "-o",
                "SOURCE,TARGET,FSTYPE,OPTIONS",
            ),
            "lsblk": _command("lsblk", "--json", "-o", "NAME,MODEL,SIZE,TYPE,FSTYPE,MOUNTPOINTS"),
            "lspci": _command("lspci", "-nn"),
            "nvme": _command("nvme", "list", "-o", "json"),
            "nvidia_smi": _command(
                "nvidia-smi",
                "--query-gpu=name,driver_version,pci.bus_id,memory.total",
                "--format=csv,noheader",
            ),
            "rocm_smi": _command(
                "rocm-smi", "--showproductname", "--showdriverversion", "--json"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
