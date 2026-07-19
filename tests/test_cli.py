from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_benchmark_json_records_runtime_shape_and_backends(tmp_path: Path) -> None:
    work_root = tmp_path / "benchmark-root"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "kvssd.cli",
            "benchmark",
            "--cpu",
            "--tokens",
            "8",
            "--kv-heads",
            "1",
            "--query-heads",
            "1",
            "--head-dim",
            "32",
            "--block-tokens",
            "8",
            "--group-size",
            "16",
            "--warmup",
            "0",
            "--iterations",
            "1",
            "--storage-backend",
            "buffered",
            "--attention-backend",
            "reference",
            "--work-dir",
            str(work_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["runtime"]["torch"]
    assert result["shape"]["tokens"] == 8
    assert result["attention_backend"]["name"] == "reference"
    assert result["storage_backend"]["name"] == "buffered"
    assert result["work_root"] == str(work_root.resolve())
    assert work_root.is_dir()
