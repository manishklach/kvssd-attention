from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_matrix_uses_requested_work_root(tmp_path: Path) -> None:
    output = tmp_path / "result.jsonl"
    work_root = tmp_path / "nvme"
    subprocess.run(
        [
            sys.executable,
            "benchmarks/run_matrix.py",
            "--output",
            str(output),
            "--work-dir",
            str(work_root),
            "--tokens",
            "8",
            "--iterations",
            "1",
            "--warmup",
            "0",
            "--storage",
            "buffered",
            "--attention",
            "reference",
            "--bits",
            "4",
        ],
        check=True,
    )
    results = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(results) == 1
    assert results[0]["work_root"] == str(work_root.resolve())
    assert results[0]["storage_backend"]["name"] == "buffered"
