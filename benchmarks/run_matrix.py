"""Run reproducible CLI benchmark combinations and write JSON Lines output."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=131072)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--attention", nargs="+", default=["triton"])
    parser.add_argument("--storage", nargs="+", default=["auto"])
    parser.add_argument("--bits", nargs="+", type=int, default=[2, 4])
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="parent directory for temporary stores; required to target a specific NVMe mount",
    )
    args = parser.parse_args()

    results = []
    for storage in args.storage:
        for attention in args.attention:
            for bits in args.bits:
                command = [
                    sys.executable,
                    "-m",
                    "kvssd.cli",
                    "benchmark",
                    "--tokens",
                    str(args.tokens),
                    "--iterations",
                    str(args.iterations),
                    "--warmup",
                    str(args.warmup),
                    "--bits",
                    str(bits),
                    "--storage-backend",
                    storage,
                    "--attention-backend",
                    attention,
                ]
                if args.work_dir is not None:
                    command.extend(["--work-dir", str(args.work_dir)])
                completed = subprocess.run(command, check=True, capture_output=True, text=True)
                results.append(json.loads(completed.stdout))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(result, sort_keys=True) + "\n" for result in results),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
