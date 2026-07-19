"""Qualify persistent KVSSD prefix reuse through a real vLLM OpenAI server."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


EXTERNAL_HITS = "vllm:external_prefix_cache_hits_total"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _memory_fraction(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("value must be in (0, 1]")
    return parsed


def _get(url: str, timeout: float = 10) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def _completion(base_url: str, model: str, prompt: str, timeout: float) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": 2,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_token_at: float | None = None
    output_parts: list[str] = []
    usage: dict[str, Any] = {}
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            choices = event.get("choices") or []
            if choices and first_token_at is None:
                first_token_at = time.perf_counter()
            output_parts.extend(str(choice.get("text", "")) for choice in choices)
            if event.get("usage"):
                usage = event["usage"]
    finished = time.perf_counter()
    if first_token_at is None:
        raise RuntimeError("vLLM stream completed without a token event")
    details = usage.get("prompt_tokens_details") or {}
    return {
        "ttft_ms": (first_token_at - started) * 1e3,
        "total_ms": (finished - started) * 1e3,
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "cached_tokens": int(details.get("cached_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "output": "".join(output_parts),
    }


def _metric_total(metrics: str, metric: str) -> float:
    candidates = {metric, metric.replace(":", "_")}
    total = 0.0
    found = False
    for line in metrics.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.match(r"([^\s{]+)(?:\{[^}]*\})?\s+([^\s]+)", line)
        if match is None or match.group(1) not in candidates:
            continue
        total += float(match.group(2))
        found = True
    if not found:
        raise KeyError(f"Prometheus metric {metric!r} was not exported")
    return total


def _record_stats(root: Path) -> dict[str, int]:
    records = list(root.rglob("*.kvssd"))
    return {"records": len(records), "bytes": sum(path.stat().st_size for path in records)}


def _wait_for_records(root: Path, timeout: float) -> dict[str, int]:
    deadline = time.monotonic() + timeout
    previous: dict[str, int] | None = None
    stable = 0
    while time.monotonic() < deadline:
        current = _record_stats(root)
        if current["records"] and current == previous:
            stable += 1
            if stable >= 2:
                return current
        else:
            stable = 0
        previous = current
        time.sleep(1)
    raise TimeoutError(f"no stable KVSSD records appeared under {root} within {timeout}s")


def _tail(path: Path, lines: int = 80) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except FileNotFoundError:
        return ""


def _wait_ready(base_url: str, process: subprocess.Popen[bytes], log: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(
                f"vLLM exited with code {returncode} before becoming ready\n{_tail(log)}"
            )
        try:
            _get(f"{base_url}/health", timeout=2)
            return
        except (OSError, urllib.error.URLError):
            time.sleep(2)
    raise TimeoutError(f"vLLM did not become ready within {timeout}s\n{_tail(log)}")


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=10)


def _server_command(args: argparse.Namespace) -> list[str]:
    sibling = Path(sys.executable).with_name("vllm")
    if os.name == "nt":
        sibling = sibling.with_suffix(".exe")
    executable = str(sibling) if sibling.is_file() else shutil.which("vllm")
    if executable is None:
        raise RuntimeError("vllm executable is unavailable; install vLLM 0.25.x")
    transfer = {
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_load_failure_policy": "fail",
        "kv_connector_extra_config": {
            "spec_name": "KVSSDOffloadingSpec",
            "spec_module_path": "kvssd.integrations.vllm",
            "cpu_bytes_to_use": args.cpu_bytes,
            "block_size": 16,
            "secondary_tiers": [
                {
                    "type": "kvssd",
                    "root_dir": str(args.root),
                    "bits": args.bits,
                    "group_size": args.group_size,
                    "kvssd_dtype": "float16",
                    "n_io_threads": args.io_threads,
                }
            ],
        },
    }
    return [
        executable,
        "serve",
        args.model,
        "--served-model-name",
        args.served_model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        "float16",
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--enable-prefix-caching",
        "--enable-prompt-tokens-details",
        "--kv-transfer-config",
        json.dumps(transfer, separators=(",", ":")),
    ]


@contextmanager
def _server(
    args: argparse.Namespace, phase: str
) -> Iterator[tuple[subprocess.Popen[bytes], list[str], Path]]:
    command = _server_command(args)
    log = args.artifact_dir / f"vllm-{phase}.log"
    with log.open("wb") as output:
        process = subprocess.Popen(
            command,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_ready(args.base_url, process, log, args.startup_timeout)
            yield process, command, log
        finally:
            _stop_process(process)


def _prompt(label: str, repetitions: int) -> str:
    sentence = f"KVSSD persistent prefix qualification segment {label}. "
    return sentence * repetitions + "\nRespond with one word: ready"


def _wait_external_hits(base_url: str, initial: float, timeout: float) -> tuple[float, str]:
    deadline = time.monotonic() + timeout
    latest = ""
    while time.monotonic() < deadline:
        latest = _get(f"{base_url}/metrics").decode("utf-8")
        total = _metric_total(latest, EXTERNAL_HITS)
        if total > initial:
            return total, latest
        time.sleep(1)
    raise TimeoutError(f"{EXTERNAL_HITS} did not increase within {timeout}s")


def _runtime_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("vllm", "torch", "triton", "kvssd-attention"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if list(args.root.rglob("*.kvssd")):
        raise ValueError(f"qualification root already contains KVSSD records: {args.root}")
    target = _prompt("persistent-target", args.prompt_repetitions)
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "model": args.model,
        "served_model_name": args.served_model_name,
        "root": str(args.root),
        "bits": args.bits,
        "group_size": args.group_size,
        "cpu_bytes": args.cpu_bytes,
        "versions": _runtime_versions(),
    }

    with _server(args, "populate") as (_, command, _):
        result["server_command"] = command
        result["cold"] = _completion(
            args.base_url, args.served_model_name, target, args.request_timeout
        )
        result["warm"] = _completion(
            args.base_url, args.served_model_name, target, args.request_timeout
        )
        evictions = []
        for index in range(args.eviction_prompts):
            evictions.append(
                _completion(
                    args.base_url,
                    args.served_model_name,
                    _prompt(f"eviction-{index}", args.prompt_repetitions),
                    args.request_timeout,
                )
            )
        result["eviction_requests"] = evictions
        result["records_before_restart"] = _wait_for_records(args.root, args.record_timeout)
        metrics = _get(f"{args.base_url}/metrics").decode("utf-8")
        (args.artifact_dir / "metrics-populate.prom").write_text(metrics, encoding="utf-8")

    with _server(args, "reload"):
        before_metrics = _get(f"{args.base_url}/metrics").decode("utf-8")
        before_hits = _metric_total(before_metrics, EXTERNAL_HITS)
        result["persistent"] = _completion(
            args.base_url, args.served_model_name, target, args.request_timeout
        )
        after_hits, after_metrics = _wait_external_hits(
            args.base_url, before_hits, args.metric_timeout
        )
        (args.artifact_dir / "metrics-reload.prom").write_text(after_metrics, encoding="utf-8")
        result["external_prefix_cache_hits_delta"] = after_hits - before_hits

    result["records_after_restart"] = _record_stats(args.root)
    if result["persistent"]["cached_tokens"] <= 0:
        raise AssertionError("restarted vLLM request reported zero cached prompt tokens")
    if result["external_prefix_cache_hits_delta"] <= 0:
        raise AssertionError("restarted vLLM request did not increment external prefix cache hits")
    result["status"] = "passed"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="verify that vLLM reloads KVSSD prefix blocks after a process restart",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local model path")
    parser.add_argument("--root", type=Path, required=True, help="new empty KVSSD record root")
    parser.add_argument("--output", type=Path, required=True, help="qualification JSON result")
    parser.add_argument(
        "--artifact-dir", type=Path, required=True, help="server-log and metrics directory"
    )
    parser.add_argument("--host", default="127.0.0.1", help="vLLM listen host")
    parser.add_argument("--port", type=_positive_int, default=8000, help="vLLM listen port")
    parser.add_argument("--served-model-name", default="kvssd-qualification")
    parser.add_argument("--bits", type=int, choices=(2, 4), default=4, help="KVSSD bit width")
    parser.add_argument("--group-size", type=_positive_int, default=32)
    parser.add_argument("--io-threads", type=_positive_int, default=8)
    parser.add_argument(
        "--cpu-bytes",
        type=_positive_int,
        default=64 * 1024 * 1024,
        help="vLLM CPU-primary capacity; a small value exercises eviction pressure",
    )
    parser.add_argument("--max-model-len", type=_positive_int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=_memory_fraction, default=0.5)
    parser.add_argument("--prompt-repetitions", type=_positive_int, default=128)
    parser.add_argument("--eviction-prompts", type=_positive_int, default=4)
    parser.add_argument("--startup-timeout", type=_positive_float, default=600)
    parser.add_argument("--request-timeout", type=_positive_float, default=300)
    parser.add_argument("--record-timeout", type=_positive_float, default=60)
    parser.add_argument("--metric-timeout", type=_positive_float, default=30)
    args = parser.parse_args()
    args.root = args.root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.artifact_dir = args.artifact_dir.expanduser().resolve()
    args.base_url = f"http://{args.host}:{args.port}"
    args.root.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = _run(args)
    except Exception as exc:
        result = {
            "schema_version": 1,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "versions": _runtime_versions(),
            "records": _record_stats(args.root),
        }
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
