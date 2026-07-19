from __future__ import annotations

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace


def _load_benchmark():
    path = Path(__file__).parents[1] / "benchmarks" / "vllm_prefix_reuse.py"
    spec = importlib.util.spec_from_file_location("kvssd_vllm_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark = _load_benchmark()


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            return
        if self.path == "/metrics":
            payload = b'vllm:external_prefix_cache_hits_total{engine="0"} 64\n'
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        assert self.path == "/v1/completions"
        length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(length))
        assert request["stream"] is True
        events = [
            {"choices": [{"text": "ready"}]},
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 128,
                    "completion_tokens": 1,
                    "prompt_tokens_details": {"cached_tokens": 96},
                },
            },
        ]
        payload = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        payload += "data: [DONE]\n\n"
        encoded = payload.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *_args) -> None:
        pass


def test_streaming_completion_and_metric_parser() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}"
        result = benchmark._completion(base_url, "model", "prompt", 5)
        metrics = benchmark._get(f"{base_url}/metrics").decode()
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    assert result["output"] == "ready"
    assert result["cached_tokens"] == 96
    assert result["prompt_tokens"] == 128
    assert benchmark._metric_total(metrics, benchmark.EXTERNAL_HITS) == 64
    underscore = metrics.replace("vllm:external", "vllm_external")
    assert benchmark._metric_total(underscore, benchmark.EXTERNAL_HITS) == 64


def test_record_stats_only_counts_kvssd_files(tmp_path: Path) -> None:
    (tmp_path / "one.kvssd").write_bytes(b"1" * 4)
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "two.kvssd").write_bytes(b"2" * 8)
    (nested / "ignored.tmp").write_bytes(b"3" * 16)
    assert benchmark._record_stats(tmp_path) == {"records": 2, "bytes": 12}


def test_server_command_uses_isolated_executable_and_fail_closed_loads(
    tmp_path: Path, monkeypatch
) -> None:
    python = tmp_path / "python"
    executable = tmp_path / ("vllm.exe" if benchmark.os.name == "nt" else "vllm")
    executable.write_bytes(b"")
    monkeypatch.setattr(benchmark.sys, "executable", str(python))
    args = SimpleNamespace(
        model="model",
        served_model_name="served",
        host="127.0.0.1",
        port=8000,
        max_model_len=2048,
        gpu_memory_utilization=0.5,
        root=tmp_path / "records",
        cpu_bytes=1024,
        bits=4,
        group_size=32,
        io_threads=2,
    )
    command = benchmark._server_command(args)
    transfer = json.loads(command[command.index("--kv-transfer-config") + 1])
    assert command[0] == str(executable)
    assert transfer["kv_load_failure_policy"] == "fail"
    assert transfer["kv_connector_extra_config"]["secondary_tiers"][0]["type"] == "kvssd"
