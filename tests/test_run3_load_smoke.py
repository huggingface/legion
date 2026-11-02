"""experiments/run3/load/load_smoke.py offline: the corpus and prompt shape, the /metrics parser,
the knee and the recommendation, and two dry runs: against tests/fake_openai.py (no /metrics)
and against a tiny vLLM-like server with /metrics (running, waiting, KV, prefix-cache
counters)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import fake_openai
from .test_world import ROOT

LS = ROOT / "experiments" / "run3" / "load" / "load_smoke.py"


def _mod():
    s = importlib.util.spec_from_file_location("run3_load_smoke", LS)
    m = importlib.util.module_from_spec(s)
    sys.modules["run3_load_smoke"] = m
    s.loader.exec_module(m)
    return m


def test_shape_corpus_metrics_and_knee():
    ls = _mod()
    assert ls.corpus(3, 5000) == ls.corpus(3, 5000) and len(ls.corpus(3, 5000)) == 5000
    sh = ls.Shape(46000)
    a, b, other = sh.messages(7, 1), sh.messages(7, 2), sh.messages(8, 1)
    assert a[0] == other[0]  # the shared system prefix
    assert a[1] == b[1] and a[1] != other[1]  # the agent's own history, stable across calls
    assert a[3]["content"] != b[3]["content"]  # one fresh tool result per request
    chars = sum(len(m["content"] or "") for m in a)
    assert 0.95 * 46000 * sh.cpt < chars + 300 * sh.cpt < 1.05 * 46000 * sh.cpt
    body = ls.body("m", a, 135, False)
    assert (body["tool_choice"], body["max_tokens"], body["chat_template_kwargs"]) == (
        "required", 135, {"thinking": False})
    assert {t["function"]["name"] for t in body["tools"]} >= {"edit_file", "run_bot", "done"}
    text = ("# HELP x\nvllm:num_requests_running{model_name=\"m\"} 12.0\n"
            "vllm:num_requests_waiting{model_name=\"m\"} 3\n"
            "vllm:kv_cache_usage_perc{model_name=\"m\"} 0.42\n"
            "vllm:prefix_cache_queries_total{model_name=\"m\"} 1000\n"
            "vllm:prefix_cache_hits_total{model_name=\"m\"} 900\n")
    assert ls.parse_metrics(text) == {"running": 12, "waiting": 3, "kv": 0.42, "hits": 900,
                                      "queries": 1000}
    s = ls.metric_summary([(0, {"running": 4, "waiting": 0, "kv": 0.2, "hits": 0, "queries": 0}),
                           (5, {"running": 8, "waiting": 2, "kv": 0.6, "hits": 90,
                                "queries": 100})])
    assert s["running_max"] == 8 and s["kv_max_pct"] == 60 and s["prefix_hit_pct"] == 90
    row = lambda c, rs, p50, err=0.0, w=0.0: {"agents": c, "req_s": rs, "p50_s": p50,
                                              "err_pct": err, "waiting_mean": w}
    assert ls.knee([row(8, 4, 2), row(16, 8, 2.1), row(32, 8.5, 3.5)]) == (
        16, "at 32: flat throughput, rising latency")
    assert ls.knee([row(8, 4, 2), row(16, 8, 2.1, w=3)]) == (8, "at 16: queueing")
    assert ls.knee([row(8, 4, 2, err=5)])[0] is None
    rec = ls.recommend([row(8, 4, 2.0), row(16, 8, 2.0)], tool_s=2.0)  # no knee: >=
    assert rec == {"knee": 16, "why": "no knee up to the last level", "duty": 0.5,
                   "agents": 25, "first_wave": 5, "at_least": True}


def _run(url: str, out, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(LS), "--base-url", url, "--model", "fake-model",
                           "--levels", "2,4", "--minutes", "0.08", "--prompt-tokens", "3000",
                           "--out", str(out), *extra], capture_output=True, text=True,
                          timeout=120, check=False)


def test_dry_run_against_the_fake_openai_server(tmp_path):
    srv = fake_openai.serve(0, str(tmp_path / "requests.jsonl"))
    try:
        r = _run(f"http://127.0.0.1:{srv.server_port}/v1", tmp_path / "out")
    finally:
        srv.shutdown()
    assert r.returncode == 0, r.stdout + r.stderr
    assert "metrics: not reachable" in r.stdout and "Recommended for one node" in r.stdout
    res = json.loads((tmp_path / "out" / "results.json").read_text())
    assert [x["agents"] for x in res["rows"]] == [2, 4] and all(x["errors"] == 0
                                                               for x in res["rows"])
    assert all(x["tool_call_pct"] == 100 for x in res["rows"])
    sent = [json.loads(x) for x in (tmp_path / "requests.jsonl").read_text().splitlines()]
    b = sent[-1]["body"]
    assert b["tool_choice"] == "required" and b["max_tokens"] == 135
    assert b["messages"][-1]["role"] == "tool"
    assert (tmp_path / "out" / "load-smoke.md").read_text().startswith("# Load smoke")


class _Vllm(BaseHTTPRequestHandler):
    """POST /v1/chat/completions (a tool call, usage with cached tokens) and GET /metrics with
    counters that move."""

    n = 0
    lock = threading.Lock()

    def log_message(self, *a):
        pass

    def _send(self, code: int, data: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        with self.lock:
            type(self).n += 1
        d = {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c", "type": "function", "function": {"name": "done",
                                                         "arguments": "{\"summary\": \"x\"}"}}]}}],
             "usage": {"prompt_tokens": 3000, "completion_tokens": 20,
                       "prompt_tokens_details": {"cached_tokens": 2700}}}
        self._send(200, json.dumps(d).encode(), "application/json")

    def do_GET(self):
        if self.path != "/metrics":
            return self._send(404, b"{}", "application/json")
        n = type(self).n
        text = (f"vllm:num_requests_running 3\nvllm:num_requests_waiting 0\n"
                f"vllm:kv_cache_usage_perc 0.25\nvllm:prefix_cache_queries_total {n * 10}\n"
                f"vllm:prefix_cache_hits_total {n * 9}\n")
        self._send(200, text.encode(), "text/plain")


def test_dry_run_reads_vllm_metrics(tmp_path):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Vllm)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        r = _run(f"http://127.0.0.1:{srv.server_port}/v1", tmp_path / "out",
                 "--metrics-every", "0.5")
    finally:
        srv.shutdown()
    assert r.returncode == 0, r.stdout + r.stderr
    assert "metrics: reachable" in r.stdout
    row = json.loads((tmp_path / "out" / "results.json").read_text())["rows"][-1]
    assert row["cached_pct"] == 90.0 and row["running_max"] == 3 and row["kv_max_pct"] == 25
    assert row["prefix_hit_pct"] == 90.0 and row["waiting_max"] == 0


def test_job_script_rehearsal(tmp_path):
    """load/job.sh as the HF Job runs it (LS_PREINSTALLED, a local dir as LS_OUT): the script
    arrives gzipped + base64 in the env, runs against the endpoint, results land in LS_OUT."""
    import base64
    import gzip
    import os

    srv = fake_openai.serve(0, str(tmp_path / "requests.jsonl"))
    env = {**os.environ, "LS_PREINSTALLED": "1", "PYTHON": sys.executable,
           "OPENAI_BASE_URL": f"http://127.0.0.1:{srv.server_port}/v1",
           "MODEL_WORKER": "fake-model", "LS_OUT": str(tmp_path / "bucket" / "load"),
           "LS_B64": base64.b64encode(gzip.compress(LS.read_bytes())).decode(),
           "LS_ARGS": "--levels 2 --minutes 0.05 --prompt-tokens 2000"}
    try:
        r = subprocess.run(["bash", "-c", (LS.parent / "job.sh").read_text()], env=env,
                           capture_output=True, text=True, timeout=120, check=False)
    finally:
        srv.shutdown()
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "load smoke: exit 0" in r.stdout
    assert (tmp_path / "bucket" / "load" / "load-smoke.md").is_file()
