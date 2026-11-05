"""A fake Hub for tests, at HF_ENDPOINT: no real Hub, bucket or job is ever reached.

- GET /api/buckets/...: every bucket is empty (`[]`); `hub.asked` lists these requests.
- POST /api/jobs/<ns>: the job runs on this machine as `python -m flock.worker.main` (what
  flock's worker image ends in, `exec flock-worker`) with the job's environment and secrets,
  minus BUCKET (the agents' state stays on this disk), its /work and HOME under `root/<job id>`.
  `hub.jobs[id]` keeps the job spec as huggingface_hub sent it.
- GET /api/jobs/<ns>/<id>: RUNNING, then COMPLETED (exit 0), ERROR (another exit) or CANCELED.
- POST /api/jobs/<ns>/<id>/cancel: a running job's processes get SIGTERM; CANCELED.
- GET /api/jobs/<ns>?label=k=v: the jobs with every one of those labels.
- Anything else: 404.
"""

from __future__ import annotations

import itertools
import json
import os
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit


class Hub:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.asked: list[str] = []
        self.jobs: dict[str, dict[str, Any]] = {}
        self.procs: dict[str, subprocess.Popen] = {}
        self.cancelled: set[str] = set()
        self._ids = itertools.count(1)  # one id per job, also from concurrent requests

    def run(self, ns: str, spec: dict[str, Any]) -> dict[str, Any]:
        job = f"job{next(self._ids)}"
        d = self.root / job
        env = {**{k: os.environ[k] for k in ("PATH", "PYTHONPATH") if k in os.environ},
               **spec.get("environment", {}), **spec.get("secrets", {}),
               "WORK_ROOT": str(d / "work" / "agents"), "HOME": str(d / "home")}
        env.pop("BUCKET", None)
        d.mkdir(parents=True)
        with (d / "log").open("wb") as log:
            self.procs[job] = subprocess.Popen(
                [sys.executable, "-m", "flock.worker.main"], cwd=d, env=env, stdout=log,
                stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
        self.jobs[job] = {**spec, "namespace": ns}
        return self.info(job)

    def info(self, job: str) -> dict[str, Any]:
        rc = self.procs[job].poll()
        stage = ("CANCELED" if job in self.cancelled else "RUNNING" if rc is None
                 else "COMPLETED" if rc == 0 else "ERROR")
        spec = self.jobs[job]
        return {**spec, "id": job, "status": {"stage": stage, "message": None},
                "owner": {"id": "0", "name": spec["namespace"], "type": "user"}}

    def cancel(self, job: str) -> None:
        if self.procs[job].poll() is None:  # an ended job keeps its stage
            self.cancelled.add(job)
            os.killpg(self.procs[job].pid, signal.SIGTERM)


def handler(hub: Hub) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: Any) -> None:
            pass

        def _json(self, obj: Any, code: int = 200) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            url = urlsplit(self.path)
            parts = url.path.strip("/").split("/")  # api, jobs, ns[, id]
            if parts[:2] == ["api", "buckets"]:
                hub.asked.append(self.path)
                self._json([])
            elif parts[:2] == ["api", "jobs"] and len(parts) == 4 and parts[3] in hub.jobs:
                self._json(hub.info(parts[3]))
            elif parts[:2] == ["api", "jobs"] and len(parts) == 3:
                want = dict(v.split("=", 1) for k, v in parse_qsl(url.query) if k == "label")
                self._json([hub.info(j) for j, s in hub.jobs.items() if s["namespace"] == parts[2]
                            and want.items() <= s.get("labels", {}).items()])
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            parts = urlsplit(self.path).path.strip("/").split("/")
            size = int(self.headers.get("content-length") or 0)
            body = json.loads(self.rfile.read(size) or b"{}")
            if parts[:2] == ["api", "jobs"] and len(parts) == 3:
                self._json(hub.run(parts[2], body))
            elif parts[:2] == ["api", "jobs"] and parts[4:] == ["cancel"] and parts[3] in hub.jobs:
                hub.cancel(parts[3])
                self._json({})
            else:
                self._json({"error": "not found"}, 404)

    return Handler


def serve(root: Path) -> tuple[Hub, ThreadingHTTPServer, str]:
    """Start on a free port in a daemon thread: (hub, server, its HF_ENDPOINT)."""
    hub = Hub(root)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler(hub))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05},  # fast shutdown()
                     daemon=True).start()
    return hub, srv, f"http://127.0.0.1:{srv.server_port}"
