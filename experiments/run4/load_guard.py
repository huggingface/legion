"""Read vLLM metrics before increasing a wave; this module launches nothing."""
from __future__ import annotations

import math
import os
import re
import time

import httpx

METRIC = re.compile(r"^(vllm:[a-z_]+)(?:\{([^}]*)\})?\s+([0-9.eE+-]+)(?:\s|$)")


def parse_metrics(text):
    values = {"running": 0.0, "waiting": 0.0, "completed": 0.0}
    seen = set()
    names = {"vllm:num_requests_running": "running", "vllm:num_requests_waiting": "waiting",
             "vllm:request_success_total": "completed"}
    for line in text.splitlines():
        match = METRIC.match(line)
        if match and match[1] in names:
            if match[1] == "vllm:request_success_total" and 'finished_reason="abort"' in (match[2] or ""):
                continue
            key = names[match[1]]
            seen.add(key)
            values[key] += float(match[3])
    if not {"running", "waiting"} <= seen:
        raise ValueError("missing vLLM request metrics")
    return values


def metrics_urls():
    explicit = os.environ.get("OPENAI_METRICS_URLS", "")
    if explicit:
        return [u.strip() for u in explicit.split(",") if u.strip()]
    urls = os.environ.get("OPENAI_BASE_URLS") or os.environ.get("OPENAI_BASE_URL", "")
    return [u.strip().removesuffix("/v1").rstrip("/") + "/metrics"
            for u in urls.split(",") if u.strip()]


def capacity_ok(urls=None, *, interval=60, get=None, sleep=time.sleep, clock=time.monotonic,
                max_request_seconds=None):
    try:
        threshold = float(os.environ.get("RUN4_MAX_REQUEST_SECONDS", "15")
                          if max_request_seconds is None else max_request_seconds)
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("invalid threshold")
    except (TypeError, ValueError):
        return False, "HOLD: RUN4_MAX_REQUEST_SECONDS must be finite and positive"
    urls = metrics_urls() if urls is None else urls
    if not urls:
        return False, "HOLD: configure vLLM metrics URLs before growing past 100 agents"
    get = get or (lambda url: httpx.get(url, timeout=20))
    samples = {url: [] for url in urls}
    try:
        for i in range(3):
            if i:
                sleep(interval)
            for url in urls:
                response = get(url)
                response.raise_for_status()
                samples[url].append((clock(), parse_metrics(response.text)))
    except (httpx.HTTPError, ValueError):
        return False, "HOLD: vLLM metrics unavailable or invalid"
    details = []
    for index, points in enumerate(samples.values()):
        start, first = points[0]
        end, last = points[-1]
        delta = last["completed"] - first["completed"]
        if delta < 0:
            return False, "HOLD: vLLM counters reset during capacity check"
        rate = delta / max(end - start, 0.001)
        tpr = last["running"] / rate if rate > 0 else (float("inf") if last["running"] else 0)
        details.append(f"node {index + 1}: waiting={last['waiting']:g}, ~{tpr:.1f}s/request")
        if all(p[1]["waiting"] > 0 for p in points) or tpr > threshold:
            return False, "HOLD: " + "; ".join(details)
    return True, "; ".join(details)


if __name__ == "__main__":
    ok, message = capacity_ok()
    print(message)
    raise SystemExit(0 if ok else 1)
