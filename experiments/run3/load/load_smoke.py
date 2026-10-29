"""Load smoke for run 3's vLLM endpoint: the run-3 request shape at fixed concurrency levels,
then the recommended first wave and agent cap for one node.

    OPENAI_BASE_URL=https://<host>/v1 MODEL_WORKER=<served id> [OPENAI_API_KEY=...] \\
      python experiments/run3/load/load_smoke.py --out <dir>

Request shape (an agent's model call in run 3): a ~46k-token prompt built from a fixed,
seeded corpus (a system prefix shared by every simulated agent + that agent's own history,
the same on each of its calls, so prefix caching sees what it sees in the run) + ONE fresh tool
result per request; the run-3 tools; `tool_choice: required`; `max_tokens` 135;
`chat_template_kwargs: {thinking: false}`. The prompt size is calibrated with one request
first (`usage.prompt_tokens`). Each level runs C closed-loop clients for --minutes (the first
20 %, at most 30 s, is warm-up and not counted).

Measured per level: req/s, p50/p95 latency, errors, prompt/output tokens, cached prompt tokens
(when the server reports them), and from vLLM's /metrics (the base URL without /v1), if
reachable: running and waiting requests, KV-cache usage, prefix-cache hit rate.

Knee: the last level before the first one that adds errors (> 1 %), queues (mean waiting >= 1),
or gains < 15 % req/s while p50 rises >= 40 %. Agents per node = 0.8 x knee / duty, where duty =
p50 / (p50 + --tool-s), the share of an agent's time spent waiting on the model (the rest is
tools and the harness). First wave = min(20, agents // 5).

Only stdlib + httpx; sends requests only to OPENAI_BASE_URL. Output: a table on stdout and
<out>/load-smoke.md + results.json. Never prints the key."""

import argparse
import asyncio
import json
import math
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path

import httpx

WORDS = ["the", "a", "to", "of", "and", "in", "that", "is", "for", "it", "on", "you", "with", "as", "be", "at", "this", "have", "from", "or", "by", "not", "but", "what", "all", "were", "when", "we", "there", "can", "an", "your", "which", "their", "said", "if", "do", "will", "each", "about", "how", "up", "out", "them", "then", "she", "many", "some", "so", "these", "would", "other", "into", "has", "more", "her", "two", "like", "him", "see", "time", "could", "no", "make", "than", "first", "been", "its", "who", "now", "people", "my", "made", "over", "did", "down", "only", "way", "find", "use", "may", "water", "long", "little", "very", "after", "words", "called", "just", "where", "most", "know", "get", "through", "back", "much", "before", "go", "good", "new", "write", "our", "used", "me", "man", "too", "any", "day", "same", "right", "look", "think", "also", "around", "another", "came", "come", "work", "three", "word", "must", "because", "does", "part", "even", "place", "well", "such", "here", "take", "why", "things", "help", "put", "years", "different", "away", "again", "off", "went", "old", "number", "great", "tell", "men", "say", "small", "every", "found", "still", "between", "name", "should", "home", "big", "give", "air", "line", "set", "own", "under", "read", "last", "never", "us", "left", "end", "along", "while", "might", "next", "sound", "below", "saw", "something", "thought", "both", "few", "those", "always", "looked", "show", "large", "often", "together", "asked", "house", "world", "going", "want", "school", "important", "until", "form", "food", "keep", "children", "feet", "land", "side", "without", "boy", "once", "animals", "life", "enough", "took", "sometimes", "four", "head", "above", "kind", "began", "almost", "live", "page", "got", "earth", "need", "far", "hand", "high", "year", "mother", "light", "parts", "country", "father", "let", "night", "following", "picture", "being", "study", "second", "eyes", "soon", "times", "story", "boys", "since", "white", "days", "ever", "paper", "hard", "near", "sentence", "better", "best", "across", "during", "today", "others", "however", "sure", "means", "knew", "try", "told", "young", "miles", "sun", "ways", "thing", "whole", "hear", "example", "heard", "several", "change", "answer", "room", "sea", "against", "top", "turned", "learn", "point", "city", "play", "toward", "five", "using", "himself", "usually"]
NAMES = ["reply", "memory", "persona", "topic", "greet", "joke", "fact", "recall", "answer", "question", "mood", "weather", "hobby", "food", "music", "travel", "pet", "job", "family", "plan", "weekend"]
TOOLS = [{"type": "function", "function": {"name": n, "description": d, "parameters": {
    "type": "object", "properties": {k: {"type": t} for k, t in ps}, "required": [ps[0][0]]}}}
    for n, d, ps in (
        ("read_file", "Read a workspace file, or some of its lines.",
         (("path", "string"), ("start", "integer"), ("end", "integer"))),
        ("grep_file", "Find the lines matching a pattern in a file or a package.",
         (("path", "string"), ("pattern", "string"))),
        ("edit_file", "Replace the exact text `old` with `new` in one of your files.",
         (("path", "string"), ("old", "string"), ("new", "string"))),
        ("run_bot", "Run a bot on one conversation.", (("path", "string"), ("lines", "array"))),
        ("add_probe", "Add a probe best fails.", (("lines", "array"), ("check", "string"),
                                                  ("evidence", "string"))),
        ("promote_best", "Promote your bot if it beats best.", (("path", "string"),
                                                                 ("reason", "string"))),
        ("post", "Post on the board.", (("text", "string"), ("thread", "string"))),
        ("search", "Search the board channel.", (("query", "string"),)),
        ("done", "End this turn with a one-line summary.", (("summary", "string"),)))]
for _t in TOOLS:  # array params need items
    for _p in _t["function"]["parameters"]["properties"].values():
        if _p["type"] == "array":
            _p["items"] = {"type": "string"}


def corpus(seed: int, chars: int) -> str:
    """Deterministic text with the run's texture: Python lines, chat lines, board posts."""
    rng, out, n = random.Random(seed), [], 0
    while n < chars:
        k = rng.random()
        if k < 0.45:
            name = rng.choice(NAMES)
            line = (f"def {name}_{rng.randint(0, 99)}(line):\n    if '{rng.choice(WORDS)}' in "
                    f"line.lower():\n        return '{' '.join(rng.choices(WORDS, k=9))}'\n")
        elif k < 0.8:
            line = f"user: {' '.join(rng.choices(WORDS, k=12))}\nbot: {' '.join(rng.choices(WORDS, k=14))}\n"
        else:
            line = (f"[{rng.choice(NAMES)}] agent-{rng.randint(0, 299):04d}: GREEN agents/agent-"
                    f"{rng.randint(0, 299):04d}/bot: {' '.join(rng.choices(WORDS, k=10))}\n")
        out.append(line)
        n += len(line)
    return "".join(out)[:chars]


class Shape:
    """The messages of agent `i`'s k-th request at a target prompt size (chars per token
    calibrated)."""

    def __init__(self, prompt_tokens: int, cpt: float = 3.6, shared_tokens: int = 6000,
                 result_tokens: int = 500):
        self.cpt, self.prompt_tokens = cpt, prompt_tokens
        self.shared_tokens = min(shared_tokens, prompt_tokens // 4)
        self.result_tokens = min(result_tokens, prompt_tokens // 8)
        self._cache: dict[tuple, str] = {}

    def _text(self, seed: int, tokens: int) -> str:
        key = (seed, tokens, self.cpt)
        if key not in self._cache:
            self._cache[key] = corpus(seed, max(1, int(tokens * self.cpt)))
        return self._cache[key]

    def messages(self, i: int, k: int) -> list[dict]:
        hist = self.prompt_tokens - self.shared_tokens - self.result_tokens - 300
        call = {"id": f"call_{i}_{k}", "type": "function",
                "function": {"name": "run_bot", "arguments": json.dumps(
                    {"path": f"agents/agent-{i:04d}/bot", "lines": ["hi", "what's my name?"]})}}
        return [{"role": "system", "content": "You are an agent building ONE chatbot with many "
                 "others. Every reply is exactly one tool call.\n" + self._text(0, self.shared_tokens)},
                {"role": "user", "content": "Your history so far:\n"  # stable: cacheable
                 + self._text(1000 + i, max(1, hist))},
                {"role": "assistant", "content": None, "tool_calls": [call]},
                {"role": "tool", "tool_call_id": call["id"],
                 "content": f"[call {k}]\n" + self._text(10 ** 6 + i * 10 ** 4 + k,
                                                          self.result_tokens)}]


def body(model: str, msgs: list[dict], max_tokens: int, thinking: bool) -> dict:
    return {"model": model, "messages": msgs, "tools": TOOLS, "tool_choice": "required",
            "max_tokens": max_tokens, "temperature": 0.6,
            "chat_template_kwargs": {"thinking": thinking}}


# ---------------------------------------------------------------------- /metrics
METRICS = {"running": ("vllm:num_requests_running",),
           "waiting": ("vllm:num_requests_waiting",),
           "kv": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
           "hits": ("vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total",
                    "vllm:prefix_cache_hits", "vllm:gpu_prefix_cache_hits"),
           "queries": ("vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries_total",
                       "vllm:prefix_cache_queries", "vllm:gpu_prefix_cache_queries"),
           "hit_rate": ("vllm:gpu_prefix_cache_hit_rate",)}


def parse_metrics(text: str) -> dict[str, float]:
    """Prometheus text -> {running, waiting, kv, hits, queries, hit_rate} (summed over labels;
    the first known name per key; missing keys left out)."""
    vals: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z_:][\w:]*)(?:\{[^}]*\})?\s+([-+0-9.eE]+|NaN)", line)
        if m and m.group(2) != "NaN":
            vals[m.group(1)] = vals.get(m.group(1), 0.0) + float(m.group(2))
    out = {}
    for key, names in METRICS.items():
        for n in names:
            if n in vals:
                out[key] = vals[n]
                break
    return out


async def poll_metrics(c: httpx.AsyncClient, url: str, samples: list, stop: asyncio.Event,
                       every: float) -> None:
    while not stop.is_set():
        try:
            r = await c.get(url, timeout=5)
            if r.status_code == 200:
                samples.append((time.time(), parse_metrics(r.text)))
        except httpx.HTTPError:
            pass
        try:
            await asyncio.wait_for(stop.wait(), every)
        except TimeoutError:
            pass


def metric_summary(samples: list) -> dict:
    if not samples:
        return {}
    col = lambda k: [s[k] for _, s in samples if k in s]
    out = {}
    if run := col("running"):
        out["running_max"] = max(run)
    if wait := col("waiting"):
        out["waiting_max"], out["waiting_mean"] = max(wait), statistics.fmean(wait)
    if kv := col("kv"):
        kv = [x * 100 if x <= 1 else x for x in kv]
        out["kv_max_pct"], out["kv_mean_pct"] = max(kv), statistics.fmean(kv)
    h, q = col("hits"), col("queries")
    if len(h) >= 2 and len(q) >= 2 and q[-1] > q[0]:
        out["prefix_hit_pct"] = 100 * (h[-1] - h[0]) / (q[-1] - q[0])
    elif rate := col("hit_rate"):
        out["prefix_hit_pct"] = 100 * statistics.fmean(rate)
    return out


# ---------------------------------------------------------------------- the load
def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, math.ceil(p * len(xs)) - 1))] if xs else float("nan")


async def level(c: httpx.AsyncClient, url: str, model: str, shape: Shape, conc: int,
                seconds: float, max_tokens: int, thinking: bool, metrics_url: str | None,
                metrics_every: float) -> dict:
    recs, stop, samples = [], asyncio.Event(), []
    t_start = time.time()
    warm = min(30.0, 0.2 * seconds)

    async def client(i: int) -> None:
        k = 0
        while time.time() - t_start < seconds:
            k += 1
            t0 = time.time()
            rec = {"agent": i, "t0": t0}
            try:
                r = await c.post(url, json=body(model, shape.messages(i, k), max_tokens, thinking))
                rec["status"] = r.status_code
                if r.status_code == 200:
                    d = r.json()
                    u = d.get("usage") or {}
                    msg = (d.get("choices") or [{}])[0].get("message") or {}
                    rec |= {"in": u.get("prompt_tokens"), "out": u.get("completion_tokens"),
                            "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                            "tool_call": bool(msg.get("tool_calls"))}
            except httpx.HTTPError as e:
                rec |= {"status": 0, "error": type(e).__name__}
            rec["t1"] = time.time()
            if rec["t1"] - t_start <= seconds:  # cut-off calls are not counted
                recs.append(rec)

    poller = (asyncio.create_task(poll_metrics(c, metrics_url, samples, stop, metrics_every))
              if metrics_url else None)
    tasks = [asyncio.create_task(client(i)) for i in range(conc)]
    await asyncio.sleep(seconds)
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    stop.set()
    if poller:
        await poller
    kept = [r for r in recs if r["t0"] - t_start >= warm]
    span = max(1e-9, seconds - warm)
    ok = [r for r in kept if r.get("status") == 200]
    lat = [r["t1"] - r["t0"] for r in ok]
    ins = [r["in"] for r in ok if r.get("in")]
    cached = [r["cached"] for r in ok if r.get("cached") is not None]
    return {"agents": conc, "calls": len(kept), "req_s": round(len(ok) / span, 3),
            "p50_s": round(pct(lat, 0.5), 3), "p95_s": round(pct(lat, 0.95), 3),
            "errors": len(kept) - len(ok),
            "err_pct": round(100 * (len(kept) - len(ok)) / max(1, len(kept)), 2),
            "in_mean": round(statistics.fmean(ins)) if ins else None,
            "out_mean": round(statistics.fmean([r["out"] for r in ok if r.get("out")]), 1)
            if any(r.get("out") for r in ok) else None,
            "cached_pct": round(100 * sum(cached) / max(1, sum(ins)), 1) if cached else None,
            "tool_call_pct": round(100 * sum(r.get("tool_call", False) for r in ok)
                                   / max(1, len(ok)), 1),
            **{k: round(v, 2) for k, v in metric_summary(samples).items()}}


def knee(rows: list[dict]) -> tuple[int | None, str]:
    """The last good level and why the next one is not (None: the first level is bad)."""
    for prev, cur in zip([None, *rows], rows):
        why = ("errors" if cur["err_pct"] > 1 else
               "queueing" if cur.get("waiting_mean", 0) >= 1 else
               "flat throughput, rising latency" if prev and prev["req_s"] > 0
               and cur["req_s"] < 1.15 * prev["req_s"] and cur["p50_s"] >= 1.4 * prev["p50_s"]
               else "")
        if why:
            return (prev["agents"] if prev else None), f"at {cur['agents']}: {why}"
    return rows[-1]["agents"] if rows else None, "no knee up to the last level"


def recommend(rows: list[dict], tool_s: float) -> dict:
    k, why = knee(rows)
    if k is None:
        return {"knee": None, "why": why, "agents": 0, "first_wave": 0}
    row = next(r for r in rows if r["agents"] == k)
    duty = row["p50_s"] / (row["p50_s"] + tool_s) if row["p50_s"] == row["p50_s"] else 1.0
    agents = int(0.8 * k / max(duty, 1e-3))
    return {"knee": k, "why": why, "duty": round(duty, 3), "agents": agents,
            "first_wave": min(20, max(1, agents // 5)),
            "at_least": why == "no knee up to the last level"}


def table(rows: list[dict]) -> str:
    cols = [("agents", "C"), ("req_s", "req/s"), ("p50_s", "p50 s"), ("p95_s", "p95 s"),
            ("errors", "errors"), ("in_mean", "prompt tok"), ("out_mean", "out tok"),
            ("cached_pct", "cached %"), ("running_max", "running max"),
            ("waiting_max", "waiting max"), ("kv_max_pct", "KV max %"),
            ("prefix_hit_pct", "prefix hit %")]
    lines = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join("-" if r.get(k) is None else str(r[k]) for k, _ in cols)
                     + " |")
    return "\n".join(lines)


async def main_async(a: argparse.Namespace) -> dict:
    base = a.base_url.rstrip("/")
    url = base + "/chat/completions"
    root = base.removesuffix("/v1")
    head = {"Authorization": f"Bearer {a.api_key}"} if a.api_key else {}
    limits = httpx.Limits(max_connections=max(a.levels) + 8, max_keepalive_connections=max(a.levels) + 8)
    async with httpx.AsyncClient(headers=head, timeout=a.timeout, limits=limits) as c:
        metrics_url = None
        if not a.no_metrics:
            try:
                r = await c.get(root + "/metrics", timeout=5)
                metrics_url = root + "/metrics" if r.status_code == 200 and "vllm:" in r.text else None
            except httpx.HTTPError:
                pass
        print(f"metrics: {'reachable' if metrics_url else 'not reachable'} ({root}/metrics)",
              flush=True)
        shape = Shape(a.prompt_tokens)
        r = await c.post(url, json=body(a.model, shape.messages(0, 0), a.max_tokens, a.thinking))
        if r.status_code != 200:
            raise SystemExit(f"calibration request failed: HTTP {r.status_code}: {r.text[:300]}")
        got = (r.json().get("usage") or {}).get("prompt_tokens")
        if got:  # chars per token from the server's own count
            shape = Shape(a.prompt_tokens, cpt=max(1.0, min(8.0, shape.cpt * a.prompt_tokens / got)))
        print(f"calibrated: {got} prompt tokens at {Shape(a.prompt_tokens).cpt} chars/token -> "
              f"{shape.cpt:.2f}", flush=True)
        rows = []
        for conc in a.levels:
            row = await level(c, url, a.model, shape, conc, a.minutes * 60, a.max_tokens,
                              a.thinking, metrics_url, a.metrics_every)
            rows.append(row)
            print(json.dumps(row), flush=True)
            if row["err_pct"] > 50:
                print(f"stopping: {row['err_pct']} % errors at {conc}", flush=True)
                break
    return {"rows": rows, "recommend": recommend(rows, a.tool_s), "metrics": bool(metrics_url),
            "model": a.model, "prompt_tokens": a.prompt_tokens, "max_tokens": a.max_tokens,
            "minutes": a.minutes, "chars_per_token": round(shape.cpt, 3)}


def report(res: dict) -> str:
    rec = res["recommend"]
    head = (f"# Load smoke: {res['model']}\n\n{res['prompt_tokens']}-token prompts, "
            f"{res['max_tokens']}-token outputs, tool_choice required, {res['minutes']} min per "
            f"level; /metrics {'read' if res['metrics'] else 'not reachable'}.\n\n")
    if rec["knee"] is None:
        tail = f"\n\nNo usable level ({rec['why']}): fix the endpoint before the run.\n"
    else:
        tail = (f"\n\nKnee: {rec['knee']} concurrent calls ({rec['why']}). Duty "
                f"{rec['duty']} (p50 / (p50 + tool time)).\n"
                f"Recommended for one node: agent cap {'>= ' if rec['at_least'] else ''}"
                f"{rec['agents']}, first wave {rec['first_wave']}.\n")
    return head + table(res["rows"]) + tail


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    ap.add_argument("--model", default=os.environ.get("MODEL_WORKER"))
    ap.add_argument("--levels", default="8,16,32,64,128",
                    type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--minutes", type=float, default=3.0)
    ap.add_argument("--prompt-tokens", type=int, default=46000)
    ap.add_argument("--max-tokens", type=int, default=135)
    ap.add_argument("--thinking", action="store_true", help="chat_template_kwargs thinking true")
    ap.add_argument("--tool-s", type=float, default=2.0,
                    help="seconds an agent spends between model calls (tools, harness)")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--metrics-every", type=float, default=5.0)
    ap.add_argument("--no-metrics", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("load-smoke-out"))
    a = ap.parse_args(argv)
    if not a.base_url or not a.model:
        ap.error("set OPENAI_BASE_URL and MODEL_WORKER (or --base-url, --model)")
    a.api_key = os.environ.get("OPENAI_API_KEY", "")
    res = asyncio.run(main_async(a))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    text = report(res)
    (a.out / "load-smoke.md").write_text(text)
    print(text, flush=True)
    return 0 if res["recommend"]["knee"] else 1


if __name__ == "__main__":
    sys.exit(main())
