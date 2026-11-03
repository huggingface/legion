# Load smoke: how many agents keep the endpoint at its knee

Goal: maximum use of a fixed-cost deployment without queueing agents behind it. More agents
mean more model calls per second until the server saturates; past that point (the knee) each
extra agent mostly adds latency. The method is the one from the earlier GPU knee probe: step
the concurrency by doubling, and the knee is the level before the first doubling that brings
under +15% calls/s while p50 latency rises at least 40% (or that brings errors).

## What it runs

`load_smoke.py` runs the real phase-1 task against the endpoint, so the load has the run's
shape (prompt sizes, tool calls, `max_turns` 12, continuous `every: turn_end`):

- for each level C in `--levels` (default 8, 16, 32, 64): C agents in C/4 teams
  (`render.py --teams`), `flock.controller.localrun` in local mode, the chatbot2 world on this machine, an
  in-memory board (`tests/fake_msgboard.py`), for `--minutes` (default 5) per level;
- between flock and the endpoint, a local proxy records every call (start, end, status,
  tokens);
- per level: model calls/s, p50 and p90 latency, errors and output tokens/s over the window
  after `--warmup` (default 60 s);
- output: a table, the knee, and the recommended agent count = knee x 0.8, in
  `<out>/load-smoke.md` and `<out>/results.json`, plus each level's raw call log.

## Run it

```sh
export SELFHOSTED_BASE_URL=https://<your endpoint>/v1   # OpenAI-compatible (vLLM, sglang)
export SELFHOSTED_API_KEY=<key>                          # sent as the bearer token
export MODEL_DS=<served model id>                        # MODEL_GLM=<id> too, for the 2-model mix
uv run python experiments/chatbot-v2/selfhosted/load_smoke.py --levels 8,16,32,64 --minutes 5 \
  --out load-smoke-out
```

About 25 min for the 4 levels. It needs flock installed (the repo's env), and nothing else on
the network than the endpoint. The model calls are real; the board and the world are local.

## Read the result

| agents | teams | calls/s | p50 ms | p90 ms | errors | out tok/s |
|---|---|---|---|---|---|---|
| ... | | | | | | |

- **Knee found**: set `teams` x `per_team` in `scale.yaml` close to the recommended count
  (e.g. 48 agents: 12 teams of 4, or 6 teams of 8).
- **No knee up to 64**: the endpoint keeps up; repeat with `--levels 64,128,256`.
- **Errors at low C**: check the server's `--max-num-seqs`, the context length (prompts reach
  the 64k compaction threshold), and the tool-call parser.
- Server side, watch running/waiting requests and KV use: a knee with low KV and nothing
  waiting points at the API frontend (vLLM `--api-server-count`), as in the earlier probe.
