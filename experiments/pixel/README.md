# pixel: canvas world service for the flock pixel-painting test drive

~100 flock agents (later 1k, 10k), each running a small model, repaint a 1024x1024 target image
(`assets/target.png`) on a shared blank canvas, a few pixels per call, from a shared claim queue.
This directory is the world: a FastAPI + numpy service, the `canvas` CLI the agents call, a replay
page, a load generator and tests. Flock core is untouched; nothing here launches jobs or models.

```
experiments/pixel/
  canvas/      service (config, world = numpy state + log + replay, auth, FastAPI app)
  cli/         `canvas` CLI (stdlib only; put cli/ on PATH)
  replay/      static replay page, served at /replay
  tests/       pytest (TestClient + one real-uvicorn CLI test)
  loadgen.py   scripted painters at N x R calls/s
  Dockerfile, run_in_job.sh
```

## Run

```sh
cd experiments/pixel
export UV_PROJECT_ENVIRONMENT=$AM_LOCAL/envs/flock-pixel
uv sync && uv run pytest -q && uv run ruff check .
CANVAS_DATA=/tmp/canvas uv run python -m canvas          # :8765, replay at /replay
CANVAS_URL=http://127.0.0.1:8765 FLOCK_AGENT=me cli/canvas claim
```

Inside an HF Job next to vLLM (plain processes): `sh experiments/pixel/run_in_job.sh vllm serve <model> --port 8000`.
It pip-installs four deps, starts the canvas in the background, waits for `/healthz`, then execs vLLM.
The `Dockerfile` is the same service as an image (`CMD python -m canvas`, `cli/` on PATH).

## Configuration (env)

| var | default | meaning |
|---|---|---|
| `CANVAS_TARGET` | `assets/target.png` | target image (RGB or RGBA) |
| `CANVAS_BACKGROUND` | `000000` | colour that alpha=0 target pixels become (partial alpha is composited over it) |
| `CANVAS_BLANK` | = background | initial canvas colour |
| `MAX_PIXELS_PER_CALL` | **10** | hard cap per `paint` call (1 works) |
| `CANVAS_MAX_CLAIM` | = `MAX_PIXELS_PER_CALL` | max pixels per `claim` |
| `CANVAS_CLAIM_TTL` | 120 | lease seconds |
| `CANVAS_CLAIM_MODE` | `advisory` | `enforced` refuses paints on pixels leased by someone else |
| `CANVAS_CLAIM_REVEAL` | 1 | claim response includes each pixel's target colour |
| `CANVAS_RATE_LIMIT` / `CANVAS_RATE_BURST` | 2 / 5 | per-agent token bucket on write calls (0 = off) |
| `CANVAS_TOKEN` | unset | mode A shared secret |
| `CANVAS_ADMIN_TOKEN` | unset | mode B: mints per-agent tokens |
| `CANVAS_READ_AUTH` | 0 | 1 = reads need a valid token too |
| `CANVAS_DATA` | unset (memory only) | data dir: log, claims snapshot, agents, canvas.png |
| `CANVAS_KEYFRAME_EVERY` / `_CACHE` | 1000 / 32 | replay keyframe spacing / LRU size |
| `CANVAS_SNAPSHOT_SECONDS` | 30 | claims snapshot + fsync interval |
| `CANVAS_HOST` / `CANVAS_PORT` | 0.0.0.0 / 8765 | bind |

**Background decision.** The shipped `target.png` is RGB (no alpha) with a black background
(466,409 px = 44.5%). Transparent pixels in an RGBA target become `CANVAS_BACKGROUND` (black). The
canvas starts as that colour, so background pixels already match: they are never handed out by
`claim`, and painting them correctly counts as `duplicate`. The work set is the 582,167 pixels that
differ from blank. Set `CANVAS_BLANK=ffffff` to make the background part of the job.

## API

All bodies are JSON. Errors are `{"error": "<short actionable message>"}` with 400/401/403/429.

| endpoint | notes |
|---|---|
| `GET /target/info` | size, background, blank, colour count, top 16 colours, work pixels, caps |
| `GET /target/region?x&y&w&h&format=json\|png\|raw` | json = rows of space-separated hex (max 4096 px); raw = RGB bytes |
| `GET /target/pixel?x&y` | `{hex,r,g,b,current,ok}` |
| `POST /claim {n?}` | up to `n` (default/cap 10) pixels that do not match and are not leased: `{pixels:[{x,y,hex}], runs:[{x,y,len}], lease_s, expires, remaining, done}` |
| `POST /release {pixels?}` | drop your leases (all, or the listed ones) |
| `POST /paint {pixels:[{x,y,hex}\|{x,y,r,g,b}], turn?}` | cap enforced; per-pixel `results[{x,y,outcome,want?}]`, `counts`, `seq`, running `score` |
| `GET /region?x&y&w&h&format=` | live canvas, same formats as target |
| `GET /canvas.png`, `GET /target.png` | full images |
| `GET /score?agents=true` | `exact_match_pct`, `coverage_pct`, `painted_pct`, `matched`, `todo`, `mse`, `psnr`, paint totals, per-agent counters |
| `GET /status` | uptime, seq, agents seen/active, claims outstanding, req/s (10 s), config, auth mode |
| `GET /log?after=<seq>&limit<=5000` | paint log, JSONL |
| `GET /replay` | replay page; `GET /replay/frame?t=<seq>` = PNG of the canvas after entry t |
| `POST /agents/register {name}` | admin token only; returns `{name, token}` (mode B) |

Names I changed or added vs the brief: paint pixels accept `hex` as well as `r,g,b`;
`/target.png` (for the replay page) and `/healthz` are extra.

**Outcomes** (swarmlab vocabulary, evaluated in commit order against the canvas before that pixel):
`useful` right colour, pixel was wrong; `duplicate` right colour, already right; `wrong` wrong colour,
pixel was already wrong; `overwritten` wrong colour on a pixel that was right (destroyed work);
`refused` enforced mode only, pixel leased by another agent, not applied. Each paint is exactly one
of these, so counters add up to pixels painted. `wrong`/`overwritten` results carry `want`.

**Scores.** `exact_match_pct` = canvas == target over all pixels (starts at 44.5% because of the
background). `coverage_pct` = exact match over the work set only (starts at 0; the progress number).
`painted_pct` = pixels touched at least once. `mse` over RGB channels (0..255), `psnr` in dB (null
at a perfect match). Matched count and SSE are kept incrementally per pixel, so `/score` is O(1)
apart from the coverage mask sum.

**Claims.** One lease set per agent: a new `claim` drops your previous leases (no hoarding). The
queue is a row-major cursor over pixels that are not matching and not leased; it wraps, so pixels
painted wrong come back around. Leases expire after the TTL (crashed agents leak nothing), and a
correct paint clears the lease. Advisory mode (default) only steers work; `enforced` refuses.

## Auth

Token header: `X-Canvas-Token: <tok>` or `Authorization: Bearer <tok>` (first wins).
`X-Canvas-Token` exists for the HF jobs proxy, where `Authorization` carries the caller's HF token.

- **Mode A, shared secret** (`CANVAS_TOKEN`): every agent uses the same token; identity is the
  `X-Flock-Agent` header (`FLOCK_AGENT`). Simple; agents can impersonate each other.
- **Mode B, per-agent** (`CANVAS_ADMIN_TOKEN`): the operator calls `POST /agents/register {name}`
  with the admin token for each agent and injects the returned `cv_...` token as that agent's
  `CANVAS_TOKEN`. Identity = token; `X-Flock-Agent` is ignored. Only SHA-256 hashes are stored
  (`agents.json`); re-registering a name rotates its token.
- Both can be on together. With neither set the service is open (dev) and trusts `X-Flock-Agent`.
- Reads are open unless `CANVAS_READ_AUTH=1`. The replay page takes `?token=` and sends it as
  `X-Canvas-Token`.

## Paint log and replay

The log (`$CANVAS_DATA/paint.log.jsonl`) is append-only and the source of truth: on start the canvas
and all paint counters are rebuilt by folding over it (a torn last line is truncated). One line per
`paint` call:

```json
{"seq":3000,"ts":1792415922.6018,"agent":"load-00099","turn":"t-41","req":"4ca656a36bad49ba",
 "px":[[572,195,"ff9d00","useful"],[573,195,"ff9d00","useful"]]}
```

Join with flock's per-agent event JSONL on `agent` + `turn` (from `FLOCK_TURN` / `X-Flock-Turn` /
body `turn`), or exactly on `req` (the CLI sends `X-Request-Id` and prints it with `--json`), with
`ts` (unix seconds) as the fallback. Claims are not logged (soft state, snapshotted to
`claims.npz` every 30 s and on shutdown).

Replay frames: the service keeps a columnar index of every applied paint (flat index + RGB, ~7 B
per pixel) and per-entry offsets. `frame(t)` = nearest keyframe at or below t (every 1000 entries,
LRU of 32, built lazily from the previous cached one) + a vectorised last-write-wins fold of the
delta. Frames at t <= seq never change, so they are served cacheable. Measured: 12-23 ms per frame
at 21k entries. The page (`/replay`) has a slider over seq, play at 1-10,000 entries per tick, the
target side by side, and the log entries around t (agent, turn, req, ts, per-pixel colour chip and
outcome; click to jump). `?t=` deep-links.

## Load test (local, single uvicorn process, loadgen on the same 8-core box)

`loadgen.py` workers loop claim -> paint (10 px), each HTTP call paced at R/s, keep-alive.
Default server settings (rate limit 2/s/agent) for the 100 x 1 run; rate limit off for the others.

| run | achieved | p50 | p95 | p99 | errors |
|---|---|---|---|---|---|
| 100 workers x 1 call/s, 60 s (the brief) | **100.0 req/s** (6,099 calls, 30,000 px) | 1.2 ms | **1.8 ms** | 23 ms | 0 |
| 100 x 5/s, 20 s | 499.6 req/s | 1.0 ms | 1.6 ms | 3.0 ms | 0 |
| 200 x 10/s, 20 s (saturation) | 1,086 req/s of 2,000 offered | 173 ms | 219 ms | 489 ms | 0 |

So one process is ~10x the 100 req/s target before queueing. Restart reload: 15.9k log entries in
1.0 s (~65 us/entry; a full 1024x1024 run at 10 px/call is ~58k entries, ~4 s). Log size ~380 B per
10-px entry (~22 MB for a full run). At 100 agents x 1 call/s alternating claim/paint, 10 px per
paint, the 582k-pixel work set takes ~20 min of pure painting.

## Scaling to 10k req/s (notes, not implemented)

1. One Python process tops out ~1k req/s (GIL, JSON, ASGI). Shard by rows: K processes each own
   H/K rows with their own log/claims; the CLI (or a thin router) maps y -> shard; claim picks a
   shard by agent hash. Score = sum of shard counters; replay merges shard logs by `ts`.
2. Group commit: buffer log lines and flush every ~10 ms instead of per call; fsync on the timer.
3. Cheaper wire: msgspec/orjson, or a binary `paint` body; `px` as packed arrays.
4. Claims at 10k agents: per-shard free lists of row runs instead of a mask scan (the scan is
   O(remaining) and fine at 100 agents, ~1 ms worst case on 1M pixels).
5. Bigger per-call caps are the cheapest lever: requests scale as pixels / cap.
6. `/score?agents=true` and `/status` are O(agents); paginate per-agent counters at 10k.

## Painter instructions (for the agent prompt)

```
You are one of many agents painting a shared 1024x1024 picture. Use only the `canvas` command.
Repeat this loop until you are told you are done:
1. Run `canvas claim`. It leases up to 10 pixels to you for 2 minutes and prints a
   ready-made `canvas paint ...` line with each pixel's target colour.
2. Run that exact `canvas paint X,Y,RRGGBB ...` line (at most 10 pixels per call).
3. Read the result. `useful` = good. `duplicate` = it was already right.
   `wrong` or `overwritten` = you sent a wrong colour; the line shows the right one, repaint it.
4. Go back to step 1. When claim says "nothing to claim ... You are done", stop.
Rules:
- Only paint pixels you claimed. The black background is already done; do not paint it.
- Colours are 6 hex digits like ff9d00. x is the column (0-1023), y is the row (0-1023).
- "rate limited": wait the seconds it says, then run the same command again.
- Any other error says what to fix. Fix it; never repeat a failing command unchanged.
- `canvas get X Y` shows one pixel's target colour; `canvas score` shows overall progress.
- Keep tool output short: do not dump regions or the log.
Env (set by the operator): CANVAS_URL, CANVAS_TOKEN, FLOCK_AGENT, FLOCK_TURN.
```

## CLI

`canvas claim [-n N] | get X Y [W H] | paint X,Y,RRGGBB ... | release | score | status | info`,
`--json` for machine output. Reads `CANVAS_URL` (default `http://127.0.0.1:8765`), `CANVAS_TOKEN`,
`FLOCK_AGENT`, `FLOCK_TURN`, and `CANVAS_PROXY_TOKEN` (sent as `Authorization: Bearer` when going
through the HF jobs proxy; the canvas token then rides in `X-Canvas-Token`). Stdlib `urllib` only, so
agents need no install. Retries 429 up to 3 times (honouring `Retry-After`, max 5 s). Exit codes:
0 ok, 1 request/usage error, 2 auth, 3 unreachable.

## Open questions

- Claim reveals target colours (`CANVAS_CLAIM_REVEAL=1`), making each pixel 2 calls with no
  `get`. Turn it off to make agents read the target (more calls, more "agentic")?
- Advisory vs enforced claims for the first run (advisory measures natural collisions).
- HF jobs proxy: which port it exposes and whether it strips `Authorization` before forwarding;
  the replay page in a browser cannot add the HF token header (may need `?token=` on the proxy side).
- Black background is pre-done (44.5%). Should the background be part of the work?
- Log claims too (for a fuller replay of agent behaviour), at ~2x log size?
