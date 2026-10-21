# Pixel run kit (first 100-agent run)

Nothing here has been launched. Files:

| file | what |
|---|---|
| `painter.md` | agent instructions (19 lines): install step, then one chained command per prompt |
| `spec-100.yaml`, `spec-smoke.yaml` | flock specs: 100 painters on 2 cpu-basic workers (+1 spare); smoke = 3 painters, 1 worker |
| `driver.py` | keeps painters prompted until the canvas is done; writes `progress.jsonl`, `prompts.jsonl` |
| `spec-spawn.yaml`, `leader.md`, `board.py`, `job-spawn.sh` | experiment 2: 10 Sonnet leaders x 100 painters on leased bands (below) |
| `spec-1k.yaml`, `job-1k.sh` | the 1k run: 3 painter types x 3 providers, 3 rtx-pro-6000 jobs (below) |
| `job.sh` | the single GPU job: vLLM (LFM2.5-2.6B, :8000) + canvas (:8765) in one container |

## How a painter works

`tool_choice: required` (PR #7) makes every flock prompt exactly one forced bash call, so the
**driver is the loop**. Each prompt carries one literal command:

1. First prompt per agent (install the stdlib-only CLI from the canvas, `GET /cli/canvas`):
   `python3 -c "import sys,urllib.request as u;u.urlretrieve(*sys.argv[1:])" $CANVAS_URL/cli/canvas canvas && python3 canvas status`
2. Every later prompt: `python3 canvas claim 10 --json | python3 canvas paint - --json`
   (`paint -` paints exactly what the claim returned; an empty claim prints "You are done").

The worker image has `python3` but no `curl`, hence the python one-liner. The agent's tool env
gets `CANVAS_URL`, `CANVAS_TOKEN` (type secrets) and `FLOCK_AGENT` (worker), which the CLI reads.
If an agent stepped 3 times without ever reaching the canvas, the driver re-sends the install.

## Launch order (run 1: jobs-proxy path)

The model and the canvas are reached through the HF jobs proxy; network groups are not used (flock
dropped provider `network_group` on purpose, 0c1b4b2; the probe in
`flock-board/probe-reachability-lfm.md` showed `Bearer $HF_TOKEN` through the proxy -> 200 for
chat completions at the same latency, and the controller's preflight can reach it too).

1. **GPU job.** Run `RUN_ID=run1-20261019 experiments/pixel/run/job.sh`. You need `HF_TOKEN` and `CANVAS_TOKEN` set locally.
   - Defaults: `rtx-pro-6000`, `--timeout 90m` (the smoke used `FLAVOR=l4x1`). The vLLM command sits in a
     marked "LAUNCH RECIPE" block (contract: API on :8000, logs `/data/vllm*.log`, pids in `VLLM_PIDS`).
   - vLLM is started with `--expose 8000`. The proxy protects it with the HF token, so vLLM itself runs **without** `--api-key`.
   - The canvas is started with `--expose-public 8765`. Reads are open; writes need `CANVAS_TOKEN`.
   - Image: `vllm/vllm-openai:v0.31.0` (the probe's image), with the canvas deps pip-installed on top; the
     command overrides the image's `vllm serve` entrypoint. `VLLM_USE_FLASHINFER_SAMPLER=0` is set too.
     The first smoke (2026-10-19) used the uv image + `pip install vllm` and died at startup: FlashInfer's
     sampler JIT needs `nvcc` ("Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist").
   - The script prints the job id and the exposed URLs.
2. **Fill in the URLs.** Confirmed at the first smoke: `hf jobs run` prints them, and
   `hf jobs inspect --json cmpatino/<JOB_ID>` has them in `status.expose_urls` (plain `inspect` prints a table).
   - model: `https://<JOB_ID>--8000.hf.jobs/v1`. Put it in `base_url` in `spec-smoke.yaml` / `spec-100.yaml`, replacing `<JOB_ID>`.
   - canvas: `https://<JOB_ID>--8765.hf.jobs`. This is `CANVAS_URL`.
3. **Wait until healthy.**
   - `hf jobs logs <JOB_ID>` shows `VLLM READY`.
   - `curl -H "Authorization: Bearer $HF_TOKEN" https://<JOB_ID>--8000.hf.jobs/v1/models` returns 200.
   - `curl https://<JOB_ID>--8765.hf.jobs/status` answers with no token.
4. **Controller:** the Space `cmpatino/flock-smoke` (it exists and is paused).
   - Deploy it from the **`llm-harness`** branch, because the worker wheel needs `harness: llm`. In a checkout of that branch:
     `W=$(uv run scripts/publish-wheel.sh hf://buckets/cmpatino/flock/dist)` then
     `uv run scripts/deploy-space.sh "$W" cmpatino/flock-smoke`.
   - `deploy-space.sh` needs `HF_TOKEN`, `FLOCK_TOKEN` and `ANTHROPIC_KEY` set. Any value works for `ANTHROPIC_KEY` if no Claude type runs.
   - Restart the Space if it stays paused.
   - Add Space secrets `CANVAS_URL` and `CANVAS_TOKEN`. Every secret a spec names must be in the controller's env. For example: `python -c "from huggingface_hub import HfApi;import os;a=HfApi();[a.add_space_secret('cmpatino/flock-smoke',k,os.environ[k]) for k in ('CANVAS_URL','CANVAS_TOKEN')]"`.
   - **HF_TOKEN for the proxy:** the provider uses `api_key_env: HF_TOKEN`, and the controller's `HF_TOKEN` secret is what workers send to the proxy.
     - The brief suggests a fine-grained read-only token under that name.
     - But workers also use `HF_TOKEN` for the wheel and **bucket sync (write)**, so a read-only `HF_TOKEN` would break that. Keep the operator token for run 1.
     - Or put the read-only token in a separate secret (e.g. `HF_PROXY_TOKEN`) and point `api_key_env` at it.
5. **Smoke.** Point `FLOCK_URL` at the Space and set `FLOCK_TOKEN`.
   - Run `flock apply experiments/pixel/run/spec-smoke.yaml`. Preflight probes the proxy URL.
   - Then `python experiments/pixel/run/driver.py --swarm pixel-smoke --canvas-url https://<JOB_ID>--8765.hf.jobs --max-steps 5`.
   - Check `/replay` and the progress Space.
   - Then `flock remove pixel-smoke`.
6. **Run.**
   - Run `flock apply experiments/pixel/run/spec-100.yaml`.
   - Then run `python experiments/pixel/run/driver.py --swarm pixel-100 --canvas-url <canvas URL> --run-id run1-20261019 --repeat --interval 3 --max-minutes 80 --gpu-usd-per-hour 2.75`, with `SPACE_ID`/`HF_TOKEN` set for the progress page. Use the job's RUN_ID as `--run-id`.
   - `--repeat` goes with `done_tool: true`: the step prompt asks the model to repeat the command within the turn, and only idle agents are prompted (a prompt sent mid-turn would be steered into the running turn). With `done_tool: false`, drop `--repeat`.
   - The driver stops when the work set is 100% done. Cap the run with `--max-minutes`, or each agent with `--max-steps`.
7. **Teardown.**
   - Run `flock remove pixel-100`.
   - Confirm the paint log is in the bucket: `$BUCKET/pixel/runs/$RUN_ID/paint.log.jsonl`. It is copied every 60 s; the driver also uploads `final/<run>/paint-log.jsonl.gz` to the Space.
   - Then `hf jobs cancel <JOB_ID>`, and pause `cmpatino/flock-smoke` again.
   - The log is the replay. A canvas started on it (`CANVAS_DATA=<dir with paint.log.jsonl> python -m canvas`) serves `/replay` locally.

Driver env: `FLOCK_URL`, `FLOCK_TOKEN` (controller), `CANVAS_URL` or `--canvas-url`. Reads on the
canvas need no token. `--gpu-usd-per-hour` (default 0.80 = l4x1) feeds the cost estimate; flock's
own cost for a self-hosted provider is $0.

## Expected numbers (estimates, not measured with the model)

- Each step is one model call (~2.3 s forced tool, probe) plus two canvas calls (~2 ms) and does 10 px.
- The probe measured ~15-20 calls/s on one L4, so 100 painters give ~150-200 px/s.
- The 622,780-px work set then takes **~55-70 min**.
- Per-agent rate is ~1 call per 5-7 s, well under the canvas limit of 2 calls/s per agent.
- GPU cost is ~$1 (l4x1). Workers are 2-3 cpu-basic.

## Needs PR #7 / open items

- `harness: llm`, `tool_choice` and `context_tokens` come from PR #7, so the controller runs from `llm-harness`. The specs validate against PR #7's
  `flock.cli.load_spec`; `main` rejects them.
- **`done_tool`** is a boolean on PR #7's head (`done_tool: done` fails validation).
  - `spec-smoke.yaml`: `false`, so each prompt is one forced bash call and the driver is the loop.
  - `spec-100.yaml`: `true` with `max_turns: 25`: a prompt runs forced bash steps until the model calls
    `done` (or 25 calls), so the big run needs fewer prompts. The driver still decides when painting stops;
    its `--max-steps` counts prompts, not bash steps.
- **No `comms` block** in the spec model, so the instructions go in via `system_prompt_file`.
- **No turn id in the tool env:** flock does not export the prompt or turn id, so `FLOCK_TURN` is
  empty. To join flock events with the paint log, use `prompts.jsonl` (agent, prompt_id, ts)
  against the canvas log's agent + ts.
- **Prompting mode.** The driver prompts per agent by default: every idle or working agent with
  nothing pending gets the next step queued, so painters never wait on the controller's view of
  their state. `--bulk` uses `flock prompt --swarm/--type/--state idle` (PR #8) for the plain step;
  it only reaches idle agents and was ~6x slower in the smoke (0.3 vs 1.8 paint calls/s, 3 agents).
  Either way the install step goes once per agent, individually.
- **Install command has no single quotes.** LFM2.5 over-escaped them in its tool-call JSON
  (`\'`) and 2 of 3 smoke painters got a Python SyntaxError.
- **Provider concurrency:** `max_working: 64` on the provider caps 100 painters at 64 in-flight
  turns. The probe measured the L4 sweet spot at c=32-64 (~15-20 calls/s).
- **Proxy URL format** confirmed: `https://<JOB_ID>--<port>.hf.jobs` (`status.expose_urls`).

## The 1k run (spec-1k.yaml, job-1k.sh)

Same conditions as run 1, except agent count, GPUs and workers (Carlos):
- **Same as run 1:** model and `extra`, `tool_choice: required`, `done_tool: true`, `max_turns: 25`,
  `context_tokens: 3000`, `turn_timeout: 6m`, `painter.md`, 25 agents per cpu-basic worker.
  The canvas keeps its defaults: `MAX_PIXELS_PER_CALL=10`, scattered claims, white blank.
- **Different:** 1,000 painters in 3 types (`painter-a/b/c`, 334/333/333), each on its own
  provider `gpu1..gpu3` (`max_working: 128`). flock has no client-side load balancing across
  providers, so the split is per type. There are 3 rtx-pro-6000 jobs and 14 + 2 workers per type
  (42 at start).
- **Driver differs:** run 1 prompted per agent. At 1k that is 1,000 POSTs per tick, and the Space
  edge already returned 429s at 100 agents. The 1k driver sends **one bulk prompt per type per
  tick** instead, with backoff. This changes how fast idle agents get their next prompt, not what
  an agent does. For a strict comparison, rerun 100 agents with the bulk driver.

### Leaders talk on the msgboard (#flock-pixel-test)
- **`run/board.py`** (stdlib; the canvas serves it at `/cli/board`, and leader.md step 1 installs it):
  - `join` redeems `BOARD_INVITE` as `$FLOCK_AGENT`. The token goes to `.board_token` (mode 600,
    never printed), and the channel id is found by name and cached.
  - `post`, `reply <id>`, `react <id> <emoji>`.
  - `read`: new channel messages since the last read, from `/v1/activity`, with a cursor file.
  - `transcript`: the whole channel as JSONL, for the observer.
  - Writes carry an Idempotency-Key and retry 429/5xx with the same key.
- **Invite:** one multi-use agent/guest invite scoped to `#flock-pixel-test`
  (`join_open_channels: false`), so each leader becomes a member of exactly that channel. It goes
  into the flock secret `BOARD_INVITE` at launch; never commit or print it. `BOARD_URL` is a
  secret too. `spec-spawn.yaml` passes `BOARD_CHANNEL_NAME=flock-pixel-test`. Painters get none
  of this.
- **Protocol** (leader.md):
  - After its lease, a leader posts "leader-000N: leasing region x,y,w,h". On a 409 it reads the
    channel and takes the next band.
  - The watch round posts one progress line every ~3 min (band done %, painters working, idle).
  - When its band is done it posts "done, free capacity: 100 painters".
  - A stalled leader (no progress for 5 min) @mentions a leader that posted done. The holder then
    lends its band to the helper's whole team with one call: `canvas grant-region
    "leader-0007-*"`. Region grants accept `<leader>-*` patterns; tests/test_leaders.py checks
    that the helper's children then paint both bands while others stay blocked.
  - Leaders read the channel once per round, post one line, and never paste raw tool output.
- **Observer:** with `BOARD_URL`, `BOARD_TOKEN` (any principal that can read the channel; never
  printed) and `BOARD_CHANNEL` or `BOARD_CHANNEL_NAME` set, `driver.py --observe` archives the
  channel at run end to `final/<run>/board-transcript.jsonl`. The progress page's run view lists
  it below the replay.
- **Join schema:** `POST /v1/join {code, name, display_name}`. A *named* invite (`agent_name`
  bound) would reject other names. Ours is unnamed and multi-use: one code serves all 10 leaders,
  and a rejoin with the same name rotates the token and keeps the principal. Agents cannot create
  channels or invites, which is why the operator makes the channel and invite first.
- **Child reports:** painters do not `flock prompt "$FLOCK_PARENT"` on completion, and painter.md
  stays run 1's.
  - With the Claude harness each report would become a Sonnet turn: 100 per leader, ~1,000
    billed turns.
  - Leaders see completion from the canvas instead: the band's done %, and children that are idle
    with `region_done`.
  - Leaders never run `flock events --wait` per child; 1,000 long-polls would hit the Spaces edge.
    They check `flock agents --parent $FLOCK_AGENT --state idle` once per 30-s round.

### Launch order
1. `RUN_ID=run1k-YYYYMMDD experiments/pixel/run/job-1k.sh`. It launches 3 jobs (`$RUN_ID-gpu1..3`,
   `--timeout 75m`), one engine each with `--api-server-count 4 --max-num-seqs 512`. gpu1 also runs
   the canvas (`--expose-public 8765`). The script prints the three `base_url`s and `CANVAS_URL`.
   Logs go to `$BUCKET/pixel/runs/$RUN_ID/gpuN/`; the paint log goes to `$BUCKET/pixel/runs/$RUN_ID/`.
2. Fill `<JOB_ID_1..3>` in `spec-1k.yaml`. Set the Space secret `CANVAS_URL` (and `CANVAS_TOKEN`).
3. Wait until healthy:
   - `VLLM READY` in all three job logs;
   - `/v1/models` returns 200 through each proxy;
   - canvas `/status` answers.
4. Controller `cmpatino/flock-smoke` on the `pixel-run` wheel (it includes the bulk-prompt route).
   Run `flock apply experiments/pixel/run/spec-1k.yaml`; expect ~42 workers and 1,000 idle agents
   after a few minutes.
5. Driver (polls every 2 s: 2 canvas reads, 1 controller status and 3 bulk POSTs per tick):
   ```
   python experiments/pixel/run/driver.py --swarm pixel-1k --types painter-a,painter-b,painter-c \
     --canvas-url <CANVAS_URL> --repeat --interval 2 --gpus 3 --gpu-usd-per-hour 8.25 \
     --run-id $RUN_ID --run-name "1k x LFM2.5-2.6B" --max-minutes 30
   ```
6. Compare on the progress Space: `?tab=compare`, run 1 against this run. The time axis switches
   to seconds for short runs.

### The 5-minute check (from `flock apply`)
By minute 5 the swarm should be painting. If not, stop and diagnose before the GPUs burn time.
- `flock status pixel-1k`: ~1,000 agents, 42 workers running, no `launch_error`.
- Driver line:
  - `active` is in the hundreds within a minute of the bulk install;
  - `px/s` is in the thousands;
  - `idle` falls;
  - `retries` stays small (a few). If retries climb every tick, raise `--interval`.
- Canvas `/status`: `active_agents_10s` ≈ agents painting; `step_gap_p50_s` ~2.5-3.5 s (run 1 at
  25 per worker). Much higher means the workers are CPU-bound.
- Each GPU's `/metrics` through the proxy: running requests > 0 on all three. A GPU at 0 means
  its painter type is not running.

### Teardown (3 GPU jobs)
1. The driver stops at 100% and uploads `final/<run>/` (canvas.png, agents.png fresh, score.json,
   paint-log.jsonl.gz).
2. Run `flock remove pixel-1k`, and wait until the workers are gone (~2 min at 100 agents; longer at 42 workers).
3. Check the bucket paint log has `seq` lines: `$BUCKET/pixel/runs/$RUN_ID/paint.log.jsonl`
   (copied every 60 s; wait one more minute after the driver ends).
4. `hf jobs cancel` all three (`$RUN_ID-gpu1..3`); `hf jobs ps` should show none. Pause the controller Space.

### Expected numbers
- **Throughput wanted:** ~400 model calls/s = 1,000 agents × ~0.4 steps/s. Run 1 at 25 agents
  per worker did 0.23 steps/s per agent, so 234 steps/s is the floor if worker CPU is the limit
  again.
- **Available:** 3 × ~160 calls/s = **~480 calls/s** (rtx-pro-6000 probe knee).
- **Painting phase:** the work set is 62,278 calls at 10 px. At ~400 calls/s that is **~2.6 min,
  about 3 minutes**; at 234 calls/s it is ~4.5 min.
- **Setup dominates:** GPU boot ~5 min, apply ~2-5 min, painting ~3 min, teardown ~3 min.
- **Cost:**
  - Cap: 3 × 75 min × $2.75/h = $10.3.
  - Expected **≈ $7** (≈50 min of wall time per GPU including setup, smoke and slack).
  - Workers: 42 cpu-basic for ~15-20 min (cents).
- **Canvas** (one process, loadgen above):
  - ~400 steps/s means ~800 canvas req/s (claim + paint), at p95 38 ms and 79% of a core.
  - The ceiling is ~865 req/s, so all 480 calls/s (960 req/s) would saturate the canvas.
- **Short-run capture:**
  - The driver samples every 2 s, publishes every 30 s, and always uploads `final/`.
  - `agents.png` is fetched fresh at the end.
  - Replay keyframes are every 1,000 entries (~62 over the run).

## Experiment 2: 10 leaders x 100 painters on leased bands (spec-spawn.yaml)

- **Agents:** 10 `leader` agents (Claude Code, `claude-sonnet-5-5`, 5 per cpu-basic worker, so 2
  workers). Each spawns 100 painters (34 `painter-a` + 33 `painter-b` + 33 `painter-c`). The
  painter types, providers and settings are exactly spec-1k's, with `replicas: 0`: every painter
  is a child named `<leader>-<type>-<NNNN>`.
- **Limits:** `spawn: {leader: {types: {painter-a: 34, painter-b: 33, painter-c: 33}, depth: 1}}`
  and `limits: {alive: 1100, spawned_total: 1500}`. `spawned_total` counts every spawn ever, so
  1,500 leaves room for 500 replacements. The painter types set `max_working_per_worker: 25`
  explicitly. The spec validates against the `pixel-run` wheel.
- **Coordination is canvas region leases** (`job-spawn.sh` = `job-1k.sh` with `REGION_ENFORCE=1`):
  - Each leader leases one of 10 full-width bands. `canvas bands 10` gives equal-*work* bands;
    equal-height bands of this image range from 389 to 92k work pixels.
  - It grants the band to its 100 children with one bulk grant. Children can claim and paint only
    there: each region has its own claim queue, so this stays O(1) per claim.
  - When the band is finished, a child's claim says "your region is finished. You are done."
- **`leader.md`:** every step is ONE shell command, so a leader makes about 6-10 tool calls in
  total: install, lease, spawn 100 plus grant, prompt 100, then watch and re-prompt loops of up to
  ~8 min. Its messages are one line. Sonnet cost is paid per tool call, so the 100 spawns and
  100 prompts are shell loops, never 100 tool calls.
  - `tests/test_leader_md.py` runs these exact commands against a real canvas and a fake `flock`.
    The leader leases band 3, spawns 100, grants, prompts, and watches until "BAND DONE". A second
    leader asking for band 3 gets band 4. Only the leader's own children paint the band.
- **Views:**
  - The progress page's **leader territory** view (`/agents.png?by=parent`, and in the replays)
    colours each pixel by its painter's leader.
  - `canvas regions` and `/regions` show per-band done %.
  - Compare and Side by side work as for any run.

### Launch order
1. `RUN_ID=spawn-YYYYMMDD experiments/pixel/run/job-spawn.sh`: 3 rtx-pro-6000 jobs; the canvas
   on gpu1 with `REGION_ENFORCE=1`. Fill `<JOB_ID_1..3>` in `spec-spawn.yaml`.
2. Wait for `VLLM READY` x3 and canvas `/status`.
3. Controller `cmpatino/flock-smoke` on the `pixel-run` wheel, restarted.
   - Space secrets: `ANTHROPIC_API_KEY` (leaders), `HF_TOKEN`, `CANVAS_URL`, `CANVAS_TOKEN`,
     `BOARD_URL`, `BOARD_INVITE` (the #flock-pixel-test invite).
   - Turn on persistent storage: spawned rows live only in the controller's SQLite.
4. `flock apply experiments/pixel/run/spec-spawn.yaml`. The 10 leaders appear and idle (2 workers,
   ~1 min). No painters exist yet.
5. Start the observer (no prompts):
   ```
   python experiments/pixel/run/driver.py --observe --swarm pixel-spawn \
     --types leader,painter-a,painter-b,painter-c --canvas-url <CANVAS_URL> --interval 3 \
     --gpus 3 --gpu-usd-per-hour 8.25 --run-id $RUN_ID --run-name "10 leaders x 100" --max-minutes 40
   ```
   - It records per leader: children by state, `first_child_s` / `all_children_s` (spawn timing,
     from `GET /swarms/{s}/agents?parent=`, one call per leader every 30 s), cost and turns.
   - It also records per-region done %, and uploads to the progress Space every 30 s.
6. Prompt the 10 leaders **once**: `flock prompt --swarm pixel-spawn --type leader "go"`. The
   leaders do the rest; there is no central driver.

### Expected timeline (from "go")
| when | what |
|---|---|
| +0-1 min | leaders install, read bands, lease (409 -> next band) |
| +1-2 min | 1,000 `flock spawn` calls (100 serial per leader, 10 leaders in parallel; the controller handles ~300 spawns/s, local scale test), one grant per leader |
| +1-3 min | painter workers boot as slots fill: `workers.initial` does NOT pre-create workers, so the first spawn of each type waits ~45 s for a job; 42 workers launch in one pass and boot in parallel; first prompts wait as pending |
| +3-7 min | painting: ~62k calls at ~400 calls/s if all bands run at once, i.e. ~3 min; uneven leaders make it longer |
| +7-8 min | leaders see BAND DONE within 30 s, reply "band N done" |

### Teardown
- `flock remove pixel-spawn` removes leaders and cascades to all 1,000 children (drain; a few minutes).
- Check the bucket paint log (`$BUCKET/pixel/runs/$RUN_ID/paint.log.jsonl`). The observer has
  uploaded `final/`.
- `hf jobs cancel` the three `$RUN_ID-gpuN` jobs and pause the Space.

### Cost
- **GPUs:** ~$8 (3 x rtx-pro-6000 for ~60 min including setup; cap 3 x 75 min = $10.3).
- **Sonnet leaders:** `max_budget_usd: 1.0` per prompt, so at most $10 for the single "go" prompt.
  Expect well under that at ~6-10 tool calls each. Workers: 2 + 42 cpu-basic for ~20 min (cents).

### What needs a flock change
- **Bulk prompts are controller-only.** The leader's scoped token (`FLOCK_AGENT_TOKEN`) cannot use
  `POST /swarms/{s}/prompts`, and it must not get the controller token. So leaders prompt
  children one by one: 1,000 prompts at start, plus re-prompts of idle children every 30 s. That
  is up to ~30 req/s on the controller, the load that gave Space-edge 429s at 100 agents in
  run 1. Asks:
  - a scoped bulk prompt for one's own children (e.g. `POST /agents/{me}/children/prompts
    {state, text}`);
  - 429 retry in the `flock` CLI;
  - a bulk spawn (`count: N`) for the 1,000 single spawn calls.
- **No worker pre-warm:** `workers.initial` are slots, not started workers. The first child on
  each new worker waits for its boot (docs/spawn.md "Not yet"), so the painters do not "place
  instantly". The pending first prompt hides this, but spawn-to-first-paint is ~1 min.
- **Idle state lags** (heartbeats). `leader.md` only re-prompts idle children with `pending == 0`
  (column 6 of `flock agents`), which avoids most duplicate prompts.
