# Pixel run kit (first 100-agent run)

Nothing here has been launched. Files:

| file | what |
|---|---|
| `painter.md` | agent instructions (19 lines): install step, then one chained command per prompt |
| `spec-100.yaml`, `spec-smoke.yaml` | flock specs: 100 painters on 2 cpu-basic workers (+1 spare); smoke = 3 painters, 1 worker |
| `driver.py` | keeps painters prompted until the canvas is done; writes `progress.jsonl`, `prompts.jsonl` |
| `job.sh` | the single GPU job: vLLM (LFM2.5-2.6B, :8000) + canvas (:8765) in one container |

## How a painter works

`tool_choice: required` (PR #7) makes every flock prompt exactly one forced bash call, so the
**driver is the loop**. Each prompt carries one literal command:

1. First prompt per agent (install the stdlib-only CLI from the canvas, `GET /cli/canvas`):
   `python3 -c "import sys,urllib.request as u;u.urlretrieve(*sys.argv[1:])" $CANVAS_URL/cli/canvas canvas && chmod +x canvas && ./canvas status`
2. Every later prompt: `./canvas claim 10 --json | ./canvas paint - --json`
   (`paint -` paints exactly what the claim returned; an empty claim prints "You are done").

The worker image has `python3` but no `curl`, hence the python one-liner. The agent's tool env
gets `CANVAS_URL`, `CANVAS_TOKEN` (type secrets) and `FLOCK_AGENT` (worker), which the CLI reads.
If an agent stepped 3 times without ever reaching the canvas, the driver re-sends the install.

## Launch order (run 1: jobs-proxy path)

The model and the canvas are reached through the HF jobs proxy; network groups are not used (flock
dropped provider `network_group` on purpose, 0c1b4b2; the probe in
`flock-board/probe-reachability-lfm.md` showed `Bearer $HF_TOKEN` through the proxy -> 200 for
chat completions at the same latency, and the controller's preflight can reach it too).

1. **GPU job.** Run `RUN_ID=pixel-100-a experiments/pixel/run/job.sh`. You need `HF_TOKEN` and `CANVAS_TOKEN` set locally.
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
   - Then run `python experiments/pixel/run/driver.py --swarm pixel-100 --canvas-url <canvas URL> --run-id pixel-100-a`, with `SPACE_ID`/`HF_TOKEN` set for the progress page.
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
