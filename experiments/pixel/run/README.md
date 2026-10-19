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
   `python3 -c "import os,urllib.request as u;u.urlretrieve(os.environ['CANVAS_URL']+'/cli/canvas','canvas')" && chmod +x canvas && ./canvas status`
2. Every later prompt: `./canvas claim 10 --json | ./canvas paint - --json`
   (`paint -` paints exactly what the claim returned; an empty claim prints "You are done").

The worker image has `python3` but no `curl`, hence the python one-liner. The agent's tool env
gets `CANVAS_URL`, `CANVAS_TOKEN` (type secrets) and `FLOCK_AGENT` (worker), which the CLI reads.
If an agent stepped 3 times without ever reaching the canvas, the driver re-sends the install.

## Launch order

1. **GPU job.** Run `RUN_ID=pixel-100-a experiments/pixel/run/job.sh`. You need `HF_TOKEN`, `VLLM_API_KEY` and `CANVAS_TOKEN` set locally.
   It syncs the code and target into the bucket, then starts the job and prints the job id.
2. **Wait until healthy.**
   - `hf jobs logs <id>` should show `VLLM READY`. That takes ~3 min of scheduling plus the install and load; the probe saw ~1.5 min from RUNNING to ready.
   - `curl <public 8765 URL>/status` should answer.
   - The public URL for port 8765 comes from the job page or `hf jobs inspect <id>`. I have not checked which one shows it.
3. **Secrets for flock.**
   - `CANVAS_URL` is the public 8765 URL.
   - `CANVAS_TOKEN` and `VLLM_API_KEY` take the same values as in step 1.
4. **Smoke.**
   - Run `flock apply experiments/pixel/run/spec-smoke.yaml`.
   - Then run `python experiments/pixel/run/driver.py --swarm pixel-smoke --canvas-url <URL> --max-steps 5`.
   - Check `/replay`. The log should show `painter-*` agents with `useful` outcomes.
   - Then `flock remove pixel-smoke`.
5. **Run.**
   - Run `flock apply experiments/pixel/run/spec-100.yaml`.
   - Then run `python experiments/pixel/run/driver.py --swarm pixel-100 --canvas-url <URL>`.
   - The driver stops when coverage hits 100%. Cap the run with `--max-minutes`, or each agent with `--max-steps`.
6. **Teardown.**
   - Run `flock remove pixel-100`.
   - Confirm the paint log is in the bucket (`$BUCKET/pixel/runs/$RUN_ID/paint.log.jsonl`; it is copied every 60 s). Optionally force a final copy by waiting one more minute.
   - Then `hf jobs cancel <id>`. Cancelling also triggers the exit copy, but don't rely on it.
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

## Needs PR #7 / blockers

- `harness: llm`, `tool_choice`, `context_tokens`: PR #7. The specs validate against PR #7's
  `flock.cli.load_spec`; `main` rejects them.
- **Network group:** flock dropped provider `network_group` in 0c1b4b2 (decision 27). Workers are not
  launched in the group, `${HF_NETWORK_GROUP_PREFIX}` is not expanded, and preflight from the
  controller cannot reach the in-group URL. Either restore b9cc630, or use the proxy fallback
  that is commented in the spec. For the fallback, vLLM is exposed with `--expose 8000` and runs
  without `--api-key`, and `HF_TOKEN` is sent as the bearer.
- **No `done` tool** in PR #7, and no `comms` block in the spec model. The driver decides when
  painting is finished, and the instructions go in as `system_prompt_file`.
- **No turn id in the tool env:** flock does not export the prompt or turn id, so `FLOCK_TURN` is
  empty. To join flock events with the paint log, use `prompts.jsonl` (agent, prompt_id, ts)
  against the canvas log's agent + ts.
- **`flock prompt --swarm/--type/--state`** (PR "prompt-type") does not exist yet. The driver
  detects it at start; until then it sends one prompt per agent over the controller API.
