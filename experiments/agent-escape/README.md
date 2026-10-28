# AgentEscapeBench: one puzzle, four independent workers

Four local Qwen3-14B agents solve the same 10-node escape-room instance, each in a
separate benchmark session. They share discoveries on a private managed message
board. One upstream `submit_answer` response with `is_win: true` establishes swarm
success; an agent saying it has finished does not.

## Setup and launch

Run from the Flock repository with its existing `.venv` (including controller and
msgboard extras). The example expects:

- Qwen3-14B served at `http://127.0.0.1:8000/v1`, with tool calling and the Qwen3
  reasoning parser enabled; the tested server has a 16,384-token context.
- `HF_TOKEN` with access to the private `HuggingFaceSwarms/Experiments` bucket.
- `hf-mount-fuse` 0.13.2 on PATH or in `.flock/tools/`, `/dev/fuse`, `fusermount3`,
  and writable `/scratch/flock-boards` storage.

Install the pinned benchmark and its separate environment once:

```bash
bash experiments/agent-escape/setup.sh
```

Launch one fresh attempt:

```bash
uv run --no-sync python experiments/agent-escape/run.py
```

Edit `spec.yaml` and `prompt.md` here; no temporary YAML is needed. `--spec PATH`
selects an alternative YAML, and `--timeout 1800` bounds an attempt to 30 minutes,
plus graceful cleanup. The adapter expects exactly `worker-0000` through
`worker-0003`; retain four replicas and one agent per worker. The launcher adds
local tool paths and supplies `LOCAL_KEY=unused` unless already configured.

## Puzzle selection

Use `--difficulty {5,10,15,20,25}` and `--scenario-index N` to select one puzzle.
Difficulty is the puzzle node count, not a guaranteed success rate. Indices are
zero-based within narrative JSON filenames sorted lexicographically at the pinned
benchmark revision. Difficulties 5, 10, 15, and 20 have 60 scenarios (indices 0–59);
difficulty 25 has 30 (indices 0–29).

Defaults are `--difficulty 10 --scenario-index 0`, preserving the original puzzle.
All four workers receive independent sessions of the selected scenario. Invalid
selections fail before creating artifacts or starting services. Startup prints the
selection; `metadata.json` and `result.json` record its difficulty, index, path,
SHA-256 hash, and benchmark revision.

For example, run difficulty 20, scenario index 7 with Cerebras:

```bash
uv run --no-sync python experiments/agent-escape/run.py --spec experiments/agent-escape/spec-hf-endpoint.yaml --difficulty 20 --scenario-index 7
```

## HF Inference Providers configuration

`spec-hf-endpoint.yaml` uses `Qwen/Qwen3.8-27B:cerebras` through
`https://router.huggingface.co/v1`. It keeps four local workers, independent copies
of the same puzzle, and the existing bucket-backed board.

Export `HF_TOKEN` with Inference Providers permission and access to the private
board bucket. No dedicated endpoint is needed. The configuration retains 4,096
output / 10,000 context tokens and explicitly requests `reasoning_effort: low`
through the model's `extra` fields. Cerebras defaults this model to `high` (Qwen's
native `xhigh`); `low` reduces reasoning effort without disabling it. See the
[Cerebras API reference](https://inference-docs.cerebras.ai/api-reference/chat-completions).

Run from `flock/`:

```bash
uv run --no-sync python experiments/agent-escape/run.py --spec experiments/agent-escape/spec-hf-endpoint.yaml
```

A Cerebras low-reasoning run solved the default puzzle in about 68 seconds, with
verified success and board persistence. Pricing from the HF router catalog on 2026-10-28 is $0.99 per million
input tokens and $1.49 per million output tokens. Cached input is conservatively
priced at the full input rate. These rates are recorded in the YAML for Flock's
cost estimates; actual charges follow the provider's current billing.

## Environment and collaboration

The benchmark is pinned to `03742c79f48646e0130886e18e3e79798cf8edd7`, with default scenario
`questions/10/narrative/skeleton_091c541b_narrative.json`. Setup does not regenerate
puzzles or modify Flock's environment. The upstream MIT-licensed implementation
lives in ignored `.flock/deps/agent-escape/`.

The adapter creates independent scenario/simulator sessions and routes actions by
Flock identity. `get_observation` returns the initial public clues, including the
victory-node ID, without resetting progress. Every three puzzle actions, the adapter adds a generic reminder
to post a useful finding and read the board; it supplies no puzzle-specific hints.
Agents cannot choose session IDs,
reset the environment, read files, or execute shell commands. They can exchange
values freely but must unlock their own copies. This is tool-interface isolation,
not OS sandboxing between mutually untrusted processes.

Reasoning is enabled, with 4,096 output tokens, 10,000 context tokens, and at most
100 model calls per prompt. The local board call allowance is one request/second
per agent. These settings were adjusted after initial non-reasoning runs failed;
see `RESULTS.md`. This is an exploratory swarm experiment, not an official
benchmark score or a controlled comparison to a single agent.

## Artifacts and stopping

Each invocation prints `.flock/agent-escape-attempts/<uuid>/`. It contains:

- `metadata.json`, frozen adapter/spec/prompts: benchmark and adapter hashes and settings.
- `trajectory.jsonl`: worker identities, puzzle actions, and server responses.
- `success.json`, when solved, and `result.json`: verified outcome and action counts.
- `flock/`: ordinary run bundle, full agent transcripts, token totals, and board metadata.
- `flock.log` and `benchmark.log`: debugging logs; evaluator logs may contain solutions.

The board remains in its unique HF bucket prefix. Its local `board/metadata.json`
records the remote location and `storage.verified_at` after successful shutdown.
Only puzzle/communication tools are available to agents; evaluator logs and raw
scenario files are not supplied as observations. Treat run artifacts as private.

The launcher stops on verified success, all agents becoming idle, timeout, or
Ctrl-C. It drains Flock before stopping the benchmark so the board can flush.
Exit 0 requires both verified success and clean Flock shutdown; otherwise exit 1.
It leaves the existing vLLM server running. Never interpret `flock_exit_code: 0`
alone as solving the puzzle.

## Validation

```bash
uv run --no-sync pytest -q tests/test_agent_escape_experiment.py
.flock/deps/agent-escape-venv/bin/python experiments/agent-escape/check_upstream.py
uv run --no-sync ruff check .
```

The first command uses a fake HTTP backend without network or real jobs. The
second checks the pinned engine's isolation and answer checker using private
state in a separate disposable session. Its synthetic correct submission is
**not** a swarm result and never enters agent prompts or experiment trajectories.
