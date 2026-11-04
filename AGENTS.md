# flock — notes for agents working in this repo

Read `docs/design.md` first. It is the agreed design; section 5 is the contract and
`src/flock/models.py` is that contract in code. Do not change models.py
casually: controller, worker and CLI all import it.

## Layout
- `src/flock/models.py` — shared pydantic contracts (spec, heartbeat, commands, API shapes).
- `src/flock/controller/` — FastAPI app, SQLite store, reconcile loop, HF Jobs client, status page.
- `src/flock/worker/` — worker loop, harness adapters (`claude`, `codex`, `pi`, `llm`; see docs/pi.md, docs/llm-harness.md), bucket sync, msgboard identity, comms hub (docs/comms.md).
- `src/flock/comms/` — msgboard client, the built-in comms tools and their rules (docs/comms.md).
- `src/flock/cli.py` — `flock` CLI (typer) over the controller API.
- `experiments/` — milestone-1 experiments (resume gate, packing measurement) with their results as markdown.
- `tests/` — pytest; no network, no real jobs. Fake the Hub client and the harness processes.

## Conventions
- Python 3.11+, pydantic v2, httpx, typer, FastAPI. Keep it small; no new frameworks.
- Build the env on local disk: `UV_PROJECT_ENVIRONMENT="$AM_LOCAL/envs/flock" uv sync --all-extras`.
- `ruff check` and `pytest` must pass before pushing.
- Always pass `namespace=` to every `huggingface_hub` jobs call (saves a `whoami` per call).
- Never launch an HF Job from tests. Experiments launch jobs only on `cpu-basic` unless the operator approved otherwise.
- Commit small; push your branch at the end of every turn that produced work.

## Tests
The suite is already slow, so be conservative about adding to it.
- **Do not add tests by default.** Add one only when the user asks for it, or when a change to **core** flock
  (spec/models, controller, worker, CLI lifecycle, harness adapters, comms, checkpoint/resume) carries a clear
  regression risk that a test covers best. Do not add tests for specific worlds or experiments.
- **Prefer one tiny real run over many isolated tests.** Exercise behaviour through the public surface (the CLI,
  the controller API, the spec) with the smallest setup that shows it: a real controller and worker and a fake
  model server, not a hand-built object around a private function.
- **Fake only what cannot run here:** the Hub, HF Jobs, Slurm, model servers. Do not stand in for flock's own
  objects with `MagicMock`, `SimpleNamespace` or monkeypatched internals; such tests keep passing when the real
  code changes.
- **Scratch tests are fine while developing:** keep them in `scratch/` or `tmp/` and do not commit them.
- **Run what you touched:** `pytest tests/test_<area>.py` (or `-k`) while iterating, and the full suite once
  before pushing.
- **Documentation-only changes need no tests:** run `git diff --check` instead.
