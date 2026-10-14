# flock — notes for agents working in this repo

Read `docs/design.md` first. It is the agreed design; section 5 is the contract and
`src/flock/models.py` is that contract in code. Do not change models.py
casually: controller, worker and CLI all import it.

## Layout
- `src/flock/models.py` — shared pydantic contracts (spec, heartbeat, commands, API shapes).
- `src/flock/controller/` — FastAPI app, SQLite store, reconcile loop, HF Jobs client, status page.
- `src/flock/worker/` — worker loop, harness adapters (`claude`, `codex`), bucket sync, msgboard identity.
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
