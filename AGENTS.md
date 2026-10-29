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
- Build the env with `uv sync --all-extras`.
- Do not add `from __future__ import annotations` to new code; Python 3.11+ does not need it.
- `ruff check` and `pytest` must pass before pushing.
- Always pass `namespace=` to every `huggingface_hub` jobs call (saves a `whoami` per call).
- Never launch an HF Job from tests. Experiments launch jobs only on `cpu-basic` unless the operator approved otherwise.
- Commit small; push your branch at the end of every turn that produced work.

## Dataset provenance
- Every dataset or config pushed to the Hub ships the script that built it in the dataset's
  own repo, as `scripts/create_dataset.py` (or `scripts/create_<config>.py` for a config added
  later), not in this repo. The script pushes its data, card and itself in the same run.
- The card starts, right after its YAML front matter, with a `subset | description |
  num_samples` table: one row per published config, with its published example count.
- The card ends with a `## Reproducing` section giving the exact command, pinned source
  revisions included, so the dataset can be rebuilt from the card alone.
