# flock — notes for agents working in this repo

Read `docs/design.md` first. It is the agreed design; section 5 is the contract and
`src/flock/models.py` is that contract in code. Do not change models.py
casually: controller, worker and CLI all import it.

## Layout
- `src/flock/models.py` — shared pydantic contracts (spec, heartbeat, commands, API shapes).
- `src/flock/controller/` — FastAPI app, SQLite store, reconcile loop, HF Jobs client, status page.
- `src/flock/worker/` — worker loop, harness adapters (`llm`, `harbor`; see docs/llm-harness.md), bucket sync, msgboard identity, comms hub (docs/comms.md).
- `src/flock/comms/` — msgboard client, the built-in comms tools and their rules (docs/comms.md).
- `src/flock/cli.py` — `flock` CLI (typer) over the controller API.
- `experiments/` — Harbor experiments (`harbor-smoke`, `swarmchat-harbor`). Earlier experiments and their results are archived in `hf://buckets/cmpatino/legacy-swarmchat-experiments` (README inside).
- `tests/` — pytest; no network, no real jobs. Fake the Hub client and the harness processes.

## Conventions
- Python 3.11+, pydantic v2, httpx, typer, FastAPI. Keep it small; no new frameworks.
- Install with `uv sync --all-extras`.
- `ruff check` and `pytest` must pass before pushing.
- Always pass `namespace=` to every `huggingface_hub` jobs call (saves a `whoami` per call).
- Never launch an HF Job from tests. Experiments launch jobs only on `cpu-basic` unless the operator approved otherwise.
- Commit small; push your branch at the end of every turn that produced work.

## Tests
Be conservative about adding tests.
- **Do not add tests by default.** Add one only when the user asks for it, or when a change to **core** flock
  (spec/models, controller, worker, CLI lifecycle, harness adapters, comms, checkpoint/resume) carries a clear
  regression risk that a test covers best. Do not add tests for specific worlds or experiments.
- **Prefer one tiny real run over many isolated tests.** Use the smallest real setup that shows the behaviour:
  a real controller and worker, a real spec, local compute and a fake model server.
- **Use real objects, not stand-ins.** Do not use `MagicMock`, `SimpleNamespace`, `object.__new__` (building an
  instance without its `__init__`) or monkeypatched internals in place of flock's own objects: they only have
  what the test author thought of, so the test keeps passing when the real code changes. Fake only what cannot
  run in the test environment: the Hub, HF Jobs, Slurm, model servers.
- **Test through the public surface** (the CLI, the controller API, the spec, public functions) rather than
  private methods: a test that calls a private method usually has to build the object around it by hand and
  breaks on any refactor. Call a private function directly only when its behaviour cannot be reached
  deterministically otherwise, such as a pure helper.
- **No `slow` tests.** Keep runs tiny. Tests that need an optional dependency or tool (a harness CLI, an SDK,
  Linux-only features) are gated with `pytest.importorskip` or `pytest.mark.skipif`, so they skip where they
  cannot run.
- **Scratch tests are fine while developing:** keep them in `scratch/` or `tmp/` and do not commit them.
- **Run what you touched:** `pytest tests/test_<area>.py` (or `-k`) while iterating, and the full suite once
  before pushing.
- **Documentation-only changes need no tests:** run `git diff --check` instead.
