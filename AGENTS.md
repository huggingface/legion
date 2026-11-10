# flock — notes for agents working in this repo

Read `docs/design.md` first. It is the current design; its "Contracts" section is the
contract and `src/flock/models.py` is that contract in code. Do not change models.py
casually: controller, worker and CLI all import it.

## Layout
- `src/flock/models.py` — shared pydantic contracts (spec, heartbeat, commands, API shapes).
- `src/flock/controller/` — FastAPI app, SQLite store, reconcile loop, HF Jobs client, status page.
- `src/flock/worker/` — worker loop, the `harbor` harness adapter (one Harbor Trial per agent), bucket sync.
- `src/flock/harbor/` — the Harbor task path: spec compile, Trials, Pi, and the `message_board`, `world`, `spawn` resources and tools (builtin/).
- `src/flock/comms/` — the controller's msgboard client and the board rules the board resource applies (docs/comms.md).
- `src/flock/cli.py` — `flock` CLI (typer) over the controller API.
- `experiments/` — Harbor experiments (`harbor-smoke`, `swarmchat-harbor`). Earlier experiments and their results are archived in `hf://buckets/cmpatino/legacy-swarmchat-experiments` (README inside).
- `tests/` — pytest; no network, no real jobs. Fake the Hub client and Harbor; tiny real runs use tests/tinyrun.py.

## Conventions
- Python 3.12+, pydantic v2, httpx, typer, FastAPI. Keep it small; no new frameworks.
- Install with `uv sync --all-extras` (Harbor is a main dependency; the extras add the controller's and the tests').
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
  a real controller and worker, a real spec, local compute and a fake Harbor environment (tests/tinyrun.py).
- **Use real objects, not stand-ins.** Do not use `MagicMock`, `SimpleNamespace`, `object.__new__` (building an
  instance without its `__init__`) or monkeypatched internals in place of flock's own objects: they only have
  what the test author thought of, so the test keeps passing when the real code changes. Fake only what cannot
  run in the test environment: the Hub, HF Jobs, Slurm, model servers, containers.
- **Test through the public surface** (the CLI, the controller API, the spec, public functions) rather than
  private methods: a test that calls a private method usually has to build the object around it by hand and
  breaks on any refactor. Call a private function directly only when its behaviour cannot be reached
  deterministically otherwise, such as a pure helper.
- **No `slow` tests.** Keep runs tiny. Tests that need an optional dependency or tool (an SDK,
  Linux-only features) are gated with `pytest.importorskip` or `pytest.mark.skipif`, so they skip where they
  cannot run.
- **Scratch tests are fine while developing:** keep them in `scratch/` or `tmp/` and do not commit them.
- **Run what you touched:** `pytest tests/test_<area>.py` (or `-k`) while iterating, and the full suite once
  before pushing.
- **Documentation-only changes need no tests:** run `git diff --check` instead.
