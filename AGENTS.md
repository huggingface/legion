# legion — notes for agents working in this repo

Read `docs/design.md` first. It is the current design; its "Contracts" section is the
contract and `src/legion/models.py` is that contract in code. Do not change models.py
casually: controller, worker and CLI all import it.

## Layout
- `src/legion/models.py` — shared pydantic contracts (spec, heartbeat, commands, API shapes).
- `src/legion/controller/` — FastAPI app, SQLite store, reconcile loop, HF Jobs client, status page.
- `src/legion/worker/` — worker loop, the `harbor` harness adapter (one Harbor Trial per agent), bucket sync.
- `src/legion/harbor/` — the Harbor task path: spec compile, Trials, Pi, and the `message_board`, `world`, `spawn` resources and tools (builtin/).
- `src/legion/comms/` — the controller's msgboard client and the board rules the board resource applies (docs/comms.md).
- `src/legion/cli.py` — `legion` CLI (typer) over the controller API.
- `experiments/` — one folder per experiment (experiments/README.md lists them); see "Experiments and datasets".
- `tests/` — pytest; no network, no real jobs. Fake the Hub client and Harbor; tiny real runs use tests/tinyrun.py.

## Conventions
- Python 3.12+, pydantic v2, httpx, typer, FastAPI. Keep it small; no new frameworks.
- Keep imports at module scope by default. Use function-local imports only for a concrete reason,
  such as avoiding a circular dependency, keeping optional dependencies optional, or deferring
  expensive initialization. Add a short comment explaining the reason.
- Install with `uv sync --all-extras` (Harbor and controller dependencies are included by default; the dev extra adds tests and linting).
- `ruff check` and `pytest` must pass before pushing.
- Always pass `namespace=` to every `huggingface_hub` jobs call (saves a `whoami` per call).
- Never launch an HF Job from tests. Experiments launch jobs only on `cpu-basic` unless the operator approved otherwise.
- Commit small; push your branch at the end of every turn that produced work.
- YAML (spec files, and YAML snippets in docs and READMEs): write mappings in block style, never
  inline `{a: 1}`; inline lists like `[a, b]` and an empty mapping `{}` are fine.

## Container engines
- `environment.backend: podman` is the default and the engine on the HF cluster: run it on
  hopper-cpu. The GPU nodes' podman 3.4 cannot create the networks legion gives each Trial, so
  model servers go there (scripts/slurm/serve-vllm.sbatch) and agents do not.
- `backend: docker` is only for `compute: local` on a machine that already runs dockerd. Slurm
  refuses it and the cluster has no Docker.
- `Dockerfile.controller` / `Dockerfile.worker` are deployment images for the Space and HF Jobs,
  not the engine a Trial runs on.
- Docker-engine docs live in the "Local runs" sections (docs/local.md, experiments/README.md);
  Slurm and cluster docs mention podman only.

## Experiments and datasets
- A dataset (env) repo holds plain Harbor tasks only: `tasks/<name>/`, generators, and graders
  and checks that run through Harbor. No legion specs, imports, tool modules, drivers or
  `.legion/` readers.
- Everything legion-specific for an env lives in `experiments/<exp>/`: `spec.yaml` (plus variant
  specs), `team.md`, `tools/`, sweep scripts, `score.py`/`metrics.py` and a README. Pin the
  dataset with `task.revision`; never point `tools_dir` or another path into a dataset download.
- The entry point is a spec that `Spec.load` accepts, launched with `legion launch` (always
  through a controller). No foreign config formats or dataset CLIs. `legion.harbor.launch.run_experiment`
  is an internal one-shot path with no CLI; don't build on it.
- Task instructions and `team.md` don't name tool commands (`legion-tool ...`): legion's own
  agent instructions describe the tools.
- A change that removes or renames a spec field, CLI command, tool name or Python API needs a
  dated `docs/migrations.md` entry. Grep `experiments/` and the datasets pinned there for uses,
  and fix or flag them in the PR.
- Legion was named Flock until 2026-11-24: `flock` → `legion` (package, CLI), `flock-tool` →
  `legion-tool`, `FLOCK_*` → `LEGION_*`, `.flock/` → `.legion/`. Don't write the old names.

## Tests
Be conservative about adding tests.
- **Do not add tests by default.** Add one only when the user asks for it, or when a change to **core** legion
  (spec/models, controller, worker, CLI lifecycle, harness adapters, comms, checkpoint/resume) carries a clear
  regression risk that a test covers best. Do not add tests for specific worlds or experiments.
- **Prefer one tiny real run over many isolated tests.** Use the smallest real setup that shows the behaviour:
  a real controller and worker, a real spec, local compute and a fake Harbor environment (tests/tinyrun.py).
- **Use real objects, not stand-ins.** Do not use `MagicMock`, `SimpleNamespace`, `object.__new__` (building an
  instance without its `__init__`) or monkeypatched internals in place of legion's own objects: they only have
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
- **Full suite in parallel:** `pytest -n auto` (pytest-xdist, as CI runs it); tests must not share fixed
  ports, files or global state.
- **Documentation-only changes need no tests:** run `git diff --check` instead.
