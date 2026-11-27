# Migrations

Spec changes that need you to edit a spec or a stored swarm, newest first.

## 2026-11-27: `inference.api_format` is back, with `inference.preflight`

[huggingface/legion#156](https://github.com/huggingface/legion/pull/156)

- `inference.api_format` picks Pi's wire API again: `openai_chat_completions` (the default, and
  the only behaviour since 2026-11-17), `openai_responses` or `anthropic_messages`. The Trial,
  the upstream launcher's connection check, the model check and preflight all use it. Specs
  without it run as before.
- `inference.preflight: false` keeps the controller's model check and preflight off endpoints
  that only the Trial reaches, such as a task's own model gateway.
- Harness registry entries (`registry.HarnessEntry`) no longer carry an `api`: the block's
  `api_format` is the API. `PreflightResult.api` is now `chat_completions`, `responses` or
  `messages`.

See [open-models.md](open-models.md) "API format" and "Endpoints inside the Trial".

## 2026-11-24: Flock is now Legion

[huggingface/legion#135](https://github.com/huggingface/legion/pull/135) (803873b) renamed
everything, with no aliases for the old names:

- Python package and distribution: `flock` → `legion` (`import flock...` → `import legion...`,
  `python -m flock.controller.localrun` → `python -m legion.controller.localrun`).
- Commands: `flock`, `flock-agent`, `flock-controller`, `flock-worker` → `legion`,
  `legion-agent`, `legion-controller`, `legion-worker`. Inside a Trial, `flock-tool` →
  `legion-tool`.
- Environment variables `FLOCK_*` → `LEGION_*`, HTTP headers `X-Flock-*` → `X-Legion-*`, and run
  directories `.flock/` → `.legion/`.

Prompts, `team.md` files and scripts that still say `flock-tool` or `FLOCK_*` fail at run time.

## 2026-11-20: a list selector for one task per agent

[huggingface/legion#134](https://github.com/huggingface/legion/pull/134) (8740fa1). A string
`task.selector` (exact name or glob) must now match exactly one task, which every agent runs; a
glob that matches several tasks is refused. To give each agent its own task, list one entry per
agent (`agent-001` runs the first entry's task):

```yaml
naming: consecutive-id
agents:
  count: 4
task:
  selector: [team-s01, team-s02, team-s03, team-s04]  # was: team-s*
```

See [tasks.md](tasks.md) "Select it in Legion".

## 2026-11-20: `bucket:` is now `bucket_id:`

[huggingface/legion#131](https://github.com/huggingface/legion/pull/131) (93f5ae3) removed
`bucket:`. Write `bucket_id:` with a bare name (`legion-runs`, under `HF_TOKEN`'s account) or
`<namespace>/<name>`; `hf://buckets/...` URLs are refused there.

- `bucket: hf://buckets/ns/name` → `bucket_id: ns/name`.
- A spec that still says `bucket:` loads without a bucket: `compute: slurm` and `hf-jobs` fail
  with "bucket_id is required", and a local run keeps no Hub checkpoint.

See [billing.md](billing.md).

## 2026-11-17: one `inference` block

[huggingface/legion#112](https://github.com/huggingface/legion/pull/112) (ece6ba7) replaced
`providers:`, `models:`, the `huggingface/` prefix, the `:provider` suffix, `base_url(s)`,
`self_hosted` and `api_format` with one `inference` block (`hf_router`, `urls` or `hf_jobs`).
Specs and stored swarms in the old shape no longer load.

- `base_url: X` → `urls: [X]`; `model` is the raw id the server takes (no `openai/` prefix).
- `api_format` was removed and every Trial spoke Chat Completions. Experiments that needed
  another API patched `legion.harbor.launch.build_trial_config` (CIB did); use
  `inference.api_format` (2026-11-27) instead.
