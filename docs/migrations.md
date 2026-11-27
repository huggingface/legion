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

## 2026-11-17: one `inference` block

[huggingface/legion#112](https://github.com/huggingface/legion/pull/112) (ece6ba7) replaced
`providers:`, `models:`, the `huggingface/` prefix, the `:provider` suffix, `base_url(s)`,
`self_hosted` and `api_format` with one `inference` block (`hf_router`, `urls` or `hf_jobs`).
Specs and stored swarms in the old shape no longer load.

- `base_url: X` → `urls: [X]`; `model` is the raw id the server takes (no `openai/` prefix).
- `api_format` was removed and every Trial spoke Chat Completions. Experiments that needed
  another API patched `legion.harbor.launch.build_trial_config` (CIB did); use
  `inference.api_format` (2026-11-27) instead.
