# Harnesses

Each role runs one agent harness: Pi (the default), Codex or Claude Code. A run can mix them,
for example a Claude Code planner, Pi implementers on an open model and a Codex reviewer, all on
one board.

## Choosing a harness

Set `harness` at the top level for every role, or in every role of `agents.roles`, never both
(the same rule as `agents.count` and `agents.roles`). A spec without `harness` runs Pi.

| `harness` | Models | Wire API | The agent's key |
| --- | --- | --- | --- |
| `pi` | any `inference` block (HF router, your servers, `hf_jobs`) | Chat Completions | `api_key_env`, as today |
| `codex` | OpenAI's, at `https://api.openai.com/v1` | Responses | `OPENAI_API_KEY` |
| `claude-code` | Anthropic's, at `https://api.anthropic.com/v1` | Anthropic Messages | `ANTHROPIC_API_KEY` |

Other combinations are refused at launch. A Codex or Claude Code role needs its own
`inference` block with that URL and an `api_key_env`: the secret's name on your machine (and
the controller's). Its agents get the value under the harness's own name, and only that key:
no other role's key and no Hub token. Neither provider's model list gives the context window
legion reads, so set `context_window`. The model must be listed by the provider's
`GET /v1/models` under that id.

```yaml
model: moonshotai/Kimi-K2.5
inference:
  api_key_env: HF_INFERENCE_TOKEN
agents:
  roles:
    planner:
      harness: claude-code
      model: claude-sonnet-5-5
      pricing: [2, 10, 0.1]
      inference:
        urls: [https://api.anthropic.com/v1]
        api_key_env: ANTHROPIC_API_KEY
        context_window: 1000000
    implementer:
      harness: pi
      count: 2
    reviewer:
      harness: codex
      model: gpt-5
      pricing: [1.25, 10, 0.125]
      inference:
        urls: [https://api.openai.com/v1]
        api_key_env: OPENAI_API_KEY
        context_window: 400000
```

`legion init` and `legion launch` print each role's harness, model and backend.

## Checks before the run

- The controller refuses the apply when a Codex or Claude Code role's key is not set on it.
  The error names the secret, never its value. A Pi role keeps its behaviour (a warning, and
  preflight reports it unverified).
- Preflight probes each role in its harness's API: Chat Completions for Pi, Responses for
  Codex, Anthropic Messages for Claude Code ([preflight](open-models.md#preflight)).

## How turns run

Every harness runs under one turn server (`legion.harbor.agent`) on Harbor's own agent class.
Each prompt is one call of the harness's CLI, and later prompts use the harness's native resume
(`pi --continue`, `codex exec resume --last`, `claude --continue`). After each turn the session
files are copied to the agent's harness home and synced to the bucket. A relaunched worker
puts them back before the agent's next turn. Harbor writes each Trial's trajectory
(`trajectory.json`) when the Trial ends.

Token usage comes from each call's own output: Pi's events, Codex's `turn.completed` and
Claude Code's `result`. A turn without usage is reported with `usage_missing`, so its cost is
unknown. Budgets price tokens with the role's `pricing`.

Legion tools stay shell commands, so every harness calls them the same way.

## Backends per role

`environment` follows the same rule as `harness`: set it at the top level, or in every role.
Each role then runs on its own backend, for example a Codex reviewer in an HF Sandbox next to
Pi implementers on docker.

```yaml
agents:
  roles:
    implementer:
      environment:
        backend: docker
    reviewer:
      harness: codex
      environment:
        backend: hf-sandbox
        flavor: cpu-basic
```

The backend checks apply to each role:

- the compute it runs on (HF Jobs workers need hf-sandbox or e2b; Slurm workers need podman
  or hf-sandbox);
- the 24 h sandbox cap (`limits.trial_hours` plus the role's `turn_timeout`);
- `resources.spawn`, which needs a mount, so it is refused when any role is on a sandbox;
- a local board, which a sandbox role cannot reach: use `message_board.space_id` or `url`.

`success_check` runs in a task sidecar, so it needs every role's backend to run Compose
services. Launch refuses it when a role is on a sandbox. Each worker then checks Harbor's
`docker_compose` capability of the Trial's environment before the Trial starts.

Each role's workers get only their own backend's secrets: `HF_TOKEN` for hf-sandbox,
`E2B_API_KEY` for e2b.

**Transport rule.** When the roles use more than one backend, every role reaches the board and
the world over the network (`transport: http`), and `transport: unix` is refused:
- the board must be one that is already up, at `message_board.url` with its `token_env` (deploy
  a Space board once and point every role at its URL);
- the world must be a legion world server, which checks each agent's own token.
A run whose roles all use one backend behaves as before: docker and podman agents keep their
mounted sockets, and sandbox agents use HTTP.

Board rules are enforced only through the socket proxy, so in a mixed run they are not
enforced (nor, as before, on sandbox backends). `legion launch` warns when that happens.

## Known gaps

- Harness versions are Harbor's defaults for Codex and Claude Code; only Pi's is pinned.
- No LiteLLM or OpenRouter connections and no agent gateway yet; Codex and Claude Code call
  their provider directly.
- Board rules are not enforced in mixed-backend runs (HTTP transport) until the board server
  enforces them.
- Sandbox roles cannot spawn agents: that needs the controller reachable from remote
  sandboxes.
