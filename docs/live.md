# Live changes: `flock update`, `flock run --watch`, `flock grant`

Everything an experiment customises lives in its folder, next to the spec: `spec.yaml`, the
prompt files it names, `tools/`, the comms protocol files. flock reads them at runtime, and
they can change while the swarm runs. Three commands do this:

- `flock update` applies an edited folder to the running swarm.
- `flock run --watch` does the same on every save, locally.
- `flock grant` changes what one agent or one type may do, without editing the spec.

## `flock update`

```console
$ flock update experiments/calc          # the folder, or its spec.yaml
tools: 3f9c0e2a7b61d4c8 -> 9a1b2c3d4e5f6a7b (+mul)
prompts changed: calc (system_prompt); comms (protocols_text)
types.calc.replicas: 2 -> 3
```

It reads the folder exactly as `flock apply` does: the spec, `system_prompt_file`, the loop
prompt files, the comms kickoff and protocols, and the tools folder. It checks the folder,
compares it with the spec the controller is running and applies it. The output is one line per
change: the tools version (old -> new, with the tools added and removed), the prompts changed
per type, then every other key as `path: old -> new`. When nothing changed it prints
`no changes` and applies nothing.

A refused update exits 1, prints the reason and applies nothing. The reasons are the ones
`flock apply` gives: a spec error, a tools folder error, an unknown tool name, or a 400/409
from the controller. On a running swarm the controller also refuses to change the harness of a
type that has agents. Use a new type name for that, or remove the type first.

Preflight runs only when the update adds a provider-backed model the running spec does not
have (`--preflight` / `--skip-preflight` to choose). Locally, `flock update` finds the run
through `.flock/current`, like the other commands do.

## `flock run --watch`

`flock run spec.yaml --watch` polls the files the spec references and the tools folder every
second. Once a change has been quiet for a second, it runs `flock update`. The terminal view
gets a `live updates` pane with the update's lines. A broken save (half-written YAML, a
syntax error in a tool) shows the refusal and `not applied: the swarm keeps running as it
was`. The next good save applies. Watching is off by default.

## What takes effect when

| change | takes effect |
|---|---|
| tools folder (a new content hash) | each agent's next turn. The worker fetches the version within a heartbeat (30 s on jobs, 5 s locally); a turn in flight keeps its version (docs/tools.md) |
| `system_prompt` / `system_prompt_file` | each agent's next turn boundary. The worker rewrites `CLAUDE.md` / `AGENTS.md` and the `llm` harness rebuilds its system message |
| `loop:` settings and loop prompt files | the next loop tick (docs/loop.md) |
| `comms.rules`, `comms.protocols` | the next heartbeat; the instructions' board section at the next turn |
| `comms.kickoff_file` | never: the kickoff is posted once per channel |
| `limits.budget_usd`, `limits.alive`, `spawn:` | at once (the controller) |
| `replicas` | the next reconcile pass |
| other type fields (`max_turns`, `max_budget_usd`, `tool_choice`, `turn_timeout`, ...) | the next turn boundary. A CLI harness takes the per-prompt limits, `model`, `env` and `secrets` at its next start, after `idle_grace` |
| a type's `harness` (with agents), `compute`, `swarm` | refused |

A turn boundary is a prompt that starts a new turn. A prompt that joins a turn in flight
changes nothing until that turn ends. Changed type specs and grants reach the workers in the
heartbeat response. The worker sends its `spec_rev`, and the controller answers with the new
`type_spec` only when they differ.

**The note.** An agent that already had a turn gets one line in front of its next prompt for
each kind of change:

```
[flock] Since your last turn: your system prompt changed; your grants changed: tools -add, bash off.

<the prompt>
```

CLI harnesses (`claude`, `codex`, `pi`) read their instructions file only at start. For them
the note says which file to re-read (`your instructions in CLAUDE.md changed; re-read them`,
`your tools changed; re-read the Tools section of AGENTS.md`).

## `flock grant`, `flock revoke`, `flock grants`

```sh
flock grant calc tools=-add bash=off              # every calc agent, from its next turn
flock grant calc-0001 +mul limits.max_turns=5     # one agent; `+name` is short for tools=+name
flock grant writer comms.post=off
flock grant calc-0002 limits.budget_usd=0.50
flock grants                                      # what is in force
flock revoke calc-0001 limits                     # some capabilities, or all of them:
flock revoke calc
```

| capability | effect |
|---|---|
| `tools=+name,-name` | adds or takes away tools that already exist: folder tools in the current bundle, and the comms tools `post`, `read`, `search` |
| `comms.post=on\|off` (also `comms.read`, `comms.search`) | the same as `tools=+post` / `tools=-post` |
| `bash=on\|off` | `llm` harness only: the `bash` tool. With bash off, the agent keeps its other tools |
| `limits.max_turns=N` | model calls per prompt, as `types.<t>.max_turns` |
| `limits.budget_usd=X` | the agent's total cost cap. At or above it: a prompt gets 409, a bulk prompt or loop prompt skips the agent (`skipped_budget`), and the worker defers one already queued (`prompt_deferred`, reason `budget`). Below it, each turn is capped at what is left (`max_budget_usd`) |

**Targets.** A target is a type (every agent of it, including later ones) or one agent. The
command finds the swarm from the target; pass `--swarm` when the name is ambiguous. An agent's
grant wins over its type's grant, key by key.

**Storage.** Grants are an overlay that the controller stores, not part of the spec. This
means `flock update` and `flock apply` keep them. Removing an agent drops its grant, and
removing the swarm drops all of them.

**Refusals.** A grant that the workers cannot enforce is refused (400):

- an unknown tool;
- a comms tool for a type that has no comms tools (its workers hold no board invite);
- `comms.search` while `comms.search` is off;
- `bash` on a type whose harness is not `llm`.

**Where grants show.** In `flock agent <name>` (`grants`, the merged grant in force), in
`flock grants`, on the swarm event stream (`grant`, `grant_revoked`; also on the agent's
stream for agent targets), and in the `flock run` header (`N grants`).

### API

| Method | Path | |
|---|---|---|
| GET | `/swarms/{swarm}/grants` | `{swarm, grants: [{target, kind: type\|agent, grant, updated_at}]}` |
| PUT | `/swarms/{swarm}/grants/{target}` | replaces the target's grant; an empty body revokes it |
| DELETE | `/swarms/{swarm}/grants/{target}` | revokes it (404 when there is none) |

```json
{"tools": {"mul": true, "comms:post": false}, "bash": false,
 "limits": {"max_turns": 5, "budget_usd": 2.0}}
```

These routes need the controller token; an agent's own token cannot grant.

**Not in this version.** These are still open:

- a `--note` that goes to the agent with a grant;
- grants issued by a world (`grantable:`);
- tools that are not already in the bundle (code arrives through `flock update`, access
  through `flock grant`).
