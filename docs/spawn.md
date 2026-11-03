# Spawning agents

A running agent (a "leader") can create child agents of other types, prompt them, read their
event streams and remove them. Children are ordinary agents: they have a harness session, a
workdir, a worker, a state and a cost like any replica, and they idle and resume the same way.
What is new is a `parent`, a `depth`, and a token that lets the agent call the controller as
itself.

Example specs: [`examples/first-spawn.yaml`](../examples/first-spawn.yaml) (the walkthrough
below) and [`examples/spawn.yaml`](../examples/spawn.yaml). Design notes and decisions:
[design.md §13](design.md#13-spawn).

## First spawn: a leader spawns 5 children

From a clone of this repo, with `uv sync --all-extras` done. Replace `<you>` everywhere,
including in `examples/first-spawn.yaml`. Cost: a few cents of Haiku and under $0.01 of jobs.

`uv sync` does not activate the environment, so every command goes through `uv run`, which
also gives the scripts a Python with `huggingface_hub`. A bare `flock` may be util-linux's
file-locking tool (`/usr/bin/flock`), a different program.

```sh
export HF_TOKEN=... ANTHROPIC_KEY=... FLOCK_TOKEN=$(openssl rand -hex 24)
uv run hf buckets create <you>/flock --private                      # once
W=$(uv run scripts/publish-wheel.sh hf://buckets/<you>/flock/dist)  # wheel the workers install
uv run scripts/deploy-space.sh "$W" <you>/flock                     # returns while it builds
export FLOCK_URL=https://<you>-flock.hf.space
# 1. wait for the Space to build (a minute or two)
until curl -fsS "$FLOCK_URL/healthz" >/dev/null; do sleep 10; done
uv run flock launch examples/first-spawn.yaml
# 2. apply returns before reconcile creates the leader: wait for its row (a few seconds)
until uv run flock agents first-spawn | grep -q '^leader-0000 '; do sleep 5; done
uv run flock prompt leader-0000 "Start: spawn the 5 analyst children now."
# 3. follow the leader (its worker boots first, ~40 s): spawned x5, then "spawned 5"; Ctrl-C
uv run flock events leader-0000 --follow --pretty
uv run flock agents first-spawn             # repeat until the children are idle
```

The children show `unknown` with `pending 1` for about 50 s: the first spawn of a type with no
agents boots its worker (a job launch). That is normal. Then each child answers, reports to the
leader (the reports arrive as the leader's turns t-2, t-3, ...), and idles:

```text
$ uv run flock agents first-spawn   # about 1 min after the prompt; last_heartbeat cut
name                      state  worker                     turns  pending  cost
leader-0000               idle   first-spawn-leader-w0000   1      0        $0.0030
leader-0000-analyst-0001  idle   first-spawn-analyst-w0000  1      0        $0.0021
...
leader-0000-analyst-0005  idle   first-spawn-analyst-w0000  1      0        $0.0019
$ uv run flock events leader-0000-analyst-0003 --pretty
08:55:31     - session_started session_id=… resumed=False parent=leader-0000
08:55:31   t-1 user     Capital of France? One word.
08:55:34   t-1 assistant Paris
08:55:35   t-1 assistant [Bash {"command": "flock prompt \"$FLOCK_PARENT\" \"leader-0000-analyst-0003: Paris\"…]
08:55:36   t-1 turn_ended completed claude-haiku-5-5 in=… out=…
```

`turns` and `state` come from worker heartbeats (every 30 s), so they lag the event stream. An
agent's answer is the text of its `assistant` messages; `turn_ended` only says the turn is over.
If a worker never gets past `launching`/`dead`, `uv run flock workers first-spawn` shows its
`last_error` (status and log tail); `hf jobs logs <job>` has the rest. A 401 there usually means
`HF_TOKEN` is missing from the type's `secrets`.

## Teardown

```sh
uv run flock remove first-spawn                    # drains and cancels its workers (about 20 s)
uv run hf jobs ps --label swarm=first-spawn        # empty once they are gone
uv run hf spaces pause <you>/flock                 # the Space is public and holds your secrets
uv run hf buckets delete <you>/flock               # optional: wheels and agent state
```

## Spec

```yaml
types:
  leader:  {harness: claude, model: claude-opus-5-5, replicas: 1, workers: {initial: 1, max_extra: 0}}
  analyst: {harness: codex,  model: gpt-5.5-codex,   replicas: 0, workers: {initial: 1, max_extra: 3}}
  helper:  {harness: codex,  model: gpt-5.5-codex,   replicas: 0, workers: {initial: 0, max_extra: 2}}
spawn:                                   # run policy, separate from what a type is
  leader:  {types: {analyst: 3, helper: 2}, depth: 2}   # per-type caps, total 5
  analyst: {types: [helper], max: 3, depth: 1}         # list + one total cap
limits:  {alive: 500, spawned_total: 2000, budget_usd: 300}   # optional
```

| Field | Default | Meaning |
|---|---|---|
| `spawn.<type>` | absent | what agents of `<type>` may spawn. A type absent from `spawn:` cannot spawn |
| `spawn.<type>.types` | required | a list of child types (then `max` is required), or a map `{child type: cap}` |
| `spawn.<type>.max` | none | cap on the agent's existing children, all types together. Optional with the map form |
| `spawn.<type>.depth` | 1 | generations allowed below the agent. 0: it cannot spawn |
| `types.<t>.workers.initial` | v1's count: `ceil(replicas / agents_per_worker)` | worker slots counted toward the cap. Not pre-created: a worker opens only when an agent needs one |
| `types.<t>.workers.max_extra` | 0 | worker slots spawned children of `<t>` may add on demand |
| `limits.alive` | none | agents that exist in the swarm, replicas and spawned together |
| `limits.spawned_total` | none | children ever spawned in the swarm (the name counter); removing one does not free a place |
| `limits.budget_usd` | none | cost of every agent of the swarm, removed ones included. Checked when spawning; it does not stop running agents |

Caps count a parent's existing children, so removing a child frees its place. A child's own
per-prompt `max_turns` and `max_budget_usd` come from its type, as for replicas.

`flock launch` (and the controller) reject: an unknown type in `spawn:`, a cycle (a type that can
reach itself, including spawning its own type), a spawnable type without `workers`, the list form
without `max`, an unknown key under `spawn.<type>`, and negative `depth`, caps, `workers` or
limits. The controller also refuses (409) a re-apply that lowers a type's `initial + max_extra`
below the workers its agents occupy: agents never move, so remove agents first.

**Depth.** A replica's depth is its type's `spawn.depth` (0 if absent), read from the current
spec. A child is born with `min(parent's depth - 1, its own type's spawn.depth)` (0 if its type
is absent from `spawn:`) and keeps it. With no cycles and finite depth, recursion ends by
construction. `depth` is what is left, so a child of a `depth: 1` leader shows `depth: 0`.

## Lifecycle

- **Name:** `<parent>-<type>-<NNNN>` (with `naming: words`: a reserved `<adjective>-<noun>`, see
  "Names"), `NNNN` from a per-swarm counter that is never reused while
  the swarm exists, e.g. `leader-0000-analyst-0007`, then `leader-0000-analyst-0007-helper-0012`.
- **Placement:** on the child type's own workers with v1's fill rule (fill in order, never move,
  open a new worker on demand). A type that declares `workers` never gets more than
  `initial + max_extra` worker slots, replicas included: `apply` rejects replicas that do not fit
  in `(initial + max_extra) x agents_per_worker`, admission refuses children past it (`workers`),
  and placement opens no further slot. Types without `workers` keep v1's unbounded placement. The first child of a type with no worker waits for a job launch
  (about 30 to 45 s).
- **First prompt:** `prompt` in the spawn request is held as a pending prompt and delivered when
  the child's worker polls, like any prompt to an agent whose worker is not up yet.
- **Scaling** `replicas` never touches spawned agents. A child goes away when it is removed
  explicitly, when its parent goes away (removed, scaled down, its type or swarm removed), or with
  its type or swarm. Removal cascades to the whole subtree; every removed agent gets an unassign
  (final sync; its bucket state is kept) and its token is revoked.
- **Idle:** a child idles like any agent; an idle parent does not stop it.
- **Persistence:** spawned rows live in the controller's SQLite, in `FLOCK_DATA` (default
  `./data`; the controller image sets `/data`). Turn on the Space's persistent storage so `/data`
  survives redeploys: unlike replicas, spawned agents cannot be rebuilt from the spec.

### Ephemeral types

`ephemeral: true` on a type makes each spawned agent of it live for one session: a judge, an
interviewer, a one-shot helper. Added 2026-10-22 for experiments/chatbot-v1's interviewers.

```yaml
types:
  interviewer: {harness: llm, model: glm, replicas: 0, ephemeral: true,
                workers: {initial: 0, max_extra: 2}, max_turns: 26, max_budget_usd: 0.15}
spawn:
  dev: {types: [interviewer], max: 2}
```

- **When.** The controller removes the agent when a worker heartbeat reports it idle (or
  stuck) after at least one turn, with no prompt pending or just handed out. How the turn
  ended does not matter: `done`, `max_turns`, its budget or an error. A heartbeat's tokens
  include the last turn, so the cost it leaves behind is complete.
- **What.** The same removal as `DELETE /agents/{name}`: `child_removed {reason: ephemeral}` on
  the parent's stream, an unassign to its worker (final sync), its token revoked, its cost
  kept in the swarm's total (`removed_cost`), and its place under the parent's `spawn.max`
  freed, since caps count live children. The worker then deletes the agent's workdir, session
  log, state, tools pin and the harness's own session dir (`AgentRuntime.purge`).
- **After.** A new spawn is a new agent with a new name (the per-swarm counter) and a fresh
  session. A run bundle (`flock run`) keeps removed agents' events (`events/<name>.jsonl`) and
  lists them in `run.json` under `removed_agents`, each with its type, tokens and `cost_usd`
  (priced from its turns' tokens, as the swarm's total counts it).
- **Refused:** `ephemeral: true` with `replicas` above 0 (replicas are never removed).
- **A follow-up prompt** from the parent must arrive while the agent is still working; a
  prompt sent after its session ended gets 404, because the agent is gone.

## Names

`naming:` (swarm-level) picks how agents are named; it applies to replicas and spawned
children alike.

| `naming` | replicas | spawned children |
|---|---|---|
| `numeric` (default) | `<type>-NNNN` (the slot) | `<parent>-<type>-NNNN` (the per-swarm counter) |
| `words` | `<adjective>-<noun>`, e.g. `amber-otter` | the same, e.g. `brisk-lantern` |

```yaml
naming: words
types:
  interviewer: {name_prefix: "iv-", ...}   # iv-teal-heron: the role stays recognizable
```

- **Words.** About 190 adjectives (colors, simple qualities) and 340 nouns (animals, fruits,
  plants, everyday objects), lowercase, letters and hyphens only, so every name is a valid
  board handle. When two-word names get crowded, a third word is added.
- **Reserved.** The controller reserves each name in its store with the type and the slot (a
  replica) or the parent (a child). A name is unique on the controller and never reused while
  the swarm exists, not even after its agent is removed. A replica slot that is recreated
  (scale down and up, a cascade removal) keeps its name, also across controller restarts (the
  reservations are in `state.db`).
- **The counter.** Spawned children still take the per-swarm counter (`limits.spawned_total`
  counts it); with `words` it no longer appears in the name.
- **Everywhere.** The name is the agent's identity: board handle, events, run bundle, status,
  `X-Flock-Agent` for worlds, endpoint pinning (`base_urls`), `flock remove <name>`. A words
  name says nothing about the parent: a world that needs the parent uses `X-Flock-Type` and
  its own records (worlds/chatbot3 binds an interview booking to the first interviewer that
  uses it).
- The CLI tells an agent from a swarm by looking the name up (a swarm wins a tie).

## API

| Method | Path | Result |
|---|---|---|
| POST | `/agents/{parent}/spawn` `{type, prompt?, key?}` | 201 `{name, created: true, parent, type, depth}`. A repeated `key` for the same parent: 200 `{created: false, name, ...}` |
| | | 409 `{reason, detail}`, `reason` one of `type_not_allowed`, `max_children`, `depth`, `workers`, `alive`, `spawned_total`, `budget`, `type_crashlooping`, `parent_removing` |
| DELETE | `/agents/{name}` | 200 `{removed: [names]}`, children first. A replica: 409 `not_spawned` (scale its type) |
| GET | `/agents/{name}` | adds `parent`, `children`, `depth` |
| GET | `/swarms/{s}/agents?parent=<name>` | the direct children of one agent |
| POST | `/agents/{name}/children/prompts` `{type?, state?, text, timeout?, reply_to?, skip_pending?}` | the same prompt to every direct child (optionally one type / state) in one request; rules and counts as `POST /swarms/{s}/prompts` (`--state idle` skips children with a pending prompt): `{swarm, type, parent, matched, queued, pending, skipped_state, skipped_removing, skipped_pending}` |
| GET | `/whoami` | `{kind: controller}` or `{kind: agent, name, swarm, type, parent, depth, children}` |

Events: `spawned {child, type, depth, key}` and `child_removed {child, type, reason}` (`reason`:
`removed`, `parent_removed`, `type_removed`, `swarm_removed`) on the parent's stream, written by
the controller. The worker's `session_started` / `session_resumed` / `session_restarted` carry
`parent` on the child's stream.

## Agent tokens

Every agent gets its own token. The worker puts it into the harness environment with the
agent's identity:

| Variable | Value |
|---|---|
| `FLOCK_AGENT` | the agent's own name |
| `FLOCK_PARENT` | its parent's name, or empty |
| `FLOCK_URL` | the controller URL at harness start (direct calls only, see "CLI inside an agent") |
| `FLOCK_AGENT_TOKEN` | the agent's scoped token (direct calls only; the gateway uses the worker's copy) |
| `FLOCK_TOKEN` | empty: the worker's own token never reaches an agent |

An agent token may: read its own status and events and call `whoami`; spawn under itself and
bulk-prompt its own children; read, prompt, spawn under, bulk-prompt the children of and remove
any agent in its subtree; prompt its parent. Everything else is
403 (its parent's status, siblings, itself as a prompt or removal target, swarms, workers). The
controller token keeps full access.

The controller stores a hash and a nonce per agent, never the token: the token is an HMAC of the
agent's name and nonce under a key derived from `FLOCK_TOKEN`, so every assign of the agent (a
relaunched worker) carries the same value. Removing the agent revokes it. Rotating `FLOCK_TOKEN`
invalidates every agent token; agents get the new one with their next assign.

## CLI inside an agent

The worker image and the bootstrap put the `flock` CLI on PATH (same package). Inside an agent,
names are always agent names.

**Through the worker.** When the agent's pin file (`FLOCK_TOOLS_PIN`) names the worker's
gateway (every CLI harness: claude, codex, pi), each call goes there as `controller:request
{method, path, params, json}`. The worker accepts only the routes an agent token may use:

| Method | Path |
|---|---|
| GET | `/whoami`, `/agents/{name}`, `/agents/{name}/events`, `/swarms/{s}/agents` |
| POST | `/agents/{name}/spawn`, `/agents/{name}/prompt`, `/agents/{name}/children/prompts` |
| DELETE | `/agents/{name}` |

It sends the call with the agent's own token through its own controller client, which re-reads
the controller's address (`FLOCK_CTL_HOST_FILE` on Slurm) before every heartbeat and after any
connection error, and retries there. So a controller that moved host is reached at its new
address; `FLOCK_URL` in the harness environment, fixed when the harness started, is not used.
The controller still checks the token's scope (below). A call that still fails after the
retries (unreachable, or 5xx) is an error for the CLI and a `controller_call_failed {path,
error}` event on the agent's stream (`flock events <agent>`).

**Directly.** Without a gateway (the `llm` harness, or no pin file) the CLI reads `FLOCK_URL`
and `FLOCK_AGENT_TOKEN` (falling back to `FLOCK_TOKEN`), as before. Both stay in the harness
environment for that fallback; with the gateway they are not needed and can be dropped.

```sh
flock spawn analyst --prompt "..." --key a1   # prints the child's name; same key, same child
flock prompt <agent> "text"                # a child, a grandchild, or $FLOCK_PARENT
flock prompt --children --state idle "text"   # all your idle children, one request
flock status [<agent>]                     # JSON; yourself by default
flock events <agent> --after N --wait      # blocks until something new (or --timeout, 600 s)
flock remove <agent>                       # a descendant and its subtree
```

The CLI retries HTTP 429 and 502/503/504 (and, for reads, 500 and dropped connections) up to
5 attempts with exponential backoff and jitter, never sooner than `Retry-After`, so a thousand
agents calling it through a Space edge do not fail on throttling.

For the operator (controller token), `flock spawn <type> --parent <agent>` spawns on an agent's
behalf, `flock agents <swarm> --parent <agent>` lists children, and `status` / `remove` take an
agent name when the name is not a swarm.

## Snippet for a leader's system prompt

```markdown
## Child agents

You can create agents to work in parallel. Each child has its own workdir and session and
keeps them until removed; it does nothing until prompted.

- `flock spawn <type> --prompt "<task>" --key <short-unique-key>` creates a child and prints
  its name. Reusing a key returns the same child, so retrying is safe.
- `flock events <child> --after <N> --wait` blocks until the child does something; the cursor for
  the next call is printed on stderr. A `turn_ended` event means it finished a turn.
- `flock prompt <child> "<text>"` sends a follow-up; `flock status <child>` shows its state.
- `flock remove <child>` removes it and everything it spawned. Remove children you no longer
  need: you have a fixed number of places.
- Children report back with `flock prompt $FLOCK_PARENT "<result>"`; those reports arrive as
  your next prompts.
- A refusal prints a reason: `max_children` (remove a finished child first), `depth`,
  `type_not_allowed`, `alive`, `budget`, `workers` or `spawned_total` (swarm limits: stop
  spawning).
```

## Not yet

- Pre-warming is a possible later feature. Today a worker is opened only for agents that
  exist, so the first spawn of a type with no agents pays the worker's boot time (a job launch,
  about 30 to 45 s), whatever `workers.initial` says.
- The instructions above are not appended to CLAUDE.md / AGENTS.md automatically; put them in the
  type's `system_prompt_file`, or in a `system_prompt_append_file` that `flock launch` appends to
  a shared prompt.
- `limits.budget_usd` is checked at spawn time only; it does not stop agents that keep working.
- Spawned rows are not mirrored to the bucket. If the controller loses its SQLite (no persistent
  storage), workers report agents the controller does not know and the controller unassigns them.
- No cap on name length and no caller-chosen child names.
