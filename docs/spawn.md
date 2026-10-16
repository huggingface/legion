# Spawning agents

A running agent (a "leader") can create child agents of other types, prompt them, read their
event streams and remove them. Children are ordinary agents: they have a harness session, a
workdir, a worker, a state and a cost like any replica, and they idle and resume the same way.
What is new is a `parent`, a `depth`, and a token that lets the agent call the controller as
itself.

Example spec: [`examples/spawn.yaml`](../examples/spawn.yaml). Design notes and decisions:
[design.md §13](design.md#13-spawn).

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
| `types.<t>.workers.initial` | v1's count: `ceil(replicas / agents_per_worker)` | worker slots the type starts with |
| `types.<t>.workers.max_extra` | 0 | worker slots spawned children of `<t>` may add on demand |
| `limits.alive` | none | agents that exist in the swarm, replicas and spawned together |
| `limits.spawned_total` | none | children ever spawned in the swarm (the name counter); removing one does not free a place |
| `limits.budget_usd` | none | cost of every agent of the swarm, removed ones included. Checked when spawning; it does not stop running agents |

Caps count a parent's existing children, so removing a child frees its place. A child's own
per-prompt `max_turns` and `max_budget_usd` come from its type, as for replicas.

`flock apply` (and the controller) reject: an unknown type in `spawn:`, a cycle (a type that can
reach itself, including spawning its own type), a spawnable type without `workers`, the list form
without `max`, an unknown key under `spawn.<type>`, and negative `depth`, caps, `workers` or
limits. The controller also refuses (409) a re-apply that lowers a type's `initial + max_extra`
below the workers its agents occupy: agents never move, so remove agents first.

**Depth.** A replica's depth is its type's `spawn.depth` (0 if absent), read from the current
spec. A child is born with `min(parent's depth - 1, its own type's spawn.depth)` (0 if its type
is absent from `spawn:`) and keeps it. With no cycles and finite depth, recursion ends by
construction.

## Lifecycle

- **Name:** `<parent>-<type>-<NNNN>`, `NNNN` from a per-swarm counter that is never reused while
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

## API

| Method | Path | Result |
|---|---|---|
| POST | `/agents/{parent}/spawn` `{type, prompt?, key?}` | 201 `{name, created: true, parent, type, depth}`. A repeated `key` for the same parent: 200 `{created: false, name, ...}` |
| | | 409 `{reason, detail}`, `reason` one of `type_not_allowed`, `max_children`, `depth`, `workers`, `alive`, `spawned_total`, `budget`, `type_crashlooping`, `parent_removing` |
| DELETE | `/agents/{name}` | 200 `{removed: [names]}`, children first. A replica: 409 `not_spawned` (scale its type) |
| GET | `/agents/{name}` | adds `parent`, `children`, `depth` |
| GET | `/swarms/{s}/agents?parent=<name>` | the direct children of one agent |
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
| `FLOCK_URL` | the controller URL |
| `FLOCK_AGENT_TOKEN` | the agent's scoped token |
| `FLOCK_TOKEN` | empty: the worker's own token never reaches an agent |

An agent token may: read its own status and events and call `whoami`; spawn under itself; read,
prompt, spawn under and remove any agent in its subtree; prompt its parent. Everything else is
403 (its parent's status, siblings, itself as a prompt or removal target, swarms, workers). The
controller token keeps full access.

The controller stores a hash and a nonce per agent, never the token: the token is an HMAC of the
agent's name and nonce under a key derived from `FLOCK_TOKEN`, so every assign of the agent (a
relaunched worker) carries the same value. Removing the agent revokes it. Rotating `FLOCK_TOKEN`
invalidates every agent token; agents get the new one with their next assign.

## CLI inside an agent

The worker image and the bootstrap put the `flock` CLI on PATH (same package). It reads
`FLOCK_URL` and `FLOCK_AGENT_TOKEN` (falling back to `FLOCK_TOKEN`). Inside an agent, names are
always agent names.

```sh
flock whoami                               # name, parent, depth, children
flock spawn analyst --prompt "..." --key a1   # prints the child's name; same key, same child
flock prompt <agent> "text"                # a child, a grandchild, or $FLOCK_PARENT
flock status [<agent>]                     # JSON; yourself by default
flock events <agent> --after N --wait      # blocks until something new (or --timeout, 600 s)
flock remove <agent>                       # a descendant and its subtree
```

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

- Pre-warmed child slots (idle children created ahead of a spawn) are a possible later feature.
  Today, `workers.initial >= 1` on the child type keeps a worker ready, so the first child skips
  the job launch.
- The instructions above are not appended to CLAUDE.md / AGENTS.md automatically; put them in the
  type's `system_prompt_file`.
- `limits.budget_usd` is checked at spawn time only; it does not stop agents that keep working.
- Spawned rows are not mirrored to the bucket. If the controller loses its SQLite (no persistent
  storage), workers report agents the controller does not know and the controller unassigns them.
- No cap on name length and no caller-chosen child names.
