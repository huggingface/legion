# Spawning agents

A running agent (a "leader") can create child agents of other roles, prompt them, read their
status and remove them. Children are ordinary agents: they have a Trial, a session, a worker, a
state and a cost like any replica, and they idle and resume the same way. What is new is a
`parent`, a `depth`, and a token that lets the agent call the controller as itself. Agents do
it with the `spawn.*` tools ("Harbor agents" below). Design: [design.md](design.md#spawn).

## Spec

```yaml
agents:
  roles:                                 # a Harbor spec's roles become flock types
    leader:  {count: 1, workers: {initial: 1, max_extra: 0}}
    analyst: {count: 0, workers: {initial: 1, max_extra: 3}}
    helper:  {count: 0, workers: {initial: 0, max_extra: 2}}
spawn:                                   # run policy, separate from what a role is
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
| `types.<t>.workers.initial` | `ceil(replicas / agents_per_worker)` | worker slots counted toward the cap. Not pre-created: a worker opens only when an agent needs one |
| `types.<t>.workers.max_extra` | 0 | worker slots spawned children of `<t>` may add on demand |
| `limits.alive` | none | agents that exist in the swarm, replicas and spawned together |
| `limits.spawned_total` | none | children ever spawned in the swarm (the name counter); removing one does not free a place |
| `limits.budget_usd` | none | cost of every agent of the swarm, removed ones included. Checked when spawning, and a hard stop for the swarm (docs/loop.md "Budget stop") |

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
- **Placement:** on the child type's own workers with the default fill rule (fill in order, never move,
  open a new worker on demand). A type that declares `workers` never gets more than
  `initial + max_extra` worker slots, replicas included: `apply` rejects replicas that do not fit
  in `(initial + max_extra) x agents_per_worker`, admission refuses children past it (`workers`),
  and placement opens no further slot. Types without `workers` have unbounded placement. The first child of a type with no worker waits for a job launch
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
interviewer, a one-shot helper.

```yaml
types:
  interviewer: {replicas: 0, ephemeral: true, workers: {initial: 0, max_extra: 2}, ...}
spawn:
  dev: {types: [interviewer], max: 2}
```

A Harbor spec (flock.harbor.spec) does not expose `ephemeral` yet: it is a key of the compiled
type.

- **When.** The controller removes the agent when a worker heartbeat reports it idle (or
  stuck) after at least one turn, with no prompt pending or just handed out. How the turn
  ended does not matter: `done`, `max_turns`, its budget or an error. A heartbeat's tokens
  include the last turn, so the cost it leaves behind is complete.
- **What.** The same removal as `DELETE /agents/{name}`: `child_removed {reason: ephemeral}` on
  the parent's stream, an unassign to its worker (final sync), its token revoked, its cost
  kept in the swarm's total (`removed_cost`), and its place under the parent's `spawn.max`
  freed, since caps count live children. The worker then deletes the agent's workdir, session
  log, state and the harness's own session dir (`AgentRuntime.purge`).
- **After.** A new spawn is a new agent with a new name (the per-swarm counter) and a fresh
  session. A run bundle (`flock launch`) keeps removed agents' events (`events/<name>.jsonl`) and
  lists them in `run.json` under `removed_agents`, each with its type, tokens and `cost_usd`
  (priced from its turns' tokens, as the swarm's total counts it).
- **Refused:** `ephemeral: true` with `replicas` above 0 (replicas are never removed).
- **A follow-up prompt** from the parent must arrive while the agent is still working; a
  prompt sent after its session ended gets 404, because the agent is gone.

## Names

`naming:` picks how agents are named; it applies to replicas and spawned children alike. A
Harbor spec takes `words` (its default) or `consecutive-id`; it cannot change in a running swarm.

| `naming` | replicas | spawned children |
|---|---|---|
| `words` (a Harbor spec's default) | `<adjective>-<noun>`, e.g. `amber-otter` | the same, e.g. `brisk-lantern` |
| `consecutive-id` (Harbor) | `agent-001`, ... numbered across the roles in order | `<parent>-<type>-NNNN` |
| `numeric` (a swarm spec's default) | `<type>-NNNN` (the slot) | `<parent>-<type>-NNNN` (the per-swarm counter) |

**For task authors:** agent names are opaque strings. Verifiers and instructions must not
assume a format; flock's default names are random words. `naming: consecutive-id` exists for tasks
that already expect `agent-NNN` (experiments/mini-swarm). With it, only the last role's count
may change in a running swarm: names run across the roles, so an earlier role's change would
rename running agents, and `flock update` refuses it. With words, each agent's instructions list
its teammates and their roles as the controller has named them when the agent's Trial starts:
every role's replicas, a ramp's up to its max. Replicas added later by `flock update` are
missing from the instructions of agents already running (they still meet on the board).

```yaml
naming: words
types:
  interviewer: {name_prefix: "iv-", ...}   # iv-teal-heron: the role stays recognizable
```

A Harbor spec sets it per role (`agents.roles.<role>.name_prefix`, words only), e.g.
`steward: {name_prefix: steward-, ...}` in experiments/pybot. Changing it with `flock update`
affects only new names: running agents and reserved names keep theirs.

- **Words.** About 190 adjectives (colors, simple qualities) and 340 nouns (animals, fruits,
  plants, everyday objects), lowercase, letters and hyphens only, so every name is a valid
  board handle. When two-word names get crowded, a third word is added.
- **Reserved.** The controller reserves each name in its store with the type and the slot (a
  replica) or the parent (a child). A name is unique on the controller and never reused while
  the swarm exists, not even after its agent is removed. A replica slot that is recreated
  (scale down and up, a cascade removal) keeps its name, also across controller restarts (the
  reservations are in `state.db`).
- **The counter.** Spawned children still take the per-swarm counter (`limits.spawned_total`
  counts it); with `words` it is not part of the name.
- **Everywhere.** The name is the agent's identity: board handle, events, run bundle, status,
  `X-Flock-Agent` for worlds, `flock-agent remove <name>`. A words name says nothing about the
  parent: a world that needs it keeps its own records (e.g. a world that books work to the first agent
  that uses it).
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

Every agent gets its own token. The worker keeps it (with `FLOCK_AGENT`, `FLOCK_PARENT`) and
sends the agent's controller calls with it; it never enters a Trial's container.

An agent token may: read its own status and events and call `whoami`; spawn under itself and
bulk-prompt its own children; read, prompt, spawn under, bulk-prompt the children of and remove
any agent in its subtree; prompt its parent. Everything else is
403 (its parent's status, siblings, itself as a prompt or removal target, swarms, workers). The
controller token keeps full access.

The controller stores a hash and a nonce per agent, never the token: the token is an HMAC of the
agent's name and nonce under a key derived from `FLOCK_TOKEN`, so every assign of the agent (a
relaunched worker) carries the same value. Removing the agent revokes it. Rotating `FLOCK_TOKEN`
invalidates every agent token; agents get the new one with their next assign.

The worker accepts only the routes an agent token may use (`controller_call` in
flock.worker.main):

| Method | Path |
|---|---|
| GET | `/whoami`, `/agents/{name}`, `/agents/{name}/events`, `/swarms/{s}/agents` |
| POST | `/agents/{name}/spawn`, `/agents/{name}/prompt`, `/agents/{name}/children/prompts` |
| DELETE | `/agents/{name}` |

It sends the call through its own controller client, which re-reads the controller's address
(`FLOCK_CTL_HOST_FILE` on Slurm) before every heartbeat and after any connection error, and
retries there; a POST that may have reached the controller is not sent again. A call that still
fails (unreachable, or 5xx) is an error for the tool and a `controller_call_failed {path,
error}` event on the agent's stream (`flock status <agent> --events`).

## Operator CLI

With the controller token, `flock-agent spawn <type> --parent <agent>` spawns on an agent's
behalf, `flock-agent remove <agent>` removes a descendant and its subtree, `flock status <swarm>
--parent <agent>` lists children, and `status` takes an agent name when the name is not a
swarm. The CLI retries HTTP 429 and 502/503/504 (and, for reads, 500 and dropped connections) up
to 5 attempts with exponential backoff and jitter, never sooner than `Retry-After`.

## Harbor agents

An agent runs in a Trial (the task's container), so it calls the controller through the `spawn`
resource and the `spawn.*` tools
(`flock.harbor.builtin`, copied into the run's `resources/` and `tools/` like `message_board`):

```yaml
resources: {spawn: {}}
tools: [spawn.*]
agents:
  roles:
    leader: {count: 1}
    helper: {count: 0, workers: {initial: 1, max_extra: 2}}
spawn: {leader: {types: [helper], max: 3}}
limits: {spawned_total: 50}
```

| Tool | Call |
|---|---|
| `spawn.spawn {type, prompt?, key?}` | `POST /agents/<me>/spawn`: the ack `{name, created, parent, type, depth}`, or the refusal's `reason: detail` |
| `spawn.prompt {agent, text}` | `POST /agents/<agent>/prompt`: a child, a descendant, or the parent |
| `spawn.status {agent?}` | `GET /agents/<agent>` (yourself by default: `parent`, `children`, `depth`) |
| `spawn.remove {agent}` | `DELETE /agents/<agent>`: a descendant and its subtree |

- **Auth.** Each agent's Trial mounts its own Unix socket (`/run/flock-spawn.sock`), served on
  the worker. The worker sends each call with that agent's own token (the route allow-list
  above): the socket, never the request, decides whose token is used, and the controller
  checks the token's scope and the caps (`spawn:`, `workers`, `limits`) exactly as for any
  other agent. No token enters the container.
- **Tools per role.** A role without a `spawn:` rule is not shown `spawn.spawn` and
  `spawn.remove` (the controller refuses them anyway); every role keeps `spawn.prompt` and
  `spawn.status`, so a child reports to its parent (`spawn.status` names it) with `spawn.prompt`.
- **Children** are Harbor agents of the child role, placed and started by the controller as
  above, one Trial each.
- **Backends:** docker and podman. `hf-sandbox` has no mounts: `resources.spawn` is refused there.

## Not yet

- Pre-warming is a possible later feature. Today a worker is opened only for agents that
  exist, so the first spawn of a type with no agents pays the worker's boot time (a job launch,
  about 30 to 45 s), whatever `workers.initial` says.
- If the controller loses its SQLite (no persistent storage and no state sync), workers report agents the controller does not know and the controller unassigns them.
- No cap on name length and no caller-chosen child names.
