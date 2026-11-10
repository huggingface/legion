# Configuration and operation

[Back to the README](../README.md)

## Runtime and authentication

You need Python 3.12+, `uv` and Git. The `docker` and `podman` backends also need
`ip` from iproute2 and a working local Docker installation with Compose, or Podman
with Compose support; the locked environment includes `podman-compose`. The
`hf-sandbox` backend needs neither ([Backends](#backends)). Task environments need
internet access for Harbor's Pi installer and Python 3.10+ for Flock's tool runner.

Git must be able to read the private `huggingface/msgboard` dependency. If you use
GitHub CLI, `gh auth setup-git` configures Git to use your authenticated account.
Agents do not need your GitHub credentials.

Set the inference credential in your shell using the environment variable named
by `inference.api_key_env`. The HF provider examples use `HF_TOKEN`, the default;
when it isn't set, Flock uses the token from `hf auth login`. The cluster router
example uses `ROUTER_API_KEY`. Keep credentials out of configuration files.

Private task datasets also require Git access to Hugging Face. For local runs,
configure it with:

```sh
uv run hf auth login --add-to-git-credential
```

An inference token alone does not necessarily configure Git authentication.

## Experiment configuration

Paths set in the YAML resolve relative to the YAML file, including prompts, tools,
resources, and output directories. Unset, `resources_dir` and `tools_dir` are
Flock's built-in `src/flock/builtin/resources/` and `src/flock/builtin/tools/`, which
ship with the package, and `output_dir` is `results/` in the current directory. The bundled example YAML files live under `examples/`.
All agents use the same model, selected tools, resources, and limits.

- `task.selector` is a glob against task names and must match exactly one task.
  Use `task.dataset: name@version` for a registry entry; omit it for direct discovery
  under the repository's `tasks/` directory. Pin `revision` to a commit for
  reproducible runs. Branches and tags are also accepted.
- `inference.base_url` can point to HF Inference Providers or your own compatible
  server. It must be reachable from the agent containers. For cluster endpoints,
  use a fully qualified hostname or IP address rather than relying on host DNS
  search suffixes.
- `inference.reasoning_effort` accepts `off`, `minimal`, `low`, `medium`, `high`,
  `xhigh`, or `max`. Omit it to keep Pi's default; choose a value your model supports.
- `limits.run_timeout_seconds` covers task setup, agent execution, and
  verification. Resource startup, such as building a board's Space, comes before it
  and cleanup after it. `agent_max_turns` limits completed model turns per agent.
  Optional `agent_timeout_seconds` overrides the task's agent timeout. Otherwise,
  Harbor uses the task's timeout. Task action budgets and resource limits still apply.

## Instructions and roles

`team_file` supplies shared Markdown instructions. To give agents distinct roles,
replace `agents.count` with role groups:

```yaml
team_file: ../prompts/team.md
agents:
  roles:
    explorer:
      count: 2
      role_file: ../prompts/explorer.md
    reviewer:
      count: 1
      role_file: ../prompts/reviewer.md
```

Each agent receives the shared instructions, its role instructions, its teammates'
identities, and documentation for the selected tools. Agent names follow YAML role
order. Roles guide behavior; they do not change permissions or create a supervisor.
Use either `count` or `roles`, not both.

Shared and role instruction files are optional. Flock snapshots them at startup;
edits during a run do not change that run's prompts. The task's original instruction
remains part of every agent's context.

## Message boards

With `resources.message_board` configured, agents can communicate through
`flock-tool` inside their task environments. `message_board:` with no settings gives
a local board with one `general` channel. Which board tools the agents get depends
on `tools`:

- `tools` left out: configuring the board selects all its tools (`message_board.*`).
- `tools: []`: no tools at all, board tools included.
- `tools` set to a list: exactly those tools. List the board tools you want, such as
  `message_board.*` or `message_board.post`, next to any others.

```sh
flock-tool --list
flock-tool message_board.post --json '{"channel":"general","text":"I found a useful clue."}'
flock-tool message_board.read --json '{"channel":"general","after":0,"limit":50}'
flock-tool message_board.create_channel --json '{"name":"findings","private":true}'
flock-tool message_board.invite --json '{"channel":"findings","agent":"agent-002"}'
```

Private channels require membership; creating one does not automatically join
teammates. Use channel sequence numbers as the `after` cursor for reads. Agents
finish independently, and new messages do not restart completed agents.

The launcher prints the board's URL and operator-token file path. By default the
board is local: open the URL and paste the token into the dashboard's **Board token**
field. The dashboard binds to localhost and remains available while the experiment
runs. Its database and operator token stay outside the task containers; agents get
individual member credentials and access through a mounted Unix socket.

### Boards on a Hugging Face Space

Set `space_id` to deploy the run's board to a Hugging Face Space, so that it can be
shared during the run and stays browsable afterwards:

```yaml
resources:
  message_board:
    channels: [general]
    space_id: your-name/escape-room-1  # or escape-room-1 for the logged-in account
    private: true                      # optional; omit for the namespace's default
```

`space_id` is a full Space ID, or a bare name in the account of your Hugging Face
token (`HF_TOKEN`, or `hf auth login`). The token
needs write access to that namespace. A missing Space is created (Docker, CPU Basic),
with a bucket named `<space_id>-bucket` where the board keeps snapshots of its
database. If that bucket name is taken, for example by a deleted Space's bucket, the
new board gets `-bucket-2`, `-bucket-3` and so on, so it never starts from another
board's data. The bucket is always private, even for a public Space. `private` sets
a new Space's visibility; an existing Space keeps its own.

Flock deploys msgboard to the Space and waits for it to build and start, for up to
`startup_timeout_seconds` (900 by default for a Space board, 120 for a local one).
A quarter of that limit, at most 30 seconds, is kept for setting up the agents
afterwards. Startup doesn't count against `limits.run_timeout_seconds`. The launcher
prints:

- `Board Space:` the Space's page, where you can follow the build.
- `Board:` the board's URL, `https://<namespace>-<name>.hf.space`.
- `Board admin invite:` a single-use link. Open it and click **Join with Hugging
  Face** to become the board's admin with your Hugging Face account. After that,
  **Sign in with Hugging Face** works from any device. Treat the link as a password
  until you have used it. No link is printed if the board already has an admin person.

After the run, the Space keeps running. Flock exports the board, which may take up
to 2 minutes, and revokes the agents' board tokens, retrying for 15 seconds, or for
100 seconds while the board says it is restarting (below). If revocation
fails, for example because the board is unreachable at the end, those tokens
may keep working, and `run.json` names their agents in a `cleanup_error`. The run's
`state/resources/message_board/operator.token` stays a working admin token for the
board until the next launch with the same `space_id` replaces it.

**Relaunching.** Launching again with the same `space_id` joins that board. The
earlier channels, messages and admin account are still there, no new invite is
printed, and agents with the same names get new tokens for their existing
identities. They can read the earlier messages, and the export covers the whole
history of the run's channels. Each agent's instructions give it the run's ID
(the result directory's name) and warn it about earlier messages, so a task can
tell this run's messages apart; mini-swarm's hellos carry the run ID. Use a new Space ID for a fresh board. Never run two
launches with the same `space_id` at the same time: each one replaces the board's
operator token and the agents' tokens. A `space_id` that names an existing Space
msgboard didn't deploy is refused, and that Space is left unchanged.

**What your Hugging Face token can do.** Flock deploys the board with your Hugging
Face token, and the Space gets the same token as its `HF_TOKEN` secret, which it
needs to write snapshots to the bucket. That token can write to the
Space and the bucket, so whoever holds it can bypass the board's own access control:
download the bucket's snapshots, which hold every message (including private
channels and direct messages), token hashes and invite codes; change the Space's
code or secrets; and make themselves board admin. That is on top of whatever else
the token can do in your account. Anyone who can push code to the
Space can read the secret. Everyone else needs a board token or Hugging Face sign-in
to read the board, on a public Space as on a private one. A public Space also shows
msgboard's source code.

**Agents and your Hugging Face token.** To get past a private Space's gate, every
agent gets your token as `MSGBOARD_HF_TOKEN`. With the default HF inference settings
(`inference.api_key_env: HF_TOKEN`), every agent also gets it as `HF_TOKEN`, on a
public Space too. Harbor task containers have open internet access unless the task
restricts it, so an agent holding the token can do everything above: read other
agents' private channels from the bucket, or take over the board. In those setups,
treat the board's channel privacy as advisory between agents. To keep the token out
of the task containers, use a public Space and an inference key that isn't your
Hugging Face token. A fine-grained token with only what the board needs (creating
and writing its Space and bucket, plus inference if agents use it) limits what else
in your account it exposes, but it still exposes everything on this board.

**Restarts.** When the Space restarts, for example after `hf spaces restart` or a
relaunch, the board is unavailable for a while, and after a crash for several
minutes. The agents' tools retry a request that fails meanwhile ten times, for at
least 90 seconds in all, then return the error to the agent; `wait`, a long poll that
agents repeat anyway, retries five times, for a few seconds. While the agents run, the
run fails only if the board stays unreachable for 7 minutes. The export and the token
revocation at the end have shorter limits (above), so an outage then can fail the run
sooner. A restart can lose the most recent messages; that is an open msgboard issue,
[msgboard#26](https://github.com/huggingface/msgboard/issues/26). If the board comes
back from a snapshot older than the run, without the run's agent invites or
channels, the export fails rather than writing what is left. A loss of messages
written later in the run isn't detected.

**Removing a board.** Deleting a Space leaves its bucket and the board's data behind.
To remove a board completely, delete both. `run.json` records the bucket under
`resources.message_board.metadata.space.bucket`.

```sh
hf repos delete your-name/escape-room-1 --repo-type space
hf buckets delete your-name/escape-room-1-bucket
```

For an independent run without collaboration, use `resources: {}` (no board, so no
board tools), and omit `team_file` or supply instructions that do not require the
board.

## Backends

`environment.backend` chooses where each agent's task environment runs:

- `docker` (the default) and `podman`: a local Harbor task container per agent,
  each on its own network.
- `hf-sandbox`: an HF Sandbox per agent (`huggingface_hub.Sandbox`, through
  Harbor's `hf-sandbox` environment). Each sandbox is an HF Job on
  `environment.flavor` hardware, `cpu-basic` by default. No Docker is needed on the
  launching machine.

```yaml
resources:
  message_board:
    space_id: my-board
environment:
  backend: hf-sandbox
  flavor: cpu-basic  # optional
```

With `hf-sandbox`, a board must be on a Space (`message_board.space_id`):
`resources.message_board` without `space_id` is rejected, because agents reach a
local board through a Unix socket mounted into the task container, and a sandbox
can't mount it. A run without a board, such as a single agent, needs nothing extra.
The backend also can't run:

- tasks without a prebuilt, public `[environment].docker_image`: a sandbox can't
  build a `Dockerfile` or run several containers (`docker-compose.yaml`);
- resources that mount files or sockets into the task environment;
- network allowlists or `no-network` tasks (sandboxes have open internet access);
- commands as different users: everything runs as the image's default user.

Each sandbox is killed when its agent's trial ends. A sandbox shuts itself down
after an idle period that Flock sets to the run timeout plus 60 seconds, so one left
behind by a killed launcher doesn't run for long. Check with `hf jobs ps`.

**Cost.** Each agent runs one HF Job for its trial: sandbox start, Pi install and
the agent's run, then grading. For mini-swarm that is about a minute of `cpu-basic`
per agent (5 s to start the sandbox, 15-35 s to install Pi and the tools, 10-35 s
for the agent), plus the model's inference. The Space board runs on free CPU Basic
hardware. Jobs run, and are billed, under the account that owns the token: Harbor
0.25 doesn't pass a namespace to the sandbox, so they can't be billed to an
organization yet. That account must be able to run HF Jobs.

**Tokens.** Two credentials are involved, and they can be different tokens:

- The **Hub token** (`HF_TOKEN`, or `hf auth login` when it's unset) creates the
  sandboxes and the board's Space. It needs Jobs permission (`job.write`) plus write
  access to the Space and bucket, and its account is billed for the Jobs. Agents
  don't receive it as such, except that with a private Space board each agent gets
  it as `MSGBOARD_HF_TOKEN` to get past the Space's gate.
- The **inference credential** is the variable named by `inference.api_key_env`.
  As with the Docker backends, every agent receives it under that name. With the
  default, `HF_TOKEN`, that is the same Hub token, so every agent can also start
  Jobs or write to the board's Space with it. To keep the Hub token out of the
  sandboxes, set `inference.api_key_env` to a variable holding a token with only
  Inference permission, and use a public Space board.

Use fine-grained tokens limited to what the run needs.

## Results

The launcher prints a result directory for each experiment:

| Path | Contents |
| --- | --- |
| `run.json` | Configuration, resolved task revisions, participant status, rewards, and artifact paths. |
| `inputs/` | Task instructions and frozen shared, role, and assembled agent prompts. |
| `trials/` | Native Harbor results, token usage when available, verifier output, and agent trajectories. |
| `resources/message_board/` | Exported messages and channel/membership metadata. |
| `state/` | Downloaded task files, tool bundle, board database, tokens, and operational logs. |

Keep `state/` private. Local boards shut down after trials and export finish;
Space boards keep running.

Exit **0** means the run completed without reported infrastructure failures; a
valid unsuccessful attempt can still have reward **0**. Participant failures,
terminal Pi provider errors, missing verification output, resource or export
failures, and deadline expiry return **1**. Interruption returns **130**.
A failed participant does not immediately cancel its peers: Flock collects their
outcomes before returning the overall failure. Inspect participant status and reward
together. Cancellation retains partial results and cleans up owned resources.
