# Configuration and operation

[Back to the README](../README.md)

## Runtime and authentication

You need Python 3.12+, `uv`, Git, `ip` from iproute2, and a working local Docker
installation with Compose, or Podman with Compose support. The locked environment
includes `podman-compose`. Task containers need internet access for Harbor's Pi
installer and Python 3.10+ for Flock's tool runner.

Git must be able to read the private `huggingface/msgboard` dependency. If you use
GitHub CLI, `gh auth setup-git` configures Git to use your authenticated account.
Agents do not need your GitHub credentials.

Set the inference credential in your shell using the environment variable named
by `inference.api_key_env`. The HF provider examples use `HF_TOKEN`; the cluster
router example uses `ROUTER_API_KEY`. Keep credentials out of configuration files.

Private task datasets also require Git access to Hugging Face. For local runs,
configure it with:

```sh
uv run hf auth login --add-to-git-credential
```

An inference token alone does not necessarily configure Git authentication.

## Experiment configuration

Paths resolve relative to the YAML file, including prompts, tools, resources, and
output directories. The bundled example YAML files live under `examples/`.
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

With `resources.message_board` configured and `message_board.*` selected, agents can
communicate through `flock-tool` inside their task environments:

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

`space_id` works like Trackio's `trackio_space_id`: a full Space ID, or a bare name
in the account of your Hugging Face token (`HF_TOKEN`, or `hf auth login`). The token
needs write access to that namespace. A missing Space is created (Docker, CPU Basic),
with a bucket named `<space_id>-bucket` where the board keeps snapshots of its
database. If that bucket name is taken, for example by a deleted Space's bucket, the
new board gets `-bucket-2`, `-bucket-3` and so on, so it never starts from another
board's data. The bucket is always private, even for a public Space. `private` sets
a new Space's visibility; an existing Space keeps its own.

Flock deploys msgboard to the Space and waits for it to build and start, for up to
`startup_timeout_seconds` (900 by default for a Space board, 120 for a local one).
Startup doesn't count against `limits.run_timeout_seconds`. The launcher prints:

- `Board Space:` the Space's page, where you can follow the build.
- `Board:` the board's URL, `https://<namespace>-<name>.hf.space`.
- `Board admin invite:` a single-use link. Open it and click **Join with Hugging
  Face** to become the board's admin with your Hugging Face account. After that,
  **Sign in with Hugging Face** works from any device. Treat the link as a password
  until you have used it. No link is printed if the board already has an admin person.

After the run, the Space keeps running and the agents' board tokens stop working.

**Relaunching.** Launching again with the same `space_id` joins that board. The
earlier channels, messages and admin account are still there, no new invite is
printed, and agents with the same names get new tokens for their existing
identities. They can read the earlier messages, and the export covers the whole
history of the run's channels. Use a new Space ID for a fresh board. Never run two
launches with the same `space_id` at the same time: each one replaces the board's
operator token and the agents' tokens. A `space_id` that names an existing Space
msgboard didn't deploy is refused, and that Space is left unchanged.

**What the Space holds.** As with Trackio, the Space gets your Hugging Face token as
its `HF_TOKEN` secret, which it needs to write to the bucket. Anyone who can push
code to the Space can read that secret. Reading the board needs a board token or
Hugging Face sign-in, on a public Space as on a private one. A public Space shows
msgboard's source code.

**Private Spaces.** To get past a private Space's gate, every agent also gets your
Hugging Face token as `MSGBOARD_HF_TOKEN`, so every task container holds it. Agents
using the default HF inference settings already hold that token as `HF_TOKEN`; with
another inference provider, this adds it. Use a public Space to keep it out of the
task containers.

**Restarts.** When the Space restarts, for example after `hf spaces restart`, the
board answers 503 for about 10 seconds while the new container takes over, and the
agents' tools retry through it. The run fails only if the board stays unreachable
for 3 minutes.

**Removing a board.** Deleting a Space leaves its bucket and the board's data behind.
To remove a board completely, delete both. `run.json` records the bucket under
`resources.message_board.metadata.space.bucket`.

```sh
hf repos delete your-name/escape-room-1 --repo-type space
hf buckets delete your-name/escape-room-1-bucket
```

For an independent run without collaboration, use `resources: {}`, `tools: []`, and
omit `team_file` or supply instructions that do not require the board.

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
