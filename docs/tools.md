# Tools folder

An experiment's tools are plain Python functions in a folder next to the spec. `flock apply`
reads the folder, ships it to the workers by content hash, and the agents call the functions as
tools. Editing a tool never touches flock's code, wheel or image: change the file, run
`flock apply` again, and running agents use the new version from their next turn.

```
experiment/
  spec.yaml
  tools/
    workspace.py        # ws_write, run_bot: tools
    _helpers.py         # leading underscore: a helper module, imported by the tools
    requirements.txt    # optional: pip requirements, installed per version on the worker
```

```yaml
# spec.yaml
types:
  dev:
    harness: llm
    tools: [ws_write, run_bot]     # or `all`
```

```console
$ flock apply experiment/spec.yaml
tools: version 3f9c0e2a7b61d4c8 (2817 bytes): ws_write, run_bot
```

A runnable version of this example is in [`examples/tools-demo/`](../examples/tools-demo/).

## Contract

**Modules.** Every `*.py` file at the top of the folder whose name does not start with `_` is a
tool module. The folder is first on `sys.path` when a module is imported, so modules can
import each other and their `_helpers`. A module must not share its name with a standard
library or installed module (`json.py`, `math.py`): `flock apply` refuses it.

**Tools.** Every function defined in a tool module (not imported into it) whose name does not
start with `_` and that has a docstring is a tool. A function without a docstring is a helper.

- name: the function name (`[A-Za-z_][A-Za-z0-9_]*`, at most 64 characters; `bash` and `done`
  are reserved for the `llm` harness's own tools; names are unique across the folder).
- description: the docstring's first paragraph.
- parameters: a JSON schema built from the type hints, `additionalProperties: false`. Per-parameter
  descriptions come from a Google-style `Args:` section.
- readonly: optional, default false. Decorate the function with `@flock.tools.readonly` when the
  tool changes nothing (a status or a listing); the manifest then has `readonly: true` (the key is
  left out when false, so existing folders keep their version). A loop turn whose calls are all
  read-only is idle for `while.idle_rounds` (docs/loop.md). The comms `read` and `search`
  built-ins are read-only too.

```python
from flock.tools import readonly


@readonly
def task_board() -> str:
    """Show the open tasks and who claimed them."""
    ...
```

| Annotation | Schema |
|---|---|
| `str`, `int`, `float`, `bool` | `string`, `integer`, `number`, `boolean` |
| `list[X]` (or `list`) | `array` with `items` from `X` |
| `dict` (or `dict[str, X]`) | `object` |
| `Literal["a", "b"]` | `enum` |
| `X \| None`, `Optional[X]` | the schema of `X` that also accepts null (`"type": ["integer", "null"]`; `None` added to an enum) |
| `Any` | `{}` |
| a default value | not required; the default is in the schema |

Requiredness follows the Python signature: a parameter without a default is required, nullable
or not (`x: int | None` must be passed, possibly as `null`); give it a default
(`x: int | None = None`) to make it omittable. Null is accepted only where the hint says
`| None`, array items included (`list[int | None]`).

A parameter without an annotation, `*args` / `**kwargs`, positional-only parameters and other
unions are refused at apply time, with the module, function and parameter in the message.

**Context.** A parameter named `ctx`, or annotated `flock.tools.Context`, receives the call's
context and is never shown to the model:

| Field | What |
|---|---|
| `agent`, `type`, `swarm` | who is calling (the swarm name is the run id) |
| `workdir` | the agent's working directory; also the call's current directory |
| `data_dir` | a directory for tool state shared by the agents on this worker, kept across tools versions, not synced to the bucket |
| `tools_dir`, `tools_version` | the extracted folder of this version and its content hash |
| `env` | the agent's environment (what its `bash` sees); also the call's process environment |

**Results.** A `str` return value goes to the model as is; anything else is serialised as JSON
(`default=str`, so a `Path` becomes a string; `None` becomes `null`). `async def` tools work too.

**Errors.** An exception becomes the result `{"error": "ValueError: <message>"}`;
`raise flock.tools.ToolError("message")` gives `{"error": "message"}`. Arguments that do not fit
the schema (a missing required argument, an unknown one, a string or `null` for an `integer`) are refused
before the call with the same shape. A tool error is a result the model reads and the turn goes
on; it is marked `isError` in the event stream.

### Example

[`examples/tools-demo/tools/workspace.py`](../examples/tools-demo/tools/workspace.py), the
shape of the Eliza experiment's tools:

```python
from flock.tools import Context, ToolError


def ws_write(filename: str, content: str, reason: str, ctx: Context) -> dict:
    """Write a file to your workspace (ws/), replacing it if it exists.

    Python files are checked for syntax errors before they are written.

    Args:
        filename: a plain file name such as bot.py (no directories)
        content: the whole file
        reason: one line: why this change
    """
    if not _NAME.match(filename):
        raise ToolError(f"bad filename {filename!r}: use a plain name like bot.py")
    ...
    (_ws(ctx) / filename).write_text(content)
    return {"written": f"ws/{filename}", "chars": len(content)}


def run_bot(path: str, messages: list[str], ctx: Context) -> dict:
    """Run a bot script on a conversation and return its replies.

    Args:
        path: the script, relative to your workdir, e.g. ws/bot.py
        messages: the user messages, in order (at most 20)
    """
    ...
    return {"conversation": [{"user": m, "bot": reply} for m, reply in ...]}
```

What an `llm` agent's model is offered for `run_bot` (next to `bash` and, with `done_tool`,
`done`):

```json
{"type": "function", "function": {
  "name": "run_bot",
  "description": "Run a bot script on a conversation and return its replies.",
  "parameters": {"type": "object", "additionalProperties": false,
    "properties": {
      "path": {"type": "string", "description": "the script, relative to your workdir, e.g. ws/bot.py"},
      "messages": {"type": "array", "items": {"type": "string"},
                   "description": "the user messages, in order (at most 20)"}},
    "required": ["path", "messages"]}}}
```

## Selection

`types.<type>.tools` lists the tools a type gets, or `all`. It is strict: `flock apply` fails
with `types.dev.tools: unknown tool(s) promote_best; the tools folder has: ws_write, run_bot`
and applies nothing. A type without `tools:` gets no folder tools.

The folder is `tools/` next to the spec file; `tools_dir:` (top level, relative to the spec
file) points elsewhere. A spec where no type selects tools and `tools_dir` is unset ships
nothing, so existing specs (even next to an unrelated `tools/` directory) are unchanged.

## Versions

1. `flock apply` imports the modules in a fresh `python -I -B` process (side effects and
   crashes stay there), builds the manifest (names, descriptions, schemas), checks the
   selection, and packs the folder as a deterministic tar.gz (`__pycache__`, `*.pyc`, dotfiles,
   `.git`, `.venv` left out; symlinks refused) with the manifest inside. The version is the
   first 16 hex digits of the archive's sha256: the same folder always gives the same version.
2. It uploads the bundle (`POST /swarms/{swarm}/tools`; the controller stores it in its SQLite
   database, so a job-mode controller's state snapshot carries it) and applies the spec with
   `tools_version` set to that hash. The controller refuses a spec whose version was not
   uploaded or whose selection names a tool the bundle lacks.
3. Every assign command and every heartbeat response carries the swarm's current version and
   the worker type's selection. A worker fetches a version it does not have
   (`GET /swarms/{swarm}/tools/{version}`), checks the hash, and extracts it to
   `<WORK_ROOT>/_tools/<version>/`, outside every agent workdir. On assign this happens before
   the agent's first turn; after a re-apply it happens in the background within one heartbeat
   (30 s), and the worker switches only once the new version is ready.
4. A turn uses the version current when it starts, to its end: the `llm` harness takes the
   tool list once per turn, and `flock-tool` reads the agent's pin file, which the worker
   rewrites at each turn start. The next turn uses the new version. Old versions stay on the
   worker's disk for its lifetime.
5. Every `turn_started`, `turn_ended` and `turn_interrupted` event carries `tools_version`; a
   tools-folder tool result (`toolResult` message) carries `toolsVersion` and `durationMs`. When
   a wanted version could not be fetched or installed, `turn_started` also carries
   `tools_error` and the agent keeps the previous version; the worker retries at each heartbeat.

`GET /swarms/{swarm}/tools` lists the stored versions (the newest 10 plus the applied one are
kept per swarm) and which one is current.

**requirements.txt.** When the bundle has a `requirements.txt` with any requirement in it, the
worker creates a venv for that version (`<WORK_ROOT>/_tools/<version>.venv`, with `uv` when it
is on `PATH`, else `python -m venv` and pip), installs the requirements, and runs that version's
calls with its python. The venv also sees the worker's own packages (flock included), after its
own. The install runs in the extracted bundle directory, so relative entries (`./wheels/dep.whl`,
`-e ./pkg`) refer to files shipped in the folder. It must finish within 10 minutes; a failure is reported as `tools_error`. `flock
apply` imports the modules with your local python, so the requirements must be installed
locally too, or imported inside the functions.

## Harnesses

**`llm`.** The type's tools are offered as Chat Completions function tools next to `bash` (and
`done`), the system preamble says they exist, and a call goes to the runner below. Forced tool
choice keeps working (`required` lets the model pick any tool; `bash` still forces `bash`). A
call written as text (`{"name": "run_bot", "arguments": {...}}`, or `[run_bot(path="...")]`) is
accepted for folder tools as for `bash`. Output is trimmed to `tool_output_chars` like bash
output.

**`claude`, `codex`, `pi`.** The worker appends a generated section to the agent's
`CLAUDE.md` / `AGENTS.md` (between `<!-- flock:tools -->` markers, rewritten when the version
changes, at the next harness start) listing each tool with its description and one example:

```
- `run_bot`: Run a bot script on a conversation and return its replies.
  Example: `flock-tool run_bot --json '{"path": "...", "messages": ["..."]}'`
```

The agent calls them from its shell with `flock-tool NAME --json '<args>'` (`--json -` reads
stdin; `flock-tool --list` prints the schemas). It prints the result and exits 0, or prints
`{"error": ...}` and exits 1. `flock-tool` is installed with flock (on `PATH` in the worker image)
and finds the agent's tools through `FLOCK_TOOLS_PIN`, which the worker sets in every harness
env from the process's start. The worker rewrites that pin file at each turn boundary (when a
prompt starts a turn): a new version, tools enabled mid-burst and tools disabled (an empty
toolset: calls fail with `unknown tool`) all reach a running harness at its next turn, without
a restart. A prompt that arrives before the harness has reported the turn it already started
joins that turn and keeps its pin. The call's environment is the harness env minus the
provider key under every name the harness holds it (`api_key_env`, and for Claude
`ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN`; the pin lists them) and `FLOCK_TOKEN`, the same
rule as for `llm` tools. A harness-native model's key (e.g. `secrets: [ANTHROPIC_API_KEY]`
with no provider) is an ordinary declared secret and is not filtered. The generated section
is added after the spec's system prompt, also in a workdir restored with an older file.
Not yet: MCP for the CLI harnesses (the model sees the tools through the instructions file, not
as native tools).

## Limits

| What | Limit |
|---|---|
| one call | 60 s, then its whole process group is killed; the result is an error |
| result | 1 MB (an error above that); the `llm` harness then trims to `tool_output_chars` |
| bundle | 10 MB compressed, 50 MB unpacked, 2,000 files |
| load at apply | 60 s for importing the modules |

Each call is a fresh `python -I` process started in the agent's workdir with the agent's
environment (what its `bash` gets: never `FLOCK_TOKEN`, never the provider key). `-I` keeps the
workdir off the import path, so a file the agent wrote cannot shadow a module. A crash, a hang
or a huge result costs that call, never the worker. Module-level state does not survive between
calls: keep it in `ctx.data_dir` or the workdir. Startup is about 60 ms per call. Tools run as
the worker's user with the same access as the agent's `bash`: this is isolation from failure,
not a sandbox.

## Built-in tools

A type's tools can also be built-ins served by the worker instead of the folder: today the comms
tools (`comms:post`, `comms:read`, `comms:search`, model names `post`, `read`, `search`;
[docs/comms.md](comms.md)). They go through the same path: offered next to the folder tools in
the `llm` harness and listed in the generated Tools section for `flock-tool`, pinned at the same
turn boundaries. A built-in call runs in the worker (for `flock-tool`, through its local gateway
on 127.0.0.1 with the agent's key from the pin), not in a subprocess, so it can use credentials
the agent never sees.

## Follow-ups

- Hot reload under `flock run` (local mode, a separate PR): watch the folder and re-apply on save.
- Per-tool settings (timeout, rate) via a decorator; a long-lived tool host per type for tools
  that want warm state.
- MCP exposure for the CLI harnesses; a "tools changed" note in the next prompt.
- `flock status` showing the tools version per worker (today: `GET /swarms/{swarm}/tools` and the
  turn events).
