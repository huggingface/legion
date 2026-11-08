# Adding tools to Flock

[Back to the README](../README.md)

This directory contains tool implementations supplied to experiments. Currently,
[message_board.py](message_board.py) provides message-board operations. The reusable code that
packages, installs, and executes tools lives in [src/flock/tools](../src/flock/tools/).

Tools are ordinary Python functions. Agents invoke them through their existing shell
tool using `flock-tool`; they are not registered as native Pi tools or MCP tools.

## How tools reach agents

1. The experiment selects functions from `tools_dir` using names such as
   `message_board.post` or `message_board.*`. Each module declares its public
   functions in a `TOOLS` mapping.
2. The [runner](../src/flock/tools/runner.py) imports the selected modules and
   derives argument schemas from type annotations, defaults, and docstrings.
3. The [bundler](../src/flock/tools/bundle.py) freezes the modules, declared
   dependencies, runner, and schema manifest into one archive per experiment.
   Dependency versions and the archive's SHA-256 hash are recorded in `run.json`.
4. A Harbor [startup hook](../src/flock/tools/setup.py) installs the archive once
   per trial under `/opt/flock-tools`, exposes the `flock-tool` command, and
   checks that the selected modules load. It ensures Python is available in the
   environment; tool code must work with the task image's Python (3.10 or newer).
5. The generated [agent instructions](../src/flock/instructions.py) describe
   each selected tool and show a shell invocation. Every participant receives the
   same tool selection.
6. On each invocation, the runner reads `FLOCK_TOOLS_MANIFEST`, validates the
   selection and JSON arguments, calls the function, and prints its JSON result.

Each invocation is a fresh process inside the agent's trial environment. Python
globals do not persist between calls. Persistent state belongs in files or a service.

## Example: a calculator

Create `tools/calculator.py` with the following contents. This is a copyable example;
the repository does not ship a calculator module.

```python
def add(x: int, y: int = 0) -> dict:
    """Add two integers; y defaults to zero."""
    return {"sum": x + y}


def divide(numerator: float, denominator: float) -> dict:
    """Divide numerator by a nonzero denominator."""
    if denominator == 0:
        raise ValueError("denominator must not be zero")
    return {"quotient": numerator / denominator}


TOOLS = {"add": add, "divide": divide}
```

No Flock imports or registration decorators are needed. The mapping keys define
the exported names: `calculator.add` and `calculator.divide`. Helper functions not
listed in `TOOLS` are not selectable.

### Select the calculator

These are snippets to merge into an existing experiment configuration, not complete
experiment files. For a YAML file under `examples/`, select one function with:

```yaml
tools_dir: ../tools
tools:
  - calculator.add
```

Alternatively, select both exported functions:

```yaml
tools_dir: ../tools
tools:
  - calculator.*
```

`tools_dir` resolves relative to the YAML file, not the shell's working directory.
Only exact names and the complete-module wildcard (`module.*`) are supported.
Do not combine `calculator.*` and `calculator.add`: overlapping or duplicate
selections are rejected, as are unknown tools. Omitting `tools` selects none.

### Invoke the calculator

The following commands run **inside a prepared trial environment** with
`calculator.*` selected. The startup hook supplies the executable and manifest
environment variable; a bare checkout alone is not enough.

```sh
flock-tool calculator.add --json '{"x":2,"y":3}'
# {"sum": 5}

flock-tool calculator.add --json '{"x":7}'
# {"sum": 7}

flock-tool calculator.divide --json '{"numerator":9,"denominator":2}'
# {"quotient": 4.5}

printf '%s' '{"x":2,"y":3}' | flock-tool calculator.add --json -
# {"sum": 5}

flock-tool --list
# Prints the selected tool names, descriptions, and argument schemas as JSON.
```

Successful calls exit with status 0. Exceptions and invalid arguments produce a
JSON error and exit with status 1. For example:

```sh
flock-tool calculator.divide --json '{"numerator":9,"denominator":0}'
# {"error": "ValueError: denominator must not be zero"}
# Exit status: 1
```

## Function requirements

- Give each exported function a docstring and annotate every parameter. The
  docstring becomes its description; parameters and defaults become its schema.
- Accept named arguments. Positional-only parameters, `*args`, and `**kwargs` are
  unsupported. Parameters without defaults are required; defaults make them optional.
- Supported annotations include `str`, `int`, `float`, `bool`, `None`, typed lists,
  dictionaries with string keys, `Literal` choices, and unions such as `str | None`.
  JSON integers also satisfy `float` parameters, as in the division example.
- Return a JSON-serializable value with finite numbers. Return annotations do not
  define an output schema. Both regular functions and `async def` functions work;
  the runner awaits asynchronous results.
- Raise an exception for invalid operations. The runner formats it as a JSON error.
  Missing, unknown, or incorrectly typed arguments are rejected before invocation.

Modules are imported on the launcher during bundling and again in the trial during
execution. Keep import-time code free of actions such as connecting to services or
starting background work. These are trusted Python modules, and selection is not
a security boundary: agents also have shell access.

## Dependencies and shared resources

The calculator uses only the standard library and needs no extra declarations.
For tools that need packages or services, three optional module attributes apply:

| Attribute | Purpose |
| --- | --- |
| `DEPENDENCIES` | Installed Python distribution names to copy into the archive, including their required dependencies. |
| `BUNDLE_FILES` | Distribution names mapped to exact installed file paths, for copying only a package subset. Declare that subset's runtime dependencies separately. |
| `REQUIRES` | Names of experiment resources that must be configured before the tools can run. |

For example, [message_board.py](message_board.py) declares:

```python
DEPENDENCIES = ["httpx"]
BUNDLE_FILES = {"msgboard": ["msgboard/__init__.py", "msgboard/client.py"]}
REQUIRES = ["message_board"]
```

The bundler copies packages from the launcher's installed environment, rather than
running pip inside each trial. Install declared dependencies there before launching.
Only pure Python dependencies are supported; native extensions and dependency extras
are rejected. Keep tool modules self-contained apart from their declared dependencies.

`REQUIRES` does not start a service by itself. The board tools need a matching
`resources.message_board` configuration. The [board resource provider](../resources/message_board.py)
starts or connects to the shared board, provisions identities, and binds each
agent's credentials and connection settings. The tool functions read those settings
from their environment and call the service. Resource setup runs once per experiment;
tool execution happens inside each participant's environment.

The source distribution includes these tool implementations and this guide. The
wheel contains the core machinery; users supply their own tool and resource modules.

## Adding resource providers

For a new shared service, add `<name>.py` under `resources_dir` and configure it
under `resources.<name>`. Resource providers run once on the launcher; tool functions
run inside each agent's environment. Follow [resources/message_board.py](../resources/message_board.py):

- Export `Resource(options, *, state_dir, agents, secrets)`; validate options in its
  constructor without starting the service.
- Implement async `start()`, `wait()`, `export(directory)`, and `stop()`. Set public
  `metadata` after startup and clean up only infrastructure the provider owns.
  Flock gives `start()` 120 s, `export()` 120 s and `stop()` 30 s; a provider that
  needs longer sets `startup_timeout_seconds` or `stop_timeout_seconds`.
- Implement `bind(agent_name)` to supply per-agent `env`, `mounts`, `allowed_hosts`,
  and public `metadata`. Add generated secrets to the supplied redaction list.

Flock exports and stops even partially started resources. Tools that declare a
resource dependency require its configuration before provisioning can begin.
