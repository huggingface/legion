# Adding tools

Tools are Python functions packaged for the agents' task environments. Start with
a module in `src/legion/harbor/builtin/tools/`, or use an experiment-specific
`tools_dir`. Adding a tool does not require changing the controller or CLI.

## Define and select a tool

For example, save `calculator.py` in the built-in tools directory:

```python
def add(x: int, y: int = 0) -> dict:
    """Add two integers."""
    return {"sum": x + y}


TOOLS = {"add": add}
```

Select the exported function by module and name:

```yaml
tools: [message_board.*, calculator.add]
```

`calculator.*` selects all exports from that module. When `tools` is omitted, a
configured message board supplies `message_board.*`. An explicit list replaces
that default; `tools: []` selects no tools.

Functions need docstrings and annotated named parameters. Supported parameter
types include `str`, `int`, `float`, `bool`, `None`, unions, `Literal`, typed lists,
and dictionaries with string keys. Defaults become optional arguments. Return
JSON-serializable values; synchronous and asynchronous functions are supported.

Legion includes tool descriptions in the agent instructions. Inside the task
environment, the agent invokes them with:

```sh
legion-tool calculator.add --json '{"x": 2, "y": 3}'
```

This command uses the tool bundle installed for that agent; it is not a separate
operator command to run from the repository checkout.

## Experiment-specific tools

Set `tools_dir: tools` to load modules from a `tools/` folder next to your spec.
It replaces the built-in directory rather than adding to it. Copy any built-in
modules you still select, such as `message_board.py`, into that folder too.
The launcher packages the folders for workers; the selected modules and their
dependencies are bundled for each agent.

For example, with `spec.yaml` and `tools/calculator.py`:

```yaml
tools_dir: tools
tools: [calculator.add]
```

Use [live updates](live.md) to apply changed experiment tools to a running swarm.

## Dependencies and resources

Declare installed Python distributions with a module-level
`DEPENDENCIES = ["httpx"]`. The bundler includes their dependencies and records
versions. Bundles require pure Python dependencies; native extensions are rejected.
Code shipped into task containers must support Python 3.10+. Install any required
system programs in the task image instead of expecting the Python bundle to
supply them.

Tools that need a managed service declare `REQUIRES = ["resource_name"]`. Configure
that resource under `resources` in the spec. Providers handle service startup,
per-agent bindings, exported results, and cleanup. Bindings supply environment
variables, allowed hosts, and, where supported, mounts.

Use the existing [message board](comms.md), [world](world.md), and
[spawn](spawn.md) resources where they fit. For a custom provider, follow the
`Resource` interface in `src/legion/harbor/resources.py` and the providers under
`src/legion/harbor/builtin/resources/`; `resources_dir` selects an alternative
provider directory. Do not put credentials in tool source or example YAML.
