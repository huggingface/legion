"""Assemble experiment instructions, runtime context and selected tool documentation."""

import json
import shlex


def _example_value(schema):
    if "enum" in schema:
        return schema["enum"][0]
    if "anyOf" in schema:
        options = schema["anyOf"]
        return _example_value(
            next((s for s in options if s.get("type") != "null"), options[0])
        )
    if schema["type"] == "array":
        return [_example_value(schema["items"])]
    return {
        "string": "...",
        "integer": 1,
        "number": 1.0,
        "boolean": True,
        "null": None,
        "object": {},
    }[schema["type"]]


def build_agent_instructions(
    agent_name, roster, manifest, instructions="", role_instructions="", run_id=None
):
    lines = [
        f"You are {agent_name}, working on your own isolated copy of this task.",
    ]
    if run_id is not None:
        # A relaunch on the same Space board shares it with earlier runs.
        lines.append(
            f"This run's ID is {run_id}. Shared resources such as the message "
            "board may hold messages from earlier runs."
        )
    if roster[agent_name] is not None:
        lines.append(f"Your role: {roster[agent_name]}.")
    teammates = [
        f"{name} ({role})" if role is not None else name
        for name, role in roster.items()
        if name != agent_name
    ]
    if teammates:
        lines.append(
            f"Other participants: {', '.join(teammates)}. Files are not shared."
        )
    if instructions.strip():
        lines.extend(["", instructions.strip()])
    if role_instructions.strip():
        lines.extend(["", role_instructions.strip()])
    if not manifest["tools"]:
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            "",
            "## Tools",
            "Use the selected tools below through your shell. They return JSON.",
            'Errors print {"error": "..."} and exit with code 1.',
            "Run `flock-tool NAME --json '{...}'` or pass JSON through stdin with `--json -`.",
            "Examples show required arguments. Run `flock-tool --list` for full argument schemas, including optional parameters and defaults.",
        ]
    )
    for name, definition in manifest["tools"].items():
        parameters = definition["parameters"]
        arguments = {
            key: _example_value(parameters["properties"][key])
            for key in parameters["required"]
        }
        arguments_json = shlex.quote(json.dumps(arguments, ensure_ascii=False))
        lines.append(f"\n- `{name}`: {definition['description']}")
        lines.append(f"  Example: `flock-tool {name} --json {arguments_json}`")
    return "\n".join(lines) + "\n"
