# from shepherd f3970bc:src/flock/tools/runner.py
"""Load and invoke selected Python tools. The sandbox runtime uses Python 3.10+."""

import argparse
import asyncio
import importlib.util
import inspect
import json
import math
import os
import re
import sys
import types
import typing
from pathlib import Path

from ..results import safe_error

TOOL_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*\.[A-Za-z][A-Za-z0-9_]*")
TOOL_SELECTION = re.compile(r"[A-Za-z][A-Za-z0-9_]*\.(?:[A-Za-z][A-Za-z0-9_]*|\*)")


def load_module(path):
    spec = importlib.util.spec_from_file_location(f"_flock_tool_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def type_schema(annotation):
    primitive = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        type(None): "null",
    }
    if annotation in primitive:
        return {"type": primitive[annotation]}
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin in (typing.Union, types.UnionType):
        return {"anyOf": [type_schema(a) for a in args]}
    if origin is typing.Literal:
        return {"enum": list(args)}
    if origin is list and len(args) == 1:
        return {"type": "array", "items": type_schema(args[0])}
    if origin is dict and len(args) == 2 and args[0] is str:
        return {"type": "object", "additionalProperties": type_schema(args[1])}
    raise ValueError(f"Unsupported tool annotation: {annotation}")


def describe_tool(function):
    description = inspect.getdoc(function)
    if not inspect.isfunction(function) or not description:
        raise ValueError("Tools must be functions with docstrings")
    hints = typing.get_type_hints(function)
    properties, required = {}, []
    for name, parameter in inspect.signature(function).parameters.items():
        if parameter.kind not in (
            parameter.POSITIONAL_OR_KEYWORD,
            parameter.KEYWORD_ONLY,
        ):
            raise ValueError("Tools must accept named arguments; no *args or **kwargs")
        if name not in hints:
            raise ValueError(f"Missing type annotation for {name}")
        schema = type_schema(hints[name])
        if parameter.default is parameter.empty:
            required.append(name)
        else:
            validate_value(parameter.default, schema, name)
            schema["default"] = parameter.default
        properties[name] = schema
    return {
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


def validate_value(value, schema, name="arguments"):
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            try:
                validate_value(value, option, name)
                return
            except ValueError:
                pass
        raise ValueError(f"{name}: value does not match an allowed type")
    if "enum" in schema:
        if not any(type(value) is type(v) and value == v for v in schema["enum"]):
            raise ValueError(f"{name}: invalid choice")
        return
    kind = schema["type"]
    expected = {
        "string": (str,),
        "integer": (int,),
        "number": (int, float),
        "boolean": (bool,),
        "null": (type(None),),
        "array": (list,),
        "object": (dict,),
    }
    if type(value) not in expected[kind]:
        raise ValueError(f"{name}: expected {kind}")
    if kind == "number" and not math.isfinite(value):
        raise ValueError(f"{name}: expected finite number")
    if kind == "array":
        for item in value:
            validate_value(item, schema["items"], name)
    if kind == "object":
        if "properties" in schema:
            if set(value) - schema["properties"].keys():
                raise ValueError(f"{name}: unknown argument")
            if set(schema["required"]) - value.keys():
                raise ValueError(f"{name}: missing required argument")
            for key, item in value.items():
                validate_value(item, schema["properties"][key], key)
        else:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError(f"{name}: expected string keys")
                validate_value(item, schema["additionalProperties"], name)


def load_selected(directory, names):
    """Import trusted modules and expand module.* to their declared TOOLS entries."""
    functions, definitions, modules = {}, {}, {}
    for selection in names:
        if not TOOL_SELECTION.fullmatch(selection):
            raise ValueError(f"Invalid tool selection: {selection}")
        module_name, export = selection.split(".")
        if module_name not in modules:
            path = directory / f"{module_name}.py"
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Missing tool module: {path}")
            modules[module_name] = load_module(path)
        exports = getattr(modules[module_name], "TOOLS", {})
        selected = exports if export == "*" else [export]
        if not selected:
            raise ValueError(f"No tools exported by module: {module_name}")
        for export in selected:
            name = f"{module_name}.{export}"
            if not TOOL_NAME.fullmatch(name) or name in functions:
                raise ValueError(f"Invalid or duplicate tool name: {name}")
            function = exports.get(export)
            if function is None:
                raise ValueError(f"Unknown tool: {name}")
            functions[name] = function
            definitions[name] = describe_tool(function)
    return functions, definitions, modules


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", nargs="?")
    parser.add_argument("--json", default="{}", help="JSON object, or - for stdin")
    parser.add_argument(
        "--list", action="store_true", help="List selected tools and schemas"
    )
    args = parser.parse_args(argv)
    try:
        path = Path(os.environ["FLOCK_TOOLS_MANIFEST"])
        manifest = json.loads(path.read_text())
        names = list(manifest["tools"])
        if args.list:
            # Load all selected modules so setup catches missing runtime dependencies.
            load_selected(path.parent / "tools", names)
            print(json.dumps(manifest["tools"], ensure_ascii=False))
            return 0
        if args.name not in names:
            raise ValueError(f"Tool is not selected: {args.name}")
        arguments = json.loads(sys.stdin.read() if args.json == "-" else args.json)
        validate_value(arguments, manifest["tools"][args.name]["parameters"])
        functions, _, _ = load_selected(path.parent / "tools", [args.name])
        result = functions[args.name](**arguments)
        if inspect.isawaitable(result):
            result = asyncio.run(result)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"error": safe_error(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
