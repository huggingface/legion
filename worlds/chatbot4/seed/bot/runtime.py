"""Fixed module interface: respond(text, memory) returns a reply or None.

Modules are tried by descending PRIORITY (default 0), then by filename.
Each conversation has its own shared memory dictionary. Import and handler failures
leave the remaining modules available. Keep this loader fixed; edit modules instead.
"""
import importlib
import pkgutil

from . import modules


def load_modules():
    handlers = []
    for entry in pkgutil.iter_modules(modules.__path__):
        try:
            module = importlib.import_module(f"{modules.__name__}.{entry.name}")
            if callable(getattr(module, "respond", None)):
                handlers.append((getattr(module, "PRIORITY", 0), entry.name, module.respond))
        except Exception:  # noqa: BLE001, S112 - preserve the bot interface
            continue
    return [fn for _, _, fn in sorted(handlers, key=lambda item: (-item[0], item[1]))]


_HANDLERS = load_modules()


def reply(text, memory):
    for respond in _HANDLERS:
        try:
            answer = respond(text, memory)
            if isinstance(answer, str) and answer.strip():
                return answer
        except Exception:  # noqa: BLE001, S112 - preserve the bot interface
            continue
    return "Hello!"
