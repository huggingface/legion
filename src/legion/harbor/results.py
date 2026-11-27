"""Atomic manifests and redacted errors."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def safe_error(exc: BaseException, secrets=()):
    env_secrets = {
        value
        for key, value in os.environ.items()
        if len(value) >= 8 and any(name in key.upper() for name in ("TOKEN", "SECRET", "API_KEY"))
    }
    values = env_secrets | {s for s in secrets if s}
    redactions = values | {repr(s)[1:-1] for s in values} | {
        json.dumps(s, ensure_ascii=False)[1:-1] for s in values
    }
    message = str(exc)
    for secret in sorted(redactions, key=len, reverse=True):
        message = message.replace(secret, "[redacted]")
    return f"{type(exc).__name__}: {message}"[:2000]
