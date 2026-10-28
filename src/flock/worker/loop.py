"""Loop prompts on the worker (docs/loop.md): rendering the built-in placeholders.

`{round}`, `{elapsed}` (minutes since the loop started), `{wall}` (minutes of `while.until`,
or "none"), `{spend}` (the agent's cost in USD, 3 decimals), `{cap}` (its `agent_usd`, 2
decimals, or "none"), `{why}` (what triggered the prompt) and `{last_summary}` (the text the
agent last passed to `done`, else its last reply; after a turn that stopped without `done`, a
line saying so). Any other `{name}`, and format specs such
as `{cap:.2f}`, are left exactly as written.
"""

from __future__ import annotations

import re
from datetime import datetime

from flock.models import LoopMeta

_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")
NO_SUMMARY = "(none yet)"


def render(template: str, meta: LoopMeta, *, spend: float, last_summary: str | None,
           now: datetime) -> str:
    elapsed = (now - meta.started_at).total_seconds() / 60 if meta.started_at else 0.0
    values = {
        "round": "" if meta.round is None else str(meta.round),
        "elapsed": f"{max(0.0, elapsed):.0f}",
        "wall": "none" if meta.wall_s is None else f"{meta.wall_s / 60:.0f}",
        "spend": f"{spend:.3f}",
        "cap": "none" if meta.cap_usd is None else f"{meta.cap_usd:.2f}",
        "why": meta.why,
        "last_summary": last_summary or NO_SUMMARY,
    }
    return _PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), template)
