# from shepherd f3970bc:src/flock/harnesses/pi.py
"""Compatibility setup for native Pi against Hugging Face's chat API."""

import json
import shlex
from pathlib import Path
from urllib.parse import urlsplit

from harbor.trial.hooks import TrialEvent

# Harbor 0.25 writes models.json after AGENT_START and does not expose Pi's
# supportsDeveloperRole setting. Use Pi's native extension hook instead.
PI_CONFIG_DIR = "/tmp/harbor-pi-agent"
HF_CHAT_EXTENSION = """export default function (pi) {
  pi.on("before_provider_request", (event) => {
    const payload = event.payload;
    if (!Array.isArray(payload?.messages)) return;
    return {
      ...payload,
      messages: payload.messages.map((message) =>
        message.role === "developer" ? { ...message, role: "system" } : message
      ),
    };
  });
}
"""


def attach_pi_setup(trial, inference):
    if (
        inference.api_format != "openai_chat_completions"
        or urlsplit(inference.base_url).hostname != "router.huggingface.co"
        or inference.reasoning_effort in (None, "off")
    ):
        return

    prepared = False

    async def prepare_pi(event):
        nonlocal prepared
        if prepared:
            return
        directory = f"{PI_CONFIG_DIR}/extensions"
        await trial.agent.exec_as_agent(
            trial.agent_environment,
            f"mkdir -p {directory} && printf %s {shlex.quote(HF_CHAT_EXTENSION)} "
            f"> {directory}/flock-hf.js",
            timeout_sec=30,
        )
        prepared = True

    trial.add_hook(TrialEvent.AGENT_START, prepare_pi)


def terminal_pi_error(path: Path, max_turns: int | None = None) -> str | None:
    """Read native Pi events without treating recovered errors as failures."""
    if not path.is_file():
        return None
    last = None
    completed_turns = 0
    turns_before_last = 0
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") == "turn_end":
                completed_turns += 1
            if event.get("type") != "message_end":
                continue
            message = event.get("message")
            if isinstance(message, dict) and message.get("role") == "assistant":
                last = message
                turns_before_last = completed_turns
    # Harbor's turn-budget extension aborts the next request after the last turn.
    if (
        last
        and max_turns is not None
        and turns_before_last >= max_turns
        and last.get("errorMessage") == "This operation was aborted"
        and not last.get("content")
    ):
        return None
    if last and (last.get("errorMessage") or last.get("stopReason") == "error"):
        return str(last.get("errorMessage") or "Pi ended with a provider error")
    return None
