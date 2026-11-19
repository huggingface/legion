"""Provider request setup for native Pi."""

import json
import shlex
from pathlib import Path

from harbor.trial.hooks import TrialEvent

# Harbor 0.25 writes models.json after AGENT_START. Pi's request hook applies these settings
# in both the worker's long-lived Trials and the direct Harbor launcher.
PI_CONFIG_DIR = "/tmp/harbor-pi-agent"


def request_extension(inference):
    """The native Pi request hook, or None when no provider adjustment is needed."""
    fix_roles = inference.hf_router is not None and inference.reasoning_effort not in (None, "off")
    if not fix_roles and not inference.extra_body:
        return None
    return """export default function (pi) {
  pi.on("before_provider_request", (event) => {
    const payload = { ...event.payload, ...__EXTRA_BODY__ };
    if (__FIX_ROLES__ && Array.isArray(payload.messages)) {
      payload.messages = payload.messages.map((message) =>
        message.role === "developer" ? { ...message, role: "system" } : message
      );
    }
    return payload;
  });
}
""".replace("__FIX_ROLES__", str(fix_roles).lower()).replace(
        "__EXTRA_BODY__", json.dumps(inference.extra_body))


def attach_pi_setup(trial, inference):
    """`inference`: the agent's block (flock.models.Inference)."""
    extension = request_extension(inference)
    if extension is None:
        return

    prepared = False

    async def prepare_pi(event):
        nonlocal prepared
        if prepared:
            return
        directory = f"{PI_CONFIG_DIR}/extensions"
        await trial.agent.exec_as_agent(
            trial.agent_environment,
            f"mkdir -p {directory} && printf %s {shlex.quote(extension)} "
            f"> {directory}/flock-request.js",
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
