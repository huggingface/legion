from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from harbor.trial.hooks import TrialEvent

from shepherd.config import InferenceSettings
from shepherd.harnesses.pi import HF_CHAT_EXTENSION, PI_CONFIG_DIR, attach_pi_setup


def make_trial():
    hooks = {}
    return SimpleNamespace(
        agent=SimpleNamespace(exec_as_agent=AsyncMock()),
        agent_environment=object(),
        add_hook=lambda event, callback: hooks.setdefault(event, callback),
        hooks=hooks,
    )


async def test_hf_reasoning_installs_extension_once():
    trial = make_trial()
    attach_pi_setup(trial, InferenceSettings(reasoning_effort="low"))
    prepare = trial.hooks[TrialEvent.AGENT_START]
    await prepare(None)
    await prepare(None)
    trial.agent.exec_as_agent.assert_awaited_once()
    args, kwargs = trial.agent.exec_as_agent.call_args
    assert args[0] is trial.agent_environment
    assert f"{PI_CONFIG_DIR}/extensions/shepherd-hf.js" in args[1]
    assert HF_CHAT_EXTENSION in args[1]
    assert kwargs["timeout_sec"] == 30


async def test_setup_failure_propagates():
    trial = make_trial()
    attach_pi_setup(trial, InferenceSettings(reasoning_effort="low"))
    trial.agent.exec_as_agent.side_effect = RuntimeError("setup failed")
    prepare = trial.hooks[TrialEvent.AGENT_START]
    with pytest.raises(RuntimeError, match="setup failed"):
        await prepare(None)
    trial.agent.exec_as_agent.assert_awaited_once()


@pytest.mark.parametrize(
    "overrides",
    [
        {"reasoning_effort": None},
        {"reasoning_effort": "off"},
        {"base_url": "https://api.example.com/v1"},
        {"base_url": "https://router.huggingface.co.example.com/v1"},
    ],
)
def test_other_connections_need_no_extension(overrides):
    trial = make_trial()
    inference = InferenceSettings(**{"reasoning_effort": "low", **overrides})
    attach_pi_setup(trial, inference)
    assert not trial.hooks


@pytest.mark.parametrize("recovered", [False, True])
def test_terminal_pi_error_ignores_noise_and_recovered_errors(tmp_path, recovered):
    import json

    from shepherd.harnesses.pi import terminal_pi_error

    path = tmp_path / "pi.txt"
    events = [
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "stopReason": "error",
                "errorMessage": "Connection error.",
            },
        }
    ]
    if recovered:
        events.append(
            {
                "type": "message_end",
                "message": {"role": "assistant", "stopReason": "stop", "content": []},
            }
        )
    path.write_text("console noise\n[]\n" + "\n".join(map(json.dumps, events)))
    assert terminal_pi_error(path) == (None if recovered else "Connection error.")


@pytest.mark.parametrize(
    "completed,expected", [(1, "This operation was aborted"), (2, None)]
)
def test_only_expected_turn_budget_abort_is_normal(tmp_path, completed, expected):
    import json

    from shepherd.harnesses.pi import terminal_pi_error

    path = tmp_path / "pi.txt"
    events = [{"type": "turn_end"}] * completed + [
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [],
                "stopReason": "error",
                "errorMessage": "This operation was aborted",
            },
        }
    ]
    path.write_text("\n".join(map(json.dumps, events)))
    assert terminal_pi_error(path, max_turns=2) == expected
    assert terminal_pi_error(path) == "This operation was aborted"
