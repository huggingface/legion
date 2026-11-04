from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from flock.config import AgentSettings, Config


def test_role_groups_expand_in_configuration_order_and_roundtrip():
    agents = AgentSettings.model_validate(
        {"roles": {"explorer": {"count": 2}, "reviewer": {}}}
    )
    expected = {
        "agent-001": "explorer",
        "agent-002": "explorer",
        "agent-003": "reviewer",
    }
    assert agents.roster() == expected
    assert (
        AgentSettings.model_validate_json(agents.model_dump_json()).roster() == expected
    )
    assert AgentSettings().roster() == {"agent-001": None, "agent-002": None}
    assert AgentSettings(count=1).roster() == {"agent-001": None}


@pytest.mark.parametrize(
    "agents",
    [
        {"count": 2, "roles": {"explorer": {"count": 1}}},
        {"roles": {}},
        {"roles": {"explorer": {"count": 0}}},
        {"roles": {"explorer": {"count": True}}},
        {"roles": {"../explorer": {}}},
        {"roles": {"explorer": {"model": "another-model"}}},
    ],
)
def test_invalid_or_ambiguous_role_settings(agents):
    with pytest.raises(ValidationError):
        AgentSettings.model_validate(agents)


def test_role_files_resolve_relative_to_yaml(tmp_path, monkeypatch):
    data = Config.load(Path("examples/pi-hf-qwen.yaml")).model_dump(mode="json")
    data["team_file"] = "../prompts/team.md"
    data["agents"] = {
        "roles": {
            "explorer": {"count": 2, "role_file": "../prompts/explorer.md"},
            "reviewer": {"role_file": str(tmp_path / "absolute.md")},
            "helper": {},
        }
    }
    configs = tmp_path / "configs"
    configs.mkdir()
    path = configs / "roles.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    monkeypatch.chdir(tmp_path)
    spec = Config.load(path)
    assert spec.team_file == tmp_path / "prompts/team.md"
    assert spec.agents.roles["explorer"].role_file == tmp_path / "prompts/explorer.md"
    assert spec.agents.roles["reviewer"].role_file == tmp_path / "absolute.md"
    assert spec.agents.roles["helper"].role_file is None
    assert len(spec.agents.roster()) == 4
