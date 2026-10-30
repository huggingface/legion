"""Run supervisor against fake processes, HTTP and environments only."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from experiments.benchmarks.common import HERE, read_json
from experiments.benchmarks.runner import run


@pytest.mark.parametrize('failure', [False, True])
def test_supervisor_grades_or_records_environment_failure(tmp_path, monkeypatch, failure):
    monkeypatch.setenv('LOCAL_KEY', 'fake')
    monkeypatch.setenv('HF_TOKEN', 'fake')
    m = {'output': str(tmp_path), 'attempt_id': 'a', 'benchmark': 'agent_escape',
         'task_id': 'test', 'agents': 2, 'mode': 'shared_task', 'board_name': 'board',
         'wall_seconds': 1, 'grading_timeout_s': 1}
    (tmp_path / 'spec.yaml').write_text((HERE / 'profile-pi.yaml').read_text())
    spec = yaml.safe_load((tmp_path / 'spec.yaml').read_text())
    spec['world'] = {'url': 'http://placeholder'}
    (tmp_path / 'spec.yaml').write_text(yaml.safe_dump(spec))
    operations = []

    class Services:
        url = 'http://environment'
        world_token = 'world'

        def __init__(self, config):
            pass

        def start(self):
            operations.append('start')

        def status(self):
            if failure:
                raise RuntimeError('Environment subprocess exited')
            return {'sealed': True, 'reason': 'native_success', 'error': None}

        def control(self, name, *args, **kwargs):
            operations.append(name)
            return {'success': True, 'metrics': {'is_win': True}}

        def close(self):
            operations.append('close')
            return []

    process = SimpleNamespace(poll=lambda: None, returncode=0)
    response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'active': 0})
    with patch('experiments.benchmarks.runner.verify_manifest', return_value=m), \
         patch('experiments.benchmarks.runner.validate'), \
         patch('experiments.benchmarks.runner.Services', Services), \
         patch('experiments.benchmarks.runner.launch', return_value=process) as launch, \
         patch('experiments.benchmarks.runner.wait_http'), \
         patch('experiments.benchmarks.runner.free_port', return_value=12345), \
         patch('experiments.benchmarks.runner.stop'), \
         patch('httpx.get', return_value=response), patch('httpx.post', return_value=response):
        result = run(tmp_path / 'manifest.json')
    assert launch.call_count == 2
    assert result['valid'] is not failure
    assert ('evaluate' in operations) is not failure
    assert operations[-1] == 'close'
    assert result['normalized_tokens'] is None  # no fabricated zero-token measurement
    assert read_json(tmp_path / 'result.json')['reason'] == (
        'infrastructure_failure' if failure else 'native_success')
    with patch('experiments.benchmarks.runner.verify_manifest', return_value=m), \
         pytest.raises(FileExistsError):
        run(tmp_path / 'manifest.json')
