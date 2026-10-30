from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from experiments.benchmarks.__main__ import analyze, load_secrets, scheduler, submit
from experiments.benchmarks.campaign import compile_spec, prepare, verify_manifest
from experiments.benchmarks.common import HERE, digest, read_json, write_json


def profile():
    return yaml.safe_load((HERE / 'profile-pi.yaml').read_text())


def test_pi_spec_preserves_inference_and_comms(tmp_path):
    spec = compile_spec(profile(), {'agents': 2, 'board_name': 'test'}, tmp_path / 'prompt.md')
    worker = spec['types']['worker']
    assert worker['harness'] == 'pi' and worker['pi_tools_only']
    assert worker['replicas'] == 2 and worker['reasoning'] == 'low'
    assert worker['env'] == {'FLOCK_COMMS_RATE': '0', 'FLOCK_COMMS_AGENT_RATE': '0'}
    assert spec['comms']['poll_cycle_s'] == 5
    assert spec['models']['qwen-27b']['extra']['maxTokens'] == 16384
    assert spec['world']['token_env'] == 'FLOCK_WORLD_TOKEN'


def test_prepare_stable_manifest_and_tamper_detection(tmp_path):
    task = tmp_path / 'task.json'
    task.write_text('{}')
    cfg = yaml.safe_load((HERE / 'pilot.yaml').read_text())
    cfg['execution_profile'] = str(HERE / 'profile-pi.yaml')
    cfg['tasks'] = [{'benchmark': 'agent_escape'}]
    source = tmp_path / 'campaign.yaml'
    source.write_text(yaml.safe_dump(cfg))
    selected = {'benchmark': 'agent_escape', 'task_id': 'task', 'task_path': str(task),
                'task_hash': digest(task), 'dataset_revision': 'pinned', 'native_wall_seconds': 900}
    with patch('experiments.benchmarks.campaign.resolve_task', return_value=selected):
        campaign = prepare(source, tmp_path / 'out')
    attempts = read_json(campaign)['attempts']
    assert len(attempts) == 2
    m = verify_manifest(Path(attempts[0]))
    assert m['wall_seconds'] == 900
    assert '-shared-n1-r1-' in m['board_name']
    assert m['mode'] == 'shared_task'
    task.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='contents changed'):
        verify_manifest(Path(attempts[0]))


def test_slurm_dryrun_serial_and_owned_cancel(tmp_path):
    path = tmp_path / 'campaign.json'
    write_json(path, {'attempts': ['a', 'b']})
    m = {'slurm': {'partition': 'cpu', 'cpus': 8, 'memory': '16G', 'time': '01:00:00'}}
    with patch('experiments.benchmarks.__main__.verify_manifest', return_value=m), \
         patch('subprocess.check_output') as subprocess:
        result = submit(path, tmp_path / 'secrets.json', dry_run=True)
        assert '0-1%1' in result['command']
        subprocess.assert_not_called()
    write_json(tmp_path / 'submission.json', {'job_id': '12345'})
    with patch('subprocess.check_output', return_value='') as subprocess:
        scheduler(path, cancel=True)
        subprocess.assert_called_once_with(['scancel', '12345'], text=True)


def test_bad_secrets_permissions(tmp_path):
    p = tmp_path / 'secrets.json'
    p.write_text('{"HF_TOKEN":"fake"}')
    p.chmod(0o644)
    with pytest.raises(ValueError, match='owner-only'):
        load_secrets(p)


def test_analysis_missing_accounting_keeps_outcomes(tmp_path):
    attempts = []
    for i, tokens in enumerate([None, 100]):
        out = tmp_path / str(i)
        write_json(out / 'result.json', {'attempt_id': str(i), 'benchmark': 'escape',
                   'task_id': 'a', 'agents': 2, 'valid': True, 'success': True,
                   'normalized_tokens': tokens})
        attempts.append(str(out / 'manifest.json'))
    path = tmp_path / 'campaign.json'
    write_json(path, {'attempts': attempts})
    summary = analyze(path)['escape:a:n2']
    assert summary['successes'] == 2
    assert summary['token_samples'] == 1
    assert summary['mean_normalized_tokens'] == 100
