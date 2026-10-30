"""Ablation gating and metrics without model calls or scheduler submissions."""
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / 'experiments/agent-escape'


@pytest.fixture
def ablation(monkeypatch):
    monkeypatch.syspath_prepend(str(HERE))
    spec = importlib.util.spec_from_file_location('ablation', HERE / 'ablation.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def attempt(ablation, tmp_path):
    task = {'config': str(tmp_path / 'config.json'), 'agents': 2}
    report = {'accounting': {'complete': True, 'session_ids': ['a', 'b'],
                             'input_tokens': 999, 'output_tokens': 40},
              'flock_exit_code': 0, 'infrastructure_outcome': 'ok',
              'communication': 'off', 'persistence_status': 'not_applicable'}
    write(tmp_path / 'summary.json', [{'attempt': str(tmp_path), 'result': report}])
    write(tmp_path / 'metadata.json', {'spec': {'types': {'worker': {'comms': False}}}})
    journal = [{'event': 'barrier_opened', 'time': 100},
               {'event': 'start', 'session_id': 'a', 'tools': sorted(ablation.PUZZLE_TOOLS)},
               {'event': 'start', 'session_id': 'b', 'tools': sorted(ablation.PUZZLE_TOOLS)}]
    (tmp_path / 'model-requests.jsonl').write_text('\n'.join(map(json.dumps, journal)))
    (tmp_path / 'trajectory.jsonl').write_text('\n'.join(json.dumps({
        'agent': a, 'action': 'initialize', 'result': {}}) for a in ['a', 'b']))
    return task, report, tmp_path


def test_matrix_counts_pilot_once(ablation):
    rows = ablation.matrix()
    assert len(rows) == 15
    assert (rows[0]['agents'], rows[0]['trial']) == (8, 1)
    assert len({(r['agents'], r['trial']) for r in rows}) == 15
    assert all(r['communication'] == 'off' and r['difficulty'] == 20 for r in rows)
    assert rows == ablation.matrix()


def test_gate_accepts_clean_unsolved_pilot_and_rejects_board_tools(ablation, attempt):
    task, _, root = attempt
    assert ablation.validate_attempt(task) == []
    with (root / 'model-requests.jsonl').open('a') as stream:
        stream.write('\n' + json.dumps({'event': 'start', 'tools': ['post']}))
    assert 'unexpected offered tools' in ablation.validate_attempt(task)


def test_gate_rejects_incomplete_accounting_and_does_not_submit(ablation, attempt, monkeypatch):
    task, report, root = attempt
    report['accounting']['complete'] = False
    write(root / 'summary.json', [{'attempt': str(root), 'result': report}])
    monkeypatch.setattr(ablation, 'batch', lambda *a, **k: pytest.fail('must not submit'))
    assert ablation.advance(root, {'tasks': [task]}) == 1
    assert not json.loads((root / 'validation.json').read_text())['passed']


def test_metrics_output_only_and_barrier_time(ablation, attempt):
    task, report, root = attempt
    report['success'] = {'time': 160}
    write(root / 'result.json', report)
    row = {**task, 'attempt': str(root), 'measurement_valid': True, 'success': True,
           'infrastructure_outcome': 'ok', 'trial': 1}
    result = ablation.metrics(row, 'off')
    assert result['generated_tokens_per_agent'] == 20
    assert result['work_time_to_success_seconds'] == 60
    assert result['valid_trial']
    result = ablation.metrics({**row, 'measurement_valid': False}, 'off')
    assert result['generated_tokens_per_agent'] is None
    assert result['work_time_to_success_seconds'] is None
    assert not result['valid_trial']


def test_gate_allows_tool_free_compaction_but_requires_puzzle_requests(ablation, attempt):
    task, _, root = attempt
    path = root / 'model-requests.jsonl'
    with path.open('a') as stream:
        stream.write('\n' + json.dumps({'event': 'start', 'session_id': 'a', 'tools': []}))
    assert ablation.validate_attempt(task) == []
    path.write_text(json.dumps({'event': 'start', 'session_id': 'a', 'tools': []}))
    assert 'unexpected offered tools' in ablation.validate_attempt(task)


def test_gate_requires_puzzle_tools_for_every_agent(ablation, attempt):
    task, _, root = attempt
    path = root / 'model-requests.jsonl'
    rows = [r for r in ablation.records(path) if r.get('session_id') != 'b']
    path.write_text('\n'.join(map(json.dumps, rows)))
    assert 'unexpected offered tools' in ablation.validate_attempt(task)
