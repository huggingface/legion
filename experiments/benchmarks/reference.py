"""Explicit no-model integration gates. Never runs during ordinary pytest."""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import httpx

from .common import read_json, write_json
from .runner import Services, docker_preflight


def escape_reference(config):
    output = Path(config['output']) / 'reference-escape'
    output.mkdir(exist_ok=False)
    cfg = {**config, 'output': str(output), 'attempt_id': uuid.uuid4().hex}
    services = Services(cfg)
    try:
        services.start()
        direct_url = services.config['escape_url']
        direct = httpx.post(direct_url + '/env/reset', params={
            'case_file_path': cfg['task_path'], 'session_id': 'reference-' + uuid.uuid4().hex}).json()
        assert direct['ok']
        case = read_json(Path(cfg['task_path']))
        target = next(n for n in case['nodes'] if n['id'] == case['metadata']['win_node_id'])
        values = (target.get('golden_output_values') or target.get('init', {}).get('outputs')
                  or {o['name']: o['value'] for o in target['outputs']})
        answer = next(iter(values.values()))
        # This reference value stays in the evaluator; it is never given to a live agent.
        headers = {'X-World-Token': services.world_token, 'X-Flock-Swarm': cfg['board_name'],
                   'X-Flock-Type': 'worker', 'X-Flock-Turn': 'reference'}
        visible = next(n['id'] for n in case['nodes'] if n.get('initial_visible'))
        sequence = [('get_observation', {}), ('investigate', {'node_id': visible}), ('submit_answer', {'answer': 'deliberately-wrong'}),
                    ('submit_answer', {'answer': str(answer)})]
        for i, (tool, args) in enumerate(sequence):
            headers.update({'X-Flock-Agent': f"worker-{i % cfg['agents']:04d}",
                            'X-Request-Id': uuid.uuid4().hex})
            response = httpx.post(services.url + '/world/tools/' + tool,
                                  json={'args': args}, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            assert data['ok'], data
            observed = data['result']['result']
            if tool == 'get_observation':
                expected = {k: direct[k] for k in ('narrative', 'available_entities',
                            'available_information', 'win_node_id') if k in direct}
            else:
                expected = httpx.post(direct_url + f"/env/{direct['case_id']}/{tool}",
                                      json=args).json()
            assert observed == expected, (tool, observed, expected)
            if tool == 'submit_answer' and args['answer'] == 'deliberately-wrong':
                assert observed['is_win'] is False
        assert services.status()['reason'] == 'native_success'
        grade = services.control('evaluate')
        assert grade['success'] is True
        return {'status': 'passed', 'checks': ['real OpenEnv WebSocket', 'World API',
                'shared agent session', 'direct upstream parity', 'wrong answer rejected',
                'reference answer accepted'], 'output': str(output)}
    finally:
        errors = services.close()
        if errors:
            raise RuntimeError('; '.join(errors))


async def terminal_reference(config):
    from harbor.agents.oracle import OracleAgent
    from harbor.models.agent.context import AgentContext

    from .terminal import TerminalSession
    docker_preflight()
    reports = []
    for oracle in (False, True):
        output = Path(config['output']) / ('reference-oracle' if oracle else 'reference-invalid')
        output.mkdir(exist_ok=False)
        cfg = {**config, 'output': str(output), 'attempt_id': uuid.uuid4().hex}
        session = TerminalSession(cfg)
        try:
            await asyncio.wait_for(session.reset(), config['setup_timeout_s'])
            if oracle:
                agent = OracleAgent(logs_dir=session.paths.agent_dir,
                                    task_dir=Path(cfg['task_path']), trial_paths=session.paths)
                await asyncio.wait_for(agent.run(session.task.instruction, session.env,
                                                  AgentContext()), config['wall_seconds'])
            else:
                # Deliberately remove the required output in a fresh task sandbox.
                await session.env.exec('rm -f /app/recovered.json', timeout_sec=10)
            session.submitted = True
            result = await asyncio.wait_for(session.evaluate(), config['grading_timeout_s'])
            assert result['success'] is oracle, result
            reports.append(result)
        finally:
            await session.close()
    return {'status': 'passed', 'checks': ['native invalid submission', 'native oracle'],
            'results': reports}


def reference(config):
    try:
        result = (escape_reference(config) if config['benchmark'] == 'agent_escape'
                  else asyncio.run(terminal_reference(config)))
    except Exception as exc:  # noqa: BLE001 - retain backend failure as explicit result
        result = {'status': 'blocked' if 'docker' in str(exc).lower() else 'failed',
                  'error': str(exc)}
    write_json(Path(config['output']) / 'reference.json', result)
    return result
