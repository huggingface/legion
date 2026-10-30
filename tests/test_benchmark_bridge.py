"""Inference adapters: no real jobs, model calls, or external services."""
import asyncio
from types import SimpleNamespace

import httpx
import pytest

from experiments.benchmarks.bridge import Bridge
from experiments.benchmarks.escape import EscapeSession
from experiments.benchmarks.terminal import TerminalSession
from flock.world import Agent, ToolError
from flock.world.server import create_app


@pytest.fixture
def config(tmp_path):
    return {'agents': 2, 'benchmark': 'agent_escape', 'attempt_id': 'trial',
            'board_name': 'board', 'seed': 42, 'output': str(tmp_path),
            'task_path': '/fixture/task.json', 'escape_url': 'http://fake'}


class Client:
    def __init__(self):
        self.calls = []
        self.resets = 0
        self.value = 0

    async def reset(self, **kwargs):
        self.resets += 1

    async def step(self, action):
        self.calls.append(action)
        await asyncio.sleep(0)
        self.value += 1
        return {'result': self.value, 'done': action['tool'] == 'submit_answer'}


def caller(i=0, request='r'):
    return Agent(f'worker-{i:04d}', swarm='board', turn='t-1', request_id=request)


@pytest.mark.asyncio
async def test_shared_session_order_identity_and_dedup(config, tmp_path):
    client = Client()
    bridge = Bridge(config, client, tmp_path)
    await bridge.start()
    results = await asyncio.gather(
        bridge.call('investigate', {'node_id': 'a'}, caller()),
        bridge.call('investigate', {'node_id': 'b'}, caller(1)))
    assert [r['result'] for r in results] == [1, 2]
    assert client.resets == 1
    assert [a['agent'] for a in client.calls] == ['worker-0000', 'worker-0001']
    assert await bridge.call('investigate', {'node_id': 'a'}, caller()) == results[0]
    assert len(client.calls) == 2
    with pytest.raises(ToolError, match='different arguments'):
        await bridge.call('investigate', {'node_id': 'c'}, caller())
    with pytest.raises(ValueError, match='already started'):
        await bridge.start()


@pytest.mark.asyncio
async def test_roster_and_tools_and_seal(config, tmp_path):
    bridge = Bridge(config, Client(), tmp_path)
    await bridge.start()
    with pytest.raises(ToolError, match='Caller'):
        await bridge.call('get_observation', {}, caller(3))
    for tool in ['reset', 'state', 'evaluate']:
        with pytest.raises(ToolError, match='Unknown'):
            await bridge.call(tool, {}, caller())
    await bridge.call('submit_answer', {'answer': 'x'}, caller())
    with pytest.raises(ToolError, match='not accepting'):
        await bridge.call('investigate', {'node_id': 'a'}, caller(request='second'))


@pytest.mark.asyncio
async def test_disconnect_invalidates_without_replay(config, tmp_path):
    client = Client()
    async def broken(action):
        raise ConnectionError('lost after mutation')
    client.step = broken
    bridge = Bridge(config, client, tmp_path)
    await bridge.start()
    with pytest.raises(ToolError, match='invalidated'):
        await bridge.call('investigate', {'node_id': 'a'}, caller())
    assert bridge.status()['reason'] == 'infrastructure_failure'
    with pytest.raises(ToolError, match='not accepting'):
        await bridge.call('investigate', {'node_id': 'a'}, caller())
    assert client.resets == 1


@pytest.mark.asyncio
async def test_world_auth_and_schema(config, tmp_path):
    bridge = Bridge(config, Client(), tmp_path)
    await bridge.start()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(bridge, 'secret')),
                                 base_url='http://test') as client:
        assert (await client.get('/manifest')).status_code == 401
        headers = {'X-World-Token': 'secret', 'X-Flock-Agent': 'worker-0000',
                   'X-Flock-Swarm': 'board', 'X-Request-Id': 'r'}
        r = await client.post('/tools/investigate', headers=headers,
                             json={'args': {'node_id': 'a', 'agent': 'worker-0001'}})
        assert not r.json()['ok']
        assert bridge.sequence == 0


@pytest.mark.asyncio
async def test_separate_trials_do_not_share_state(config, tmp_path):
    a, b = Bridge(config, Client(), tmp_path), Bridge(config, Client(), tmp_path)
    await a.start()
    await b.start()
    await a.call('investigate', {'node_id': 'a'}, caller())
    assert b.client.value == 0


@pytest.mark.asyncio
async def test_escape_reset_once_filter_and_win(config):
    requests = []
    def handler(request):
        requests.append(request)
        if request.url.path == '/env/reset':
            return httpx.Response(200, json={'case_id': 'shared', 'narrative': 'room',
                                             'solution': 'secret', 'case_file_path': '/secret'})
        return httpx.Response(200, json={'is_win': 'right' in request.content.decode()})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url='http://fake') as c:
        env = EscapeSession(config, c)
        assert (await env.reset())['result'] == {'narrative': 'room'}
        assert not (await env.step('submit_answer', {'answer': 'wrong'}, 'a'))['done']
        assert (await env.step('submit_answer', {'answer': 'right'}, 'b'))['done']
        assert (await env.evaluate())['success']
        assert requests[-1].url.path == '/env/shared/submit_answer'
        with pytest.raises(RuntimeError):
            await env.reset()


class Sandbox:
    def __init__(self):
        self.gate = asyncio.Event()
        self.stopped = False
        self.commands = []

    async def exec(self, command, **kwargs):
        self.commands.append(command)
        if command.startswith('timeout'):
            await self.gate.wait()
        return SimpleNamespace(return_code=0, stdout='', stderr='')

    async def stop(self, delete):
        self.stopped = True


@pytest.mark.asyncio
async def test_terminal_mutations_busy_and_submission(config):
    sandbox = Sandbox()
    env = TerminalSession(config, sandbox, SimpleNamespace(instruction='task'))
    await env.reset()
    first = await env.step('exec_start', {'command': 'echo test'}, 'a')
    assert first['result']['job_id']
    assert 'busy' in (await env.step('fs_write', {'path': '/a', 'content': 'x'}, 'b'))['error']
    assert not (await env.step('task_submit', {}, 'b'))['done']
    sandbox.gate.set()
    await env.jobs[first['result']['job_id']]['task']
    assert (await env.step('task_submit', {}, 'b'))['done']
    assert (await env.step('fs_write', {'path': '/a', 'content': 'x'}, 'a'))['done']
    await env.close()
    assert sandbox.stopped


@pytest.mark.asyncio
async def test_terminal_cancel_stops_container_not_just_host_future(config):
    sandbox = Sandbox()
    env = TerminalSession(config, sandbox, SimpleNamespace(instruction='task'))
    await env.reset()
    first = await env.step('exec_start', {'command': 'sleep 200'}, 'a')
    result = await env.step('exec_cancel', first['result'], 'b')
    assert result['reason'] == 'cancelled'
    assert sandbox.stopped
    with pytest.raises(RuntimeError, match='stopped'):
        await env.evaluate()
