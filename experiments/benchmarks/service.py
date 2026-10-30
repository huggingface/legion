"""OpenEnv server and separate authenticated World facade, bound to loopback."""
from __future__ import annotations

import asyncio
import hmac
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import HTTPException, Request
from openenv.core.env_server import Environment, create_app
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from websockets.asyncio.client import connect

from flock.world.server import create_app as world_app

from .bridge import Bridge
from .common import read_json
from .escape import EscapeSession
from .terminal import TerminalSession


class TaskAction(Action):
    tool: str
    args: dict = Field(default_factory=dict)
    agent: str


class TaskObservation(Observation):
    payload: dict


class TaskEnvironment(Environment):
    def __init__(self, session):
        super().__init__()
        self.session = session
        self._state = State()

    def reset(self, **kwargs):
        raise RuntimeError('Use the asynchronous OpenEnv session')

    async def reset_async(self, **kwargs):
        data = await self.session.reset(**kwargs)
        self._state = State(episode_id=kwargs.get('episode_id'), step_count=0)
        return TaskObservation(payload=data, done=data.get('done', False))

    def step(self, action, **kwargs):
        raise RuntimeError('Use the asynchronous OpenEnv session')

    async def step_async(self, action, **kwargs):
        data = await self.session.step(action.tool, action.args, action.agent)
        self._state.step_count += 1
        return TaskObservation(payload=data, done=data.get('done', False))

    @property
    def state(self):
        return self._state


class SessionClient:
    """The documented OpenEnv WebSocket protocol; no automatic reconnect/replay."""
    def __init__(self, url, token, timeout=60):
        self.url, self.token, self.timeout = url, token, timeout
        self.ws = None
        self.used = False

    async def reset(self, **kwargs):
        if self.used:
            raise RuntimeError('Session already opened')
        self.used = True
        self.ws = await connect(self.url, additional_headers={'X-Benchmark-Control': self.token},
                                max_size=1024 * 1024, proxy=None)
        return await self._exchange('reset', kwargs)

    async def step(self, data):
        return await self._exchange('step', data)

    async def _exchange(self, kind, data):
        if self.ws is None:
            raise RuntimeError('OpenEnv session unavailable')
        await self.ws.send(json.dumps({'type': kind, 'data': data}))
        reply = json.loads(await asyncio.wait_for(self.ws.recv(), self.timeout))
        if reply.get('type') != 'observation':
            raise RuntimeError(f"OpenEnv {kind} failed: {reply.get('data')}")
        body = reply['data']
        return body.get('observation', body)['payload']

    async def close(self):
        if self.ws is not None:
            await self.ws.close()


class ControlBoundary:
    """World calls have their own token. Every other route requires the control token."""
    def __init__(self, app, token):
        self.app, self.token = app, token

    async def __call__(self, scope, receive, send):
        if scope['type'] in {'http', 'websocket'} and not scope['path'].startswith('/world/'):  # noqa: SIM102
            if not hmac.compare_digest(Headers(scope=scope).get('X-Benchmark-Control', ''),
                                       self.token):
                if scope['type'] == 'websocket':
                    await send({'type': 'websocket.close', 'code': 1008})
                else:
                    await JSONResponse({'error': 'unauthorized'}, status_code=401)(
                        scope, receive, send)
                return
        await self.app(scope, receive, send)


def build_app(config, port, world_token, control_token):
    session = EscapeSession(config) if config['benchmark'] == 'agent_escape' else TerminalSession(config)
    environment = TaskEnvironment(session)
    app = create_app(lambda: environment, TaskAction, TaskObservation, max_concurrent_envs=1)
    client = SessionClient(f'ws://127.0.0.1:{port}/ws', control_token,
                           timeout=config.get('setup_timeout_s', 600))
    bridge = Bridge(config, client, Path(config['output']))
    app.mount('/world', world_app(bridge, world_token))
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app):
        async with original_lifespan(app):
            try:
                yield
            finally:
                await client.close()
                await session.close()
    app.router.lifespan_context = lifespan

    @app.get('/control/status')
    async def status():
        return bridge.status()

    @app.post('/control/start')
    async def start():
        await bridge.start()
        client.timeout = 45
        return bridge.status()

    @app.post('/control/seal')
    async def seal(request: Request):
        data = await request.json()
        await bridge.seal(data.get('reason', 'submission'))
        return bridge.status()

    @app.post('/control/evaluate')
    async def evaluate():
        if not bridge.sealed or bridge.reason in {'infrastructure_failure', 'cancelled'}:
            raise HTTPException(409, 'Task is not eligible for grading')
        return await asyncio.wait_for(session.evaluate(), config.get('grading_timeout_s', 300))

    @app.post('/control/close')
    async def close():
        await bridge.seal('cancelled')
        await client.close()
        await session.close()
        return {'closed': True}

    app.add_middleware(ControlBoundary, token=control_token)
    return app


def main():
    import argparse

    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('--port', required=True, type=int)
    args = parser.parse_args()
    uvicorn.run(build_app(read_json(args.config), args.port, os.environ['FLOCK_WORLD_TOKEN'],
                         os.environ['BENCHMARK_CONTROL_TOKEN']), host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
