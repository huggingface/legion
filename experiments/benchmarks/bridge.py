"""Authenticated Flock World facade for one non-reconnecting OpenEnv session."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from jsonschema import validate

from flock.world import Agent, ToolError, World

from .common import write_json
from .tools import tools_for


class Bridge(World):
    name, version = 'openenv-task', '0.1.0'
    description = ('All team members share ONE task state. Coordinate through the board. '
                   'Use only task tools; a teammate finishing does not reset the task.')

    def __init__(self, manifest: dict, client, output: Path):
        self.config, self.client, self.output = manifest, client, output
        self.roster = {f'worker-{i:04d}' for i in range(manifest['agents'])}
        self.tools = {t['name']: t for t in tools_for(manifest['benchmark'])}
        self.lock = asyncio.Lock()
        self.cache = {}
        self.started = False
        self.sealed = False
        self.reason = None
        self.error = None
        self.sequence = 0

    def manifest(self):
        return {'world_api': 'v1', 'name': self.name, 'version': self.version,
                'description': self.description, 'tools': list(self.tools.values())}

    async def start(self):
        async with self.lock:
            if self.started:
                raise ValueError('This session has already started; reset is forbidden')
            self.started = True  # Never retry a failed reset into a new session.
            try:
                await self.client.reset(seed=self.config['seed'],
                                        episode_id=self.config['attempt_id'])
            except Exception as exc:
                self.fail(exc)
                raise

    def fail(self, exc):
        self.sealed, self.reason, self.error = True, 'infrastructure_failure', str(exc)
        write_json(self.output / 'session.json', self.status())

    def status(self):
        return {'started': self.started, 'sealed': self.sealed, 'reason': self.reason,
                'error': self.error, 'actions': self.sequence}

    async def seal(self, reason):
        async with self.lock:
            self.sealed = True
            self.reason = self.reason or reason
            write_json(self.output / 'session.json', self.status())

    async def call(self, name: str, args: dict, agent: Agent):
        if agent.name not in self.roster or agent.swarm != self.config['board_name']:
            raise ToolError('Caller is not in this trial', 'unknown_agent')
        if name not in self.tools:
            raise ToolError('Unknown task tool', 'unknown_tool')
        validate(args, self.tools[name]['parameters'])
        if not agent.request_id:
            raise ToolError('A request ID is required', 'missing_request_id')
        key = (agent.name, agent.request_id)
        signature = json.dumps([name, args], sort_keys=True)
        async with self.lock:
            if key in self.cache:
                old, result = self.cache[key]
                if old != signature:
                    raise ToolError('Request ID reused with different arguments', 'conflict')
                return result
            if not self.started or self.sealed:
                raise ToolError('Task is not accepting actions', 'task_closed')
            self.sequence += 1
            event = {'sequence': self.sequence, 'trial_id': self.config['attempt_id'],
                     'agent': agent.name, 'conversation_id':
                     f"{self.config['attempt_id']}:{agent.name}", 'turn': agent.turn,
                     'request_id': agent.request_id, 'tool': name, 'args': args,
                     'started_at': time.time()}
            self._record({**event, 'phase': 'started'})
            try:
                result = await self.client.step({'tool': name, 'args': args, 'agent': agent.name})
            except Exception as exc:
                self.fail(exc)
                self._record({**event, 'phase': 'failed', 'error': str(exc)})
                raise ToolError('Environment lost; attempt invalidated', 'environment_lost') from exc
            self.cache[key] = (signature, result)
            self._record({**event, 'phase': 'completed', 'finished_at': time.time(),
                          'result': result})
            if result.get('done'):
                self.sealed = True
                self.reason = result.get('reason', 'submission')
                write_json(self.output / 'session.json', self.status())
            return result

    def _record(self, event):
        with (self.output / 'actions.jsonl').open('a') as f:
            f.write(json.dumps(event, ensure_ascii=False) + '\n')
            f.flush()
