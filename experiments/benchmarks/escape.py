"""Adapter to the unmodified, pinned AgentEscapeBench HTTP server."""
from __future__ import annotations

import httpx


class EscapeSession:
    def __init__(self, config, client=None):
        self.config = config
        self.client = client or httpx.AsyncClient(base_url=config['escape_url'], timeout=20)
        self.case_id = None
        self.observation = None
        self.success = False

    async def reset(self, **kwargs):
        if self.case_id is not None:
            raise RuntimeError('Shared escape session cannot be reset')
        r = await self.client.post('/env/reset', params={
            'case_file_path': self.config['task_path'], 'session_id': self.config['attempt_id']})
        r.raise_for_status()
        data = r.json()
        self.case_id = data['case_id']
        self.observation = {k: data[k] for k in
                            ('narrative', 'available_entities', 'available_information',
                             'win_node_id') if k in data}
        return {'result': self.observation, 'done': False}

    async def step(self, tool, args, agent):
        if tool == 'get_observation':
            return {'result': self.observation, 'done': self.success}
        response = await self.client.post(f'/env/{self.case_id}/{tool}', json=args)
        if response.status_code >= 500:
            response.raise_for_status()
        if response.status_code >= 400:
            return {'error': 'Invalid puzzle action', 'done': False}
        result = response.json()
        self.success = tool == 'submit_answer' and result.get('is_win') is True
        return {'result': result, 'done': self.success,
                **({'reason': 'native_success'} if self.success else {})}

    async def evaluate(self):
        return {'success': self.success, 'metrics': {'is_win': self.success},
                'grader': 'AgentEscapeBench submit_answer', 'valid': True}

    async def close(self):
        await self.client.aclose()
