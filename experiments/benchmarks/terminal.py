"""Harbor Docker task lifecycle without Harbor's autonomous agent loop."""
from __future__ import annotations

import asyncio
import base64
import json
import shlex
import uuid
from pathlib import Path

from .common import write_json


class TerminalSession:
    def __init__(self, config, environment=None, task=None):
        self.config, self.env, self.task = config, environment, task
        self.jobs = {}
        self.active = None
        self.closed = False
        self.stopped = False
        self.submitted = False
        self.paths = None

    async def reset(self, **kwargs):
        if self.env is None:
            from harbor.environments.docker.docker import DockerEnvironment
            from harbor.models.task.task import Task
            from harbor.models.trial.paths import TrialPaths
            self.task = Task(Path(self.config['task_path']))
            if self.task.has_steps or self.task.config.environment.gpus:
                raise ValueError('Pilot supports single-step CPU tasks only')
            self.paths = TrialPaths(Path(self.config['output']) / 'harbor')
            self.paths.mkdir()
            self.env = DockerEnvironment(
                environment_dir=self.task.paths.environment_dir,
                environment_name='flock-' + self.config['attempt_id'],
                session_id=self.config['attempt_id'], trial_paths=self.paths,
                task_env_config=self.task.config.environment)
            await self.env.start(force_build=True)
            image = self.env._env_vars.main_image_name  # Harbor 0.25.0, pinned in runtime lock.
            process = await asyncio.create_subprocess_exec(
                'docker', 'image', 'inspect', '--format', '{{json .Id}}', image,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await asyncio.wait_for(process.communicate(), 15)
            if process.returncode:
                raise RuntimeError('Cannot record built task image: ' + stderr.decode())
            write_json(Path(self.config['output']) / 'environment.json', {
                'harbor_version': '0.25.0', 'image_id': json.loads(stdout),
                'environment_id': self.env.environment_id})
        self.instruction = self.task.instruction
        result = await self.env.exec('command -v bash timeout setsid base64 head tail', timeout_sec=10)
        if result.return_code:
            raise RuntimeError('Task image lacks required bounded-command utilities')
        return {'result': {'instruction': self.instruction}, 'done': False}

    def _busy(self):
        return self.active is not None and not self.jobs[self.active]['task'].done()

    async def _execute(self, job_id, command, timeout_s):
        job = self.jobs[job_id]
        output = f'/tmp/flock-{job_id}.log'
        job['output_path'] = output
        # Redirect in the container: arbitrary command output never accumulates on the host.
        # setsid isolates this shell's process group; EXIT terminates ordinary background jobs.
        script = "trap 'trap \"\" TERM; kill -TERM -- -$$ 2>/dev/null || true' EXIT\nulimit -f 16384\n" + command
        wrapped = (f'timeout --signal=TERM --kill-after=5 {timeout_s} '
                   f'setsid bash -lc {shlex.quote(script)} > {output} 2>&1')
        try:
            r = await self.env.exec(wrapped, timeout_sec=timeout_s + 15)
            job.update(status='completed', return_code=r.return_code)
        except asyncio.CancelledError:
            job['status'] = 'cancelled'
            raise
        except Exception as exc:  # noqa: BLE001 - retain backend failure as explicit result
            job.update(status='failed', error=str(exc))

    async def step(self, tool, args, agent):
        if self.closed or self.submitted:
            return {'error': 'Task closed', 'done': True}
        if any(j['status'] == 'failed' for j in self.jobs.values()):
            raise RuntimeError('Sandbox command failed at the infrastructure layer')
        if tool == 'get_observation':
            return {'result': {'instruction': self.instruction, 'active_job':
                               self.active if self._busy() else None}, 'done': False}
        if tool in {'exec_status', 'exec_output', 'exec_cancel'}:
            job = self.jobs.get(args['job_id'])
            if job is None:
                return {'error': 'Unknown job handle', 'done': False}
            if tool == 'exec_cancel':
                await self.close()
                return {'done': True, 'reason': 'cancelled', 'result': 'Sandbox stopped'}
            if tool == 'exec_status':
                return {'result': {k: v for k, v in job.items()
                                   if k not in {'task', 'output_path'}}, 'done': False}
            if self._busy():
                return {'error': 'Output is available after the active command completes',
                        'done': False}
            offset = args.get('offset', 0)
            r = await self.env.exec(
                f"dd if={shlex.quote(job['output_path'])} bs=32768 iflag=skip_bytes,count_bytes "
                f"skip={offset} count=32768 status=none | base64",
                timeout_sec=10)
            if r.return_code:
                raise RuntimeError('Cannot retrieve sandbox output')
            data = base64.b64decode(r.stdout or '')
            return {'result': {'text': data.decode(errors='replace'),
                               'next_offset': offset + len(data)}, 'done': False}
        if self._busy():
            return {'error': 'Shared sandbox busy; poll the active job before another mutation',
                    'job_id': self.active, 'done': False}
        if tool == 'exec_start':
            job_id = uuid.uuid4().hex
            self.jobs[job_id] = {'status': 'running', 'agent': agent}
            self.active = job_id
            self.jobs[job_id]['task'] = asyncio.create_task(
                self._execute(job_id, args['command'], args.get('timeout_s', 60)))
            return {'result': {'job_id': job_id}, 'done': False}
        if tool in {'fs_read', 'fs_write'}:
            path = shlex.quote(args['path'])
            if tool == 'fs_read':
                command = (f"dd if={path} bs=32768 iflag=skip_bytes,count_bytes "
                           f"skip={args.get('offset', 0)} count=32768 status=none | base64")
            else:
                encoded = base64.b64encode(args['content'].encode()).decode()
                command = f"printf %s {shlex.quote(encoded)} | base64 -d > {path}"
            r = await self.env.exec("bash -o pipefail -c " + shlex.quote(command), timeout_sec=10)
            if r.return_code:
                return {'error': 'File operation failed', 'done': False}
            result = (base64.b64decode(r.stdout or '').decode(errors='replace')
                      if tool == 'fs_read' else {'written': len(args['content'].encode())})
            return {'result': result, 'done': False}
        if tool == 'task_submit':
            self.submitted = True
            return {'done': True, 'reason': 'submission', 'result': 'Team submission sealed'}
        raise ValueError('Unknown terminal action')

    async def evaluate(self):
        if self.closed:
            raise RuntimeError('Cannot grade a stopped sandbox')
        if self._busy():
            await self.jobs[self.active]['task']
        if any(j['status'] == 'failed' for j in self.jobs.values()):
            raise RuntimeError('Cannot grade after sandbox infrastructure failure')
        from harbor.verifier.verifier import Verifier
        # Preserve the pilot's submitted artifact before the verifier executes or cleanup runs.
        exists = await self.env.exec('test -f /app/recovered.json', timeout_sec=10)
        if exists.return_code == 0:
            artifact = Path(self.config['output']) / 'submission' / 'recovered.json'
            artifact.parent.mkdir(exist_ok=True)
            await self.env.download_file('/app/recovered.json', artifact)
        result = await Verifier(task=self.task, trial_paths=self.paths,
                                environment=self.env).verify()
        metrics = result.rewards
        if not metrics or 'reward' not in metrics:
            raise ValueError('Terminal-Bench verifier did not return its native reward')
        report = {'success': metrics['reward'] == 1, 'metrics': metrics,
                  'grader': 'Harbor native verifier', 'valid': True}
        write_json(Path(self.config['output']) / 'grader.json', report)
        return report

    async def close(self):
        if self.stopped:
            return
        self.closed = True
        tasks = [j['task'] for j in self.jobs.values()]
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self.env is not None:
            await self.env.stop(delete=True)
        self.stopped = True
