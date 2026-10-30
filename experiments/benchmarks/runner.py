"""One bounded local attempt; no inference occurs during prepare or validate."""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import signal
import subprocess
import time
import uuid
from pathlib import Path

import httpx
import yaml

from .campaign import verify_manifest
from .common import ROOT, RUNTIME, clean_env, free_port, read_json, stop, write_json


def docker_preflight():
    for command in (['docker', 'info', '--format', '{{.ServerVersion}}'],
                    ['docker', 'compose', 'version']):
        result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or 'Docker preflight failed')


def validate(path, runtime=False):
    m = verify_manifest(path)
    if not RUNTIME.exists():
        raise RuntimeError('Run experiments/benchmarks/setup.sh first')
    if runtime and m['benchmark'] == 'terminal_bench':
        docker_preflight()
    return {'valid': True, 'runtime_checked': runtime, 'attempt_id': m['attempt_id']}


def launch(command, output, name, env, cwd=ROOT):
    with (output / f'{name}.log').open('w') as log:
        return subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)


def wait_http(url, process, headers=None, seconds=60):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'Service exited before readiness: {url}')
        try:
            response = httpx.get(url, headers=headers, timeout=1)
            if response.is_success:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise RuntimeError(f'Service readiness timed out: {url}')


class Services:
    def __init__(self, config):
        self.config = dict(config)
        self.output = Path(config['output'])
        self.processes = []
        self.world_token = uuid.uuid4().hex
        self.control_token = uuid.uuid4().hex
        self.url = None

    def start(self):
        env = clean_env()
        env['PYTHONPATH'] = os.pathsep.join([str(ROOT), str(ROOT / 'src')])
        if self.config['benchmark'] == 'agent_escape':
            port = free_port()
            source = ROOT / '.flock/deps/agent-escape'
            command = [str(ROOT / '.flock/deps/agent-escape-venv/bin/python'), '-m', 'uvicorn',
                       'server:app', '--app-dir', str(source), '--host', '127.0.0.1',
                       '--port', str(port)]
            p = launch(command, self.output, 'escape-server', env, cwd=self.output)
            self.processes.append(p)
            self.config['escape_url'] = f'http://127.0.0.1:{port}'
            wait_http(self.config['escape_url'] + '/health', p)
        else:
            docker_preflight()
        port = free_port()
        self.url = f'http://127.0.0.1:{port}'
        env.update(FLOCK_WORLD_TOKEN=self.world_token, BENCHMARK_CONTROL_TOKEN=self.control_token)
        path = self.output / 'runtime.json'
        write_json(path, self.config)
        p = launch([str(RUNTIME), '-m', 'experiments.benchmarks.service', str(path),
                    '--port', str(port)], self.output, 'environment', env)
        self.processes.append(p)
        wait_http(self.url + '/control/status', p, self.headers, seconds=90)
        self.control('start', timeout=self.config['setup_timeout_s'] + 15)

    @property
    def headers(self):
        return {'X-Benchmark-Control': self.control_token}

    def control(self, action, body=None, timeout=10):
        r = httpx.post(self.url + '/control/' + action, json=body or {},
                       headers=self.headers, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def status(self):
        if any(p.poll() is not None for p in self.processes):
            raise RuntimeError('Environment subprocess exited')
        r = httpx.get(self.url + '/control/status', headers=self.headers, timeout=5)
        r.raise_for_status()
        return r.json()

    def close(self):
        errors = []
        if self.url and self.processes and self.processes[-1].poll() is None:
            try:
                self.control('close', timeout=30)
            except Exception as exc:  # noqa: BLE001 - persist failure and finish owned cleanup
                errors.append(str(exc))
        for p in reversed(self.processes):
            try:
                stop(p)
            except Exception as exc:  # noqa: BLE001 - persist failure and finish owned cleanup
                errors.append(str(exc))
        return errors


def usage_module():
    spec = importlib.util.spec_from_file_location('benchmark_usage',
                                                 ROOT / 'experiments/agent-escape/usage_proxy.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(path: Path):
    m = verify_manifest(path)
    output = Path(m['output'])
    # Exclusive attempt marker: interruption requires a new prepared attempt, never a hidden reset.
    fd = os.open(output / 'started.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as f:
        json.dump({'time': time.time(), 'pid': os.getpid()}, f)
    services, flock, proxy = Services(m), None, None
    start = time.monotonic()
    reason, error, grade = 'infrastructure_failure', None, None
    proxy_url = None
    cleanup_errors = []

    def safe_stop(process):
        try:
            stop(process)
        except (OSError, subprocess.SubprocessError) as exc:
            cleanup_errors.append(str(exc))

    old_handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}

    def interrupt(*_):
        raise KeyboardInterrupt
    for s in old_handlers:
        signal.signal(s, interrupt)
    try:
        validate(path, runtime=True)
        if not os.environ.get('LOCAL_KEY') or not os.environ.get('HF_TOKEN'):
            raise RuntimeError('LOCAL_KEY and HF_TOKEN must be supplied at execution time')
        services.start()
        spec = yaml.safe_load((output / 'spec.yaml').read_text())
        provider = spec['providers']['local']
        # A read-only readiness check, never a test completion.
        r = httpx.get(provider['base_url'].rstrip('/') + '/models', timeout=15,
                      headers={'Authorization': 'Bearer ' + os.environ['LOCAL_KEY']})
        r.raise_for_status()
        spec['world']['url'] = services.url + '/world'
        proxy_port = free_port()
        proxy_url = f'http://127.0.0.1:{proxy_port}'
        proxy_env = clean_env()
        proxy_env['LOCAL_KEY'] = os.environ['LOCAL_KEY']
        proxy = launch([str(ROOT / '.venv/bin/python'),
                        str(ROOT / 'experiments/agent-escape/usage_proxy.py'),
                        '--port', str(proxy_port), '--agents', str(m['agents']),
                        '--upstream', provider['base_url'], '--output', str(output)],
                       output, 'usage-proxy', proxy_env)
        wait_http(proxy_url + '/health', proxy)
        provider['base_url'] = proxy_url + '/v1'
        effective = output / 'effective-spec.yaml'
        effective.write_text(yaml.safe_dump(spec, sort_keys=False))
        env = os.environ.copy()
        env['FLOCK_WORLD_TOKEN'] = services.world_token
        env['PATH'] = os.pathsep.join([str(ROOT / '.flock/tools'), str(ROOT / '.venv/bin'),
                                     env.get('PATH', '')])
        flock = launch([str(ROOT / '.venv/bin/flock'), 'run', str(effective),
                        '--skip-preflight', '--data-dir', str(output / 'flock'), '--until', 'idle',
                        '--prompt', ('Read get_observation. Coordinate on the board and solve '
                        'your shared task. Submit only when the team is ready.')], output, 'flock', env)
        deadline = time.monotonic() + m['wall_seconds']
        while True:
            state = services.status()
            if state['sealed']:
                reason, error = state['reason'], state['error']
                break
            if proxy.poll() is not None:
                raise RuntimeError('Usage proxy exited')
            if flock.poll() is not None:
                if flock.returncode:
                    raise RuntimeError(f'Flock exited with status {flock.returncode}')
                reason = 'agents_finished'
                break
            if time.monotonic() >= deadline:
                reason = 'timeout'
                break
            time.sleep(1)
        services.control('seal', {'reason': reason})
        stop(flock)
        if reason not in {'infrastructure_failure', 'cancelled'}:
            grade = services.control('evaluate', timeout=m['grading_timeout_s'] + 10)
    except KeyboardInterrupt:
        reason = 'cancelled'
    except Exception as exc:  # noqa: BLE001 - persist failure and finish owned cleanup
        error, reason = str(exc), 'infrastructure_failure'
    finally:
        for s in old_handlers:
            signal.signal(s, signal.SIG_IGN)
        if services.url:
            try:
                services.control('seal', {'reason': reason})
            except Exception as exc:  # noqa: BLE001 - cleanup must continue
                logging.getLogger(__name__).warning('Could not seal environment: %s', exc)
        safe_stop(flock)
        if proxy and proxy.poll() is None:
            try:
                httpx.post(proxy_url + '/close', timeout=5).raise_for_status()
                # Producers retain upstream ownership after Pi disconnects. Preserve trailing usage.
                drain_deadline = time.monotonic() + 905
                while httpx.get(proxy_url + '/health', timeout=5).json()['active']:
                    if time.monotonic() >= drain_deadline:
                        break  # summarize() marks unfinished requests incomplete, never zero.
                    time.sleep(1)
            except httpx.HTTPError:
                pass
        cleanup = services.close()
        safe_stop(proxy)
        cleanup.extend(cleanup_errors)
        if cleanup:
            error = '; '.join([error or '', *cleanup])
            reason = 'infrastructure_failure'
        accounting = usage_module().summarize(output / 'model-requests.jsonl')
        bundles = list((output / 'flock').glob('*/run.json'))
        bundle = read_json(bundles[0]) if len(bundles) == 1 else {}
        actions = 0
        journal = output / 'actions.jsonl'
        for line in journal.read_text().splitlines() if journal.exists() else []:
            try:
                actions += json.loads(line).get('phase') == 'completed'
            except (ValueError, AttributeError):
                error, reason = 'Incomplete action journal', 'infrastructure_failure'
        result = {'attempt_id': m['attempt_id'], 'benchmark': m['benchmark'],
                  'task_id': m['task_id'], 'agents': m['agents'], 'mode': m['mode'],
                  'reason': reason, 'error': error, 'grade': grade,
                  'success': grade.get('success') if grade else None,
                  'valid': reason not in {'infrastructure_failure', 'cancelled'} and grade is not None,
                  'elapsed_seconds': time.monotonic() - start, 'accounting': accounting,
                  'normalized_tokens': accounting['total_tokens'] / m['agents']
                  if accounting['complete'] else None,
                  'board_name': m['board_name'], 'tool_calls': actions,
                  'flock_stop_reason': bundle.get('stop_reason'),
                  'flock_bundle': str(bundles[0]) if len(bundles) == 1 else None}
        if str(result['flock_stop_reason']).startswith('board'):
            result.update(valid=False, reason='infrastructure_failure', error='Message board failed')
        write_json(output / 'result.json', result)
        for s, handler in old_handlers.items():
            signal.signal(s, handler)
    return result
