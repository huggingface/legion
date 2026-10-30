"""Prepare immutable attempts and compile ordinary Flock specs."""
from __future__ import annotations

import copy
import re
import subprocess
import tomllib
import uuid
from datetime import UTC, datetime
from pathlib import Path

import yaml

from flock.models import SwarmSpec

from .common import ESCAPE_REVISION, HERE, ROOT, digest, read_json, tree_hash, write_json


def checkout_revision(path, expected):
    actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain',
                                     '--untracked-files=no'], text=True).strip()
    if actual != expected or dirty:
        raise ValueError(f'Task checkout must be clean and pinned to {expected}: {path}')


def resolve_task(selection):
    benchmark = selection['benchmark']
    if benchmark == 'agent_escape':
        source = ROOT / '.flock/deps/agent-escape'
        checkout_revision(source, ESCAPE_REVISION)
        difficulty = selection.get('difficulty', 5)
        if difficulty not in (5, 10, 15, 20, 25):
            raise ValueError('Invalid Agent Escape difficulty')
        cases = sorted((source / 'questions' / str(difficulty) / 'narrative').glob('*.json'))
        index = selection.get('scenario_index', 0)
        if not isinstance(index, int) or not 0 <= index < len(cases):
            raise ValueError('Invalid scenario index')
        path = cases[index]
        return {'benchmark': benchmark, 'task_id': path.stem, 'task_path': str(path),
                'task_hash': digest(path), 'dataset_revision': ESCAPE_REVISION, 'source_dir': str(source),
                'difficulty': difficulty, 'scenario_index': index}
    if benchmark == 'terminal_bench':
        source = read_json(HERE / 'terminal-source.json')
        task_id = selection['task_id']
        if task_id != source['name']:
            raise ValueError('Initial Terminal-Bench pilot supports only the pinned db-wal-recovery')
        checkout = ROOT / '.flock/deps/terminal-bench-2'
        checkout_revision(checkout, source['git_commit_id'])
        path = checkout / source['path']
        for name in ('task.toml', 'instruction.md', 'environment', 'tests', 'solution'):
            if not (path / name).exists():
                raise ValueError(f'Missing task component: {name}')
        return {'benchmark': benchmark, 'task_id': task_id, 'task_path': str(path),
                'task_hash': tree_hash(path), 'dataset_revision': source['git_commit_id'],
                'source_dir': str(checkout),
                'dataset': 'terminal-bench@2.0',
                'native_wall_seconds': tomllib.loads((path / 'task.toml').read_text())['agent']['timeout_sec']}
    raise ValueError(f'Unknown benchmark: {benchmark}')


def compile_spec(profile, manifest, prompt_path):
    spec = copy.deepcopy(profile)
    if set(spec['types']) != {'worker'} or set(spec['providers']) != {'local'}:
        raise ValueError('Pilot profile requires one worker type and one local provider')
    provider = spec['providers']['local']
    if provider.get('api_key_env') != 'LOCAL_KEY' or provider.get('type') != 'openai_compat':
        raise ValueError('Pilot requires openai_compat with api_key_env: LOCAL_KEY')
    if any(k in provider for k in ('api_key', 'token')):
        raise ValueError('Use environment credential references, not inline credentials')
    worker = spec['types']['worker']
    worker.update(harness='pi', replicas=manifest['agents'], agents_per_worker=1,
                  max_working_per_worker=1, pi_tools_only=True, reasoning='low',
                  system_prompt_file=str(prompt_path), max_turns=200, turn_timeout='35m')
    worker.pop('tools', None)
    worker['env'] = {'FLOCK_COMMS_RATE': '0', 'FLOCK_COMMS_AGENT_RATE': '0'}
    provider['max_working'] = manifest['agents']
    spec.update(swarm=manifest['board_name'], compute='local')
    spec.pop('spawn', None)
    spec['world'] = {'url': 'http://127.0.0.1:1/world', 'token_env': 'FLOCK_WORLD_TOKEN'}
    spec['comms']['poll_cycle_s'] = 5
    spec['comms']['kickoff'] = ('One shared task for the entire team. Coordinate ownership of '
                              'work and share verified findings; task submission ends the team run.')
    SwarmSpec.model_validate(spec)
    return spec


def prepare(campaign_path: Path, output: Path):
    cfg = yaml.safe_load(campaign_path.read_text())
    if cfg.get('version') != 1:
        raise ValueError('Campaign version must be 1')
    profile_path = (campaign_path.parent / cfg['execution_profile']).resolve()
    profile = yaml.safe_load(profile_path.read_text())
    counts = cfg.get('agent_counts', [1, 2])
    if len(set(counts)) != len(counts) or any(type(n) is not int or not 1 <= n <= 32 for n in counts):
        raise ValueError('Agent counts must be unique integers between 1 and 32')
    repeats = cfg.get('repetitions', 1)
    if type(repeats) is not int or repeats < 1:
        raise ValueError('Repetitions must be a positive integer')
    limits = cfg.get('limits', {})
    for name in ('wall_seconds', 'setup_timeout_s', 'grading_timeout_s'):
        if limits.get(name, 1) <= 0:
            raise ValueError(f'{name} must be positive')
    tasks = [resolve_task(s) for s in cfg['tasks']]
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    attempts = []
    stamp = datetime.now(UTC).strftime('%Y%m%dt%H%M%Sz')
    for task in tasks:
        for n in counts:
            for repetition in range(1, repeats + 1):
                aid = uuid.uuid4().hex
                slug = re.sub('[^a-z0-9-]', '-', task['task_id'].lower())[:24]
                board = (f"{task['benchmark'].replace('_', '-')}-{slug}-shared-n{n}"
                         f'-r{repetition}-{stamp}-{aid[:8]}')
                folder = output / board
                folder.mkdir(mode=0o700)
                m = {**task, 'version': 1, 'attempt_id': aid, 'board_name': board,
                     'mode': 'shared_task', 'agents': n, 'repetition': repetition,
                     'seed': cfg.get('seed', 42), 'output': str(folder),
                     'wall_seconds': min(limits.get('wall_seconds', 1800),
                                         task.get('native_wall_seconds', 1800)),
                     'setup_timeout_s': limits.get('setup_timeout_s', 600),
                     'grading_timeout_s': limits.get('grading_timeout_s', 300),
                     'slurm': cfg.get('slurm', {'partition': 'hopper-cpu', 'cpus': 8,
                                               'memory': '16G', 'time': '01:15:00'})}
                prompt = folder / 'prompt.md'
                prompt.write_text((HERE / 'prompt.md').read_text())
                spec_path = folder / 'spec.yaml'
                spec_path.write_text(yaml.safe_dump(compile_spec(profile, m, prompt), sort_keys=False))
                sources = list(HERE.glob('*.py')) + [HERE / 'requirements.lock',
                          HERE / 'run.slurm', ROOT / 'experiments/agent-escape/usage_proxy.py',
                          spec_path, prompt] + list((ROOT / 'src/flock').rglob('*.py'))
                sources += [ROOT / 'src/flock/worker/adapters/pi_tools.mjs']
                m['source_hashes'] = {str(p.resolve()): digest(p) for p in sources}
                m['flock_revision'] = subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
                write_json(folder / 'manifest.json', m)
                attempts.append(str(folder / 'manifest.json'))
    write_json(output / 'campaign.json', {'version': 1, 'attempts': attempts,
                                         'max_active_attempts': 1})
    return output / 'campaign.json'


def verify_manifest(path):
    m = read_json(path)
    for filename, expected in m['source_hashes'].items():
        if digest(Path(filename)) != expected:
            raise ValueError(f'Prepared source changed; prepare a new campaign: {filename}')
    if m.get('source_dir'):
        checkout_revision(Path(m['source_dir']), m['dataset_revision'])
    task_path = Path(m['task_path'])
    actual = digest(task_path) if task_path.is_file() else tree_hash(task_path)
    if actual != m['task_hash']:
        raise ValueError('Prepared task contents changed')
    return m


def catalog():
    return [{'benchmark': 'agent_escape', 'difficulties': [5, 10, 15, 20, 25],
             'requires': 'Pinned upstream checkout and its Python environment'},
            {'benchmark': 'terminal_bench', 'dataset': 'terminal-bench@2.0',
             'task_ids': ['db-wal-recovery'], 'requires': 'Docker daemon, Compose and Harbor'}]
