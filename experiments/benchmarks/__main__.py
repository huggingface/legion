"""CLI for reviewable, inference-only benchmark campaigns."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import re
import stat
import subprocess
from pathlib import Path

from .campaign import catalog, prepare, verify_manifest
from .common import HERE, ROOT, read_json, write_json
from .runner import run, validate


def load_secrets(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError('Secrets file must be an owner-only regular file')
    values = read_json(path)
    if set(values) - {'HF_TOKEN', 'LOCAL_KEY'} or not all(isinstance(v, str) for v in values.values()):
        raise ValueError('Only HF_TOKEN and LOCAL_KEY are allowed in the secrets file')
    os.environ.update(values)


def submit(path, secrets_file, dry_run=False):
    campaign = read_json(path)
    receipt = path.parent / 'submission.json'
    with (path.parent / '.submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt.exists():
            return read_json(receipt)
        intent = path.parent / 'submission-intent.json'
        if intent.exists():
            raise RuntimeError('Submission outcome uncertain; reconcile scheduler before retrying')
        configs = [verify_manifest(Path(p)) for p in campaign['attempts']]
        if not configs:
            raise ValueError('Empty campaign')
        s = configs[0]['slurm']
        command = ['sbatch', '--parsable', '--export=NONE',
                   '--array', f'0-{len(configs)-1}%1', '--partition', s['partition'],
                   '--cpus-per-task', str(s['cpus']), '--mem', s['memory'], '--time', s['time'],
                   '--output', str(path.parent / 'slurm-%A_%a.log'), str(HERE / 'run.slurm'),
                   str(ROOT / '.venv/bin/python'), str(path.resolve()), str(secrets_file.resolve())]
        if dry_run:
            return {'command': command}
        # Validate permissions without putting values in sbatch arguments or its environment.
        previous = {k: os.environ.get(k) for k in ('HF_TOKEN', 'LOCAL_KEY')}
        try:
            load_secrets(secrets_file)
        finally:
            for k, v in previous.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        env = {k: v for k, v in os.environ.items() if k in {'PATH', 'HOME', 'USER', 'LANG'}}
        write_json(intent, {'command': command, 'state': 'submitting'})
        job = subprocess.check_output(command, cwd=ROOT, env=env, text=True).strip().split(';')[0]
        if not re.fullmatch(r'\d+', job):
            raise RuntimeError('Unrecognized Slurm job receipt; inspect scheduler before retrying')
        result = {'job_id': job, 'command': command}
        write_json(receipt, result)
        return result


def scheduler(path, cancel=False):
    receipt = read_json(path.parent / 'submission.json')
    job = receipt['job_id']
    if not re.fullmatch(r'\d+', job):
        raise ValueError('Invalid recorded job ID')
    command = ['scancel', job] if cancel else ['sacct', '-j', job, '--parsable2',
                                               '--format=JobID,State,ExitCode']
    return {'job_id': job, 'output': subprocess.check_output(command, text=True)}


def analyze(path):
    rows = []
    for attempt in read_json(path)['attempts']:
        folder = Path(attempt).parent
        if not (folder / 'result.json').exists():
            continue
        r = read_json(folder / 'result.json')
        rows.append({k: r.get(k) for k in ('attempt_id', 'benchmark', 'task_id', 'agents',
                     'mode', 'success', 'valid', 'reason', 'elapsed_seconds', 'normalized_tokens')})
    with (path.parent / 'results.csv').open('w') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    groups = {}
    for row in rows:
        key = f"{row['benchmark']}:{row['task_id']}:n{row['agents']}"
        groups.setdefault(key, []).append(row)
    summary = {}
    for key, values in groups.items():
        valid = [r for r in values if r['valid']]
        tokens = [r['normalized_tokens'] for r in valid if r['normalized_tokens'] is not None]
        summary[key] = {'attempts': len(values), 'valid_attempts': len(valid),
                        'successes': sum(r['success'] is True for r in valid),
                        'token_samples': len(tokens),
                        'mean_normalized_tokens': sum(tokens) / len(tokens) if tokens else None}
    write_json(path.parent / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('catalog')
    p = sub.add_parser('prepare')
    p.add_argument('campaign', type=Path)
    p.add_argument('--output', type=Path, required=True)
    for name in ('validate', 'run', 'reference', 'status', 'cancel', 'analyze', 'submit'):
        p = sub.add_parser(name)
        p.add_argument('path', type=Path)
        if name == 'validate':
            p.add_argument('--runtime', action='store_true')
        if name in ('submit', 'run'):
            p.add_argument('--secrets-file', type=Path, required=name == 'submit')
        if name == 'submit':
            p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('array-run')
    p.add_argument('path', type=Path)
    p.add_argument('secrets_file', type=Path)
    args = parser.parse_args()
    if args.command == 'catalog':
        result = catalog()
    elif args.command == 'prepare':
        result = {'campaign': str(prepare(args.campaign, args.output))}
    elif args.command == 'validate':
        result = validate(args.path, args.runtime)
    elif args.command in ('run', 'array-run'):
        if args.secrets_file:
            load_secrets(args.secrets_file)
        path = args.path
        if args.command == 'array-run':
            path = Path(read_json(path)['attempts'][int(os.environ['SLURM_ARRAY_TASK_ID'])])
        result = run(path)
    elif args.command == 'reference':
        # This command runs in the isolated runtime to import Harbor when needed.
        from .reference import reference
        result = reference(verify_manifest(args.path))
    elif args.command == 'submit':
        result = submit(args.path, args.secrets_file, args.dry_run)
    elif args.command in ('status', 'cancel'):
        result = scheduler(args.path, cancel=args.command == 'cancel')
    else:
        result = analyze(args.path)
    print(json.dumps(result, indent=2))
    if isinstance(result, dict) and (result.get('status') in {'failed', 'blocked'} or
                                    result.get('reason') == 'infrastructure_failure'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
