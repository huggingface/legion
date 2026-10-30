"""Gated difficulty-20 communication ablation against a frozen historical baseline."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import random
import statistics
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path

import httpx
import slurm
import sweep

ROOT, HERE = slurm.ROOT, slurm.HERE
PUZZLE_TOOLS = {'get_observation', 'investigate', 'tool_call', 'use_item', 'submit_answer'}
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'NODE_FAIL', 'TIMEOUT', 'OUT_OF_MEMORY',
            'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def matrix():
    rows = [{'difficulty': 20, 'agents': n, 'trial': t, 'communication': 'off'}
            for n in sweep.COUNTS for t in range(1, 4)]
    pilot = next(r for r in rows if r['agents'] == 8 and r['trial'] == 1)
    rows.remove(pilot)
    random.Random(42).shuffle(rows)
    return [pilot, *rows]


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate_attempt(task):
    """Success is not required; isolation, complete accounting and clean operation are."""
    summary = slurm.read_json(Path(task['config']).parent / 'summary.json')[0]
    result, attempt = summary['result'], Path(summary['attempt'])
    problems = []
    if (result.get('error') or result.get('flock_exit_code') != 0
            or result.get('infrastructure_outcome') != 'ok'):
        problems.append('unclean runtime or infrastructure failure')
    accounting = result.get('accounting', {})
    if not accounting.get('complete') or len(accounting.get('session_ids', [])) != task['agents']:
        problems.append('incomplete accounting or agent roster')
    if result.get('communication') != 'off' or result.get('persistence_status') != 'not_applicable':
        problems.append('incorrect condition/persistence metadata')
    if list((attempt / 'flock').glob('*/board/metadata.json')):
        problems.append('unexpected board process')
    spec = slurm.read_json(attempt / 'metadata.json')['spec']
    if spec.get('comms') or spec['types']['worker'].get('comms') is not False:
        problems.append('communication was configured')
    journal = records(attempt / 'model-requests.jsonl')
    starts = [r for r in journal if r.get('event') == 'start']
    if not starts or any(set(r.get('tools', [])) != PUZZLE_TOOLS for r in starts):
        problems.append('unexpected offered tools')
    if any(r.get('event') == 'barrier_timeout' for r in journal):
        problems.append('startup barrier timeout')
    trajectory = records(attempt / 'trajectory.jsonl')
    if len({r['agent'] for r in trajectory if r['action'] == 'initialize'}) != task['agents']:
        problems.append('not all agents initialized independent puzzle sessions')
    if any('collaboration_reminder' in r.get('result', {}) for r in trajectory):
        problems.append('communication reminders were emitted')
    for path in (attempt / 'flock').glob('*/events/*.jsonl'):
        for event in records(path):
            if event.get('type', '').startswith('comms_'):
                problems.append('communication event in transcript')
            content = event.get('data', {}).get('message', {}).get('content', [])
            if isinstance(content, list):
                for block in content:
                    if (isinstance(block, dict) and block.get('type') == 'toolCall'
                            and block.get('name') not in PUZZLE_TOOLS):
                        problems.append('unexpected tool call in transcript')
    return sorted(set(problems))


def metrics(row, condition):
    result = {k: row.get(k) for k in ('agents', 'trial', 'attempt', 'success',
        'measurement_valid', 'infrastructure_outcome', 'puzzle_outcome', 'scheduler_state')}
    result.update(communication=condition, generated_tokens=None,
                  generated_tokens_per_agent=None, work_time_to_success_seconds=None,
                  valid_trial=False)
    if not row.get('attempt'):
        return result
    attempt = Path(row['attempt'])
    report = sweep.read_report(attempt / 'result.json', {})
    accounting = report.get('accounting', {})
    result['valid_trial'] = bool(row.get('measurement_valid') and accounting.get('complete')
        and row.get('infrastructure_outcome') == 'ok' and report.get('flock_exit_code') == 0)
    if result['valid_trial']:
        result['generated_tokens'] = accounting['output_tokens']
        result['generated_tokens_per_agent'] = accounting['output_tokens'] / row['agents']
    if row.get('success') and result['valid_trial']:
        times = [r['time'] for r in records(attempt / 'model-requests.jsonl')
                 if r.get('event') == 'barrier_opened']
        if len(times) != 1:
            raise ValueError(f'Expected one startup barrier timestamp: {attempt}')
        seconds = report['success']['time'] - times[0]
        if seconds < 0:
            raise ValueError(f'Solution before startup barrier: {attempt}')
        result['work_time_to_success_seconds'] = seconds
    return result


def analyze(root, new_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    baseline = slurm.read_json(root / 'baseline.json')
    rows = [metrics(r, 'on') for r in baseline] + [metrics(r, 'off') for r in new_rows]
    slurm.write_json(root / 'comparison-results.json', rows)
    with (root / 'comparison-results.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for condition in ['on', 'off']:
        for n in sweep.COUNTS:
            group = [r for r in rows if r['communication'] == condition and r['agents'] == n]
            valid = [r for r in group if r['valid_trial']]
            solved = [r for r in valid if r['success']]
            record = {'communication': condition, 'agents': n, 'planned': 3,
                      'valid_trials': len(valid), 'successes': len(solved),
                      'missing_or_infrastructure_failures': 3 - len(valid),
                      'success_rate': len(solved) / len(valid) if valid else None}
            for key in ['generated_tokens_per_agent', 'work_time_to_success_seconds']:
                values = [r[key] for r in solved]
                record[key + '_mean'] = statistics.mean(values) if values else None
                record[key + '_sd'] = statistics.stdev(values) if len(values) > 1 else None
            summary.append(record)
    slurm.write_json(root / 'comparison-summary.json', summary)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    fig.subplots_adjust(left=.065, right=.98, bottom=.27, top=.69, wspace=.3)
    definitions = [('generated_tokens_per_agent', 'Generation cost', 'Generated tokens / agent'),
                   ('success_rate', 'Success rate', 'Successful / valid trials (%)'),
                   ('work_time_to_success_seconds', 'Time to solution', 'Minutes from first-request barrier')]
    for ax, (key, title, ylabel) in zip(axes, definitions, strict=True):
        bounds = [0]
        for condition, color, marker in [('on', '#111111', 's'), ('off', '#217dc0', 'o')]:
            group = [r for r in summary if r['communication'] == condition]
            factor = 100 if key == 'success_rate' else 1 / 60 if key.endswith('seconds') else 1
            means = [r[key if key == 'success_rate' else key + '_mean'] for r in group]
            ax.plot(range(5), [v * factor if v is not None else float('nan') for v in means],
                    color=color, marker=marker, linewidth=2.2, markersize=6)
            for x, (r, mean) in enumerate(zip(group, means, strict=True)):
                if key == 'success_rate':
                    ax.text(x, 1.06 if condition == 'on' else 1.005,
                            f"{r['successes']}/{r['valid_trials']} ({r['valid_trials']}/3)",
                            transform=ax.get_xaxis_transform(), ha='center', color=color, fontsize=8)
                if mean is None:
                    continue
                sd = r.get(key + '_sd')
                bounds.extend([(mean - (sd or 0)) * factor, (mean + (sd or 0)) * factor])
                if sd is not None:
                    ax.errorbar(x, mean * factor, yerr=sd * factor, color=color, capsize=4)
            # The two success-rate series can overlap; the count rows retain both conditions.
        low, high = min(bounds), max(bounds)
        ax.set_ylim((0, 110) if key == 'success_rate' else
                    (min(0, low * 1.05), max(1, high * 1.15)))
        ax.set_title(title, loc='left', fontsize=16, weight='bold', pad=35, color='#1e293b')
        ax.set_xticks(range(5), sweep.COUNTS)
        ax.set_xlabel('Number of agents', color='#64748b')
        ax.set_ylabel(ylabel, color='#64748b')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
        ax.grid(axis='y', color='#dde5ef')
        ax.tick_params(colors='#64748b', labelsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#cbd5e1')
    fig.suptitle('Does communication improve swarm performance?', x=.065, y=.96,
                 ha='left', fontsize=24, weight='bold', color='#1e293b')
    fig.text(.065, .88, 'AgentEscapeBench · 20-node puzzle · scenario 0 · Pi / Qwen3.8-27B',
             color='#64748b', fontsize=12)
    fig.legend(handles=[Line2D([0], [0], color=c, marker=m, linewidth=2.2, label=label)
        for c, m, label in [('#111111', 's', 'Communication on · historical baseline'),
                           ('#217dc0', 'o', 'Communication off')]],
        loc='upper left', bbox_to_anchor=(.06, .84), ncols=2, frameon=False)
    fig.text(.065, .17, 'Cost and time: successful valid trials, mean ± sample SD. '
             'Count labels: successes / valid trials (coverage / 3 planned).',
             color='#64748b', fontsize=10)
    fig.text(.065, .12, 'Generated tokens include shutdown/draining and exclude all input. '
             'Missing infrastructure results are not puzzle failures; no SD for a single success.',
             color='#64748b', fontsize=10)
    fig.text(.065, .07, 'Historical, single-scenario comparison; serving conditions and sampling '
             'differ. Trials are not paired. Success-conditioned means can reflect selection bias.',
             color='#64748b', fontsize=10)
    for suffix in ['png', 'svg', 'pdf']:
        fig.savefig(root / f'communication-ablation.{suffix}', dpi=180, facecolor='white')
    plt.close(fig)


def batch(root, mode, *, array=None, dependency=None):
    command = ['sbatch', '--parsable', '--export=NONE', '--partition=hopper-cpu',
               '--cpus-per-task=8', '--mem=16G', '--time=01:30:00',
               '--output=' + str(root / (mode + '-%A_%a.log')),
               '--job-name=' + root.name + '-' + mode]
    if array:
        command += ['--array=' + array]
    if dependency:
        command += ['--dependency=afterany:' + str(dependency)]
    command += [str(HERE / 'ablation.slurm'), mode, str(root)]
    env = {k: v for k, v in os.environ.items() if k in {'PATH', 'HOME', 'USER', 'LOGNAME', 'LANG'}}
    job = subprocess.check_output(command, cwd=ROOT, env=env, text=True).strip().split(';')[0]
    return {'job_id': job, 'command': command}


def submit(args):
    baseline = [r for r in slurm.read_json(args.baseline_results) if r['difficulty'] == 20]
    assert len(baseline) == 15 and sum(r['success'] for r in baseline) == 11
    assert len({(r['agents'], r['trial']) for r in baseline}) == 15
    endpoint = slurm.read_json(args.server_run / 'endpoint.json')
    assert endpoint['model'] == 'Qwen/Qwen3.8-27B'
    assert slurm.read_json(args.server_run / 'config.json')['context'] == 65536
    key = (args.server_run / 'api-key').read_text().strip()
    response = httpx.get(endpoint['base_url'] + '/models',
                        headers={'Authorization': 'Bearer ' + key}, timeout=15)
    response.raise_for_status()
    assert endpoint['model'] in [m['id'] for m in response.json()['data']]
    root = ROOT / '.flock/agent-escape-ablations' / (
        datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ-comm-off-') + uuid.uuid4().hex[:8])
    root.mkdir(parents=True, mode=0o700)
    slurm.write_json(root / 'baseline.json', baseline)
    # Use an actual frozen baseline spec, not an independently drifting template.
    source_spec = Path(slurm.read_json(baseline[0]['config'])['spec'])
    tasks = []
    for i, row in enumerate(matrix()):
        config = slurm.prepare(argparse.Namespace(server_run=args.server_run, spec=source_spec,
            difficulties=[20], scenario_index=0, timeout=3600, agents=row['agents'],
            trial=row['trial'], attempt_number=1, communication='off', output=root / f'task-{i:02d}'))
        tasks.append({**row, 'config': str(config / 'config.json')})
    sources = [HERE / n for n in ['ablation.py', 'ablation.slurm', 'sweep.py']]
    slurm.write_json(root / 'manifest.json', {'tasks': tasks, 'baseline_source': str(args.baseline_results),
        'baseline_sha256': hashlib.sha256(args.baseline_results.read_bytes()).hexdigest(),
        'server_run': str(args.server_run), 'seed': 42, 'condition': 'off',
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}})
    slurm.write_json(root / 'pilot-submit-intent.json', {'job_name': root.name + '-pilot'})
    pilot = batch(root, 'pilot')
    slurm.write_json(root / 'pilot-submission.json', pilot)
    print(json.dumps({'root': str(root), 'pilot': pilot['job_id'], 'trials': 15,
                      'remaining': '14 will be submitted only after pilot validation'}))


def advance(root, doc):
    with (root / '.gate.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        problems = validate_attempt(doc['tasks'][0])
        slurm.write_json(root / 'validation.json', {'passed': not problems, 'errors': problems})
        if problems:
            print(json.dumps({'validation_failed': problems}))
            return 1
        if (root / 'array-submission.json').exists():
            submission = slurm.read_json(root / 'array-submission.json')
        else:
            if (root / 'array-submit-intent.json').exists():
                raise RuntimeError('Ambiguous submission: reconcile scheduler before resubmitting')
            slurm.write_json(root / 'array-submit-intent.json', {'job_name': root.name + '-task'})
            submission = batch(root, 'task', array='1-14%8')
            slurm.write_json(root / 'array-submission.json', submission)
        if not (root / 'finalizer-submission.json').exists():
            if (root / 'finalizer-submit-intent.json').exists():
                raise RuntimeError('Ambiguous finalizer submission: reconcile scheduler')
            slurm.write_json(root / 'finalizer-submit-intent.json', {'dependency': submission['job_id']})
            finalizer = batch(root, 'finalize', dependency=submission['job_id'])
            slurm.write_json(root / 'finalizer-submission.json', finalizer)
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('submit')
    p.add_argument('--baseline-results', type=Path, required=True)
    p.add_argument('--server-run', type=Path, required=True)
    for mode in ['pilot', 'advance', 'task', 'finalize']:
        sub.add_parser(mode).add_argument('root', type=Path)
    args = parser.parse_args()
    if args.mode == 'submit':
        submit(args)
        return 0
    root = args.root
    doc = slurm.read_json(root / 'manifest.json')
    for filename, digest in doc['source_sha256'].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
            raise ValueError('Ablation source changed after submission')
    if args.mode == 'pilot':
        slurm.run(doc['tasks'][0]['config'])
        return advance(root, doc)
    if args.mode == 'advance':
        return advance(root, doc)
    if args.mode == 'task':
        return slurm.run(doc['tasks'][int(os.environ['SLURM_ARRAY_TASK_ID'])]['config'])
    if (root / 'finalized.json').exists():
        return 0
    pilot_job = slurm.read_json(root / 'pilot-submission.json')['job_id']
    array_job = slurm.read_json(root / 'array-submission.json')['job_id']
    states = {**sweep.scheduler_states(pilot_job), **sweep.scheduler_states(array_job)}
    mapped = {str(i): states.get(pilot_job if i == 0 else f'{array_job}_{i}') for i in range(15)}
    if any(s not in TERMINAL for s in mapped.values()):
        raise RuntimeError('Missing or nonterminal jobs; retry finalization after completion')
    slurm.write_json(root / 'scheduler-states.json', mapped)
    rows = [sweep.attempt_result(task, mapped[str(i)]) for i, task in enumerate(doc['tasks'])]
    slurm.write_json(root / 'ablation-results.json', rows)
    analyze(root, rows)
    slurm.write_json(root / 'finalized.json', {'complete': True})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
