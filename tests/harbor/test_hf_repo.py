"""Regression checks for the collaboration tools' destructive/error boundaries."""

import importlib.util
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def tool(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "repo_tool", Path(__file__).parents[2] / "src/flock/harbor/builtin/tools/hf_repo.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("FLOCK_HF_REPO", "example/collaboration")
    monkeypatch.setenv("FLOCK_HF_REPO_AGENT", "agent-001")
    monkeypatch.setenv("FLOCK_HF_REPO_TOKEN", "secret-test-token")
    return module


@pytest.fixture
def checkout(tool, tmp_path):
    def git(*args):
        subprocess.run(
            ["git", "-C", str(tmp_path), *args], check=True, capture_output=True
        )

    git("init", "-b", "main")
    git("remote", "add", "origin", tool._url())
    git(
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--allow-empty",
        "-m",
        "initial",
    )
    return tmp_path


def test_explicit_staging_and_dirty_checkout(tool, checkout):
    (checkout / "code.py").write_text("print('original')\n")
    (checkout / "private.txt").write_text("not selected\n")
    tool.commit(str(checkout), ["code.py"], "Add original code")
    assert (
        subprocess.check_output(
            ["git", "-C", str(checkout), "ls-files"], text=True
        ).strip()
        == "code.py"
    )
    with pytest.raises(ValueError, match="local changes"):
        tool.sync(str(checkout))
    with pytest.raises(ValueError, match="relative file"):
        tool.commit(str(checkout), ["../private.txt"], "bad")
    with pytest.raises(ValueError, match="individual files"):
        tool.commit(str(checkout), ["." + "/"], "bad")
    (checkout / "executable").write_text("reference")
    with pytest.raises(ValueError, match="reference"):
        tool.commit(str(checkout), ["executable"], "bad")


def test_staged_changes_not_swept_into_commit(tool, checkout):
    (checkout / "private.txt").write_text("not selected")
    subprocess.run(["git", "-C", str(checkout), "add", "private.txt"], check=True)
    (checkout / "code.py").write_text("pass")
    with pytest.raises(ValueError, match="staged changes"):
        tool.commit(str(checkout), ["code.py"], "Add code")


def test_foreign_push_remote_rejected(tool, checkout):
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "remote",
            "set-url",
            "--push",
            "origin",
            "https://example.invalid/other",
        ],
        check=True,
    )
    with pytest.raises(ValueError, match="does not match"):
        tool.status(str(checkout))


def test_missing_git_is_actionable_and_redacted(tool, checkout, monkeypatch):
    monkeypatch.setenv("PATH", str(checkout))
    with pytest.raises(RuntimeError, match="Git is required") as exc:
        tool.status(str(checkout))
    assert "secret-test-token" not in str(exc.value)


async def test_shared_resource_across_workers_and_restart(tmp_path, monkeypatch, respx_mock):
    import json

    import httpx
    from harbor.models.trial.config import TrialConfig

    from flock.harbor.config import BUILTIN
    from flock.harbor.resources import Resources
    from flock.harbor.spec import Spec, resource_secrets

    monkeypatch.setenv("TEAM_REPO_TOKEN", "test-resource-credential")
    options = {"hf_repo": {"repo_id": "example/team", "token_env": "TEAM_REPO_TOKEN"}}
    spec = Spec.model_validate({
        "name": "team", "model": "test-model",
        "task": {"repo": "https://huggingface.co/datasets/example/tasks", "selector": "task"},
        "agents": {"count": 2}, "resources": options,
    })
    assert resource_secrets(spec) == ["TEAM_REPO_TOKEN"]
    info = respx_mock.get("https://huggingface.co/api/datasets/example/team/revision/main")
    info.return_value = httpx.Response(200, json={"id": "example/team", "sha": "a" * 40, "private": True})
    respx_mock.get("https://huggingface.co/api/datasets/example/team/discussions").respond(
        200, json={"discussions": [], "count": 0, "start": 0})
    for worker in ("worker-one", "worker-two", "worker-one"):
        resources = Resources(BUILTIN / "resources", options, run_dir=tmp_path / worker,
                              agents=[], secrets=[], required=["hf_repo"], incremental=True)
        try:
            await resources.start()
            await resources.add(["amber-otter"])
            config = TrialConfig(task={"path": tmp_path / "task"})
            resources.bind("amber-otter", config)
            assert config.agent.env["FLOCK_HF_REPO"] == "example/team"
            assert config.agent.env["FLOCK_HF_REPO_AGENT"] == "amber-otter"
            assert config.agent.env["FLOCK_HF_REPO_TOKEN"] == "test-resource-credential"
            assert "huggingface.co" in config.agent.extra_allowed_hosts
            await resources.remove(["amber-otter"])
        finally:
            assert not await resources.finish()
        result = (tmp_path / worker / "resources/hf_repo/repository.json").read_text()
        assert json.loads(result)["main_sha"] == "a" * 40
        assert "test-resource-credential" not in result
    assert all(call.request.method == "GET" for call in respx_mock.calls)
    info.return_value = httpx.Response(200, json={"id": "example/team", "sha": "a" * 40, "private": False})
    resources = Resources(BUILTIN / "resources", options, run_dir=tmp_path / "public",
                          agents=[], secrets=[])
    try:
        with pytest.raises(ValueError, match="must be private"):
            await resources.start()
    finally:
        await resources.finish()


def test_repository_bundle_runs_without_host_site_packages(tmp_path):
    import json
    import os
    import sys
    import tarfile

    from flock.harbor.config import BUILTIN
    from flock.harbor.tools.bundle import build_tool_bundle

    bundle = build_tool_bundle(BUILTIN / "tools", ["hf_repo.*", "message_board.*"],
                               tmp_path / "tools.tar.gz")
    with tarfile.open(bundle.archive) as archive:
        names = archive.getnames()
        for package in ("huggingface_hub", "pyyaml", "tomli"):
            metadata = [n for n in names if n.lower().startswith(package + "-")
                        and ".dist-info/" in n]
            assert any(n.endswith("/METADATA") for n in metadata)
            assert not any(n.endswith("/WHEEL") for n in metadata)
        archive.extractall(tmp_path / "bundle", filter="data")
    result = subprocess.run(
        [sys.executable, "-S", str(tmp_path / "bundle/bin/flock-tool"), "--list"],
        env={**os.environ, "FLOCK_TOOLS_MANIFEST": str(tmp_path / "bundle/manifest.json")},
        capture_output=True, text=True, check=False, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    tools = json.loads(result.stdout)
    assert len([name for name in tools if name.startswith("hf_repo.")]) == 13
    assert "message_board.post" in tools


@pytest.mark.parametrize('kind', ['fetch', 'push'])
def test_additional_remote_urls_rejected(tool, checkout, kind):
    args = ['git', '-C', str(checkout), 'remote', 'set-url']
    if kind == 'push':
        subprocess.run([*args, '--push', 'origin', tool._url()], check=True)
        args.append('--push')
    subprocess.run([*args, '--add', 'origin', 'https://example.invalid/leak'], check=True)
    with pytest.raises(ValueError, match='does not match'):
        tool.status(str(checkout))


def test_git_errors_include_redacted_stderr(tool, checkout):
    with pytest.raises(RuntimeError) as error:
        tool.commit(str(checkout), ['secret-test-token-missing.py'], 'missing file')
    assert 'pathspec' in str(error.value)
    assert '[REDACTED]' in str(error.value)
    assert 'secret-test-token' not in str(error.value)


def test_infrastructure_credentials_rejected(tmp_path, monkeypatch):
    from flock.harbor.builtin.resources.hf_repo import Resource
    from flock.harbor.spec import Spec, resource_secrets

    spec = Spec.model_validate({
        'name': 'team', 'model': 'test-model',
        'task': {'repo': 'https://huggingface.co/datasets/example/tasks', 'selector': 'task'},
        'resources': {'hf_repo': {'repo_id': 'example/team', 'token_env': 'HF_TOKEN'}},
    })
    with pytest.raises(ValueError, match='separate repository-scoped token'):
        resource_secrets(spec)
    monkeypatch.setenv('HF_TOKEN', 'infrastructure-secret')
    monkeypatch.setenv('HF_REPO_TOKEN', 'infrastructure-secret')
    with pytest.raises(ValueError, match='must differ') as error:
        Resource({'repo_id': 'example/team'}, state_dir=tmp_path, agents=[], secrets=[])
    assert 'infrastructure-secret' not in str(error.value)


@pytest.fixture
def git_transport(tool, checkout, tmp_path, monkeypatch):
    """Replace only Git's HF transport with a real local bare repo; all Git operations run."""
    remote = tmp_path.parent / (tmp_path.name + '-remote.git')
    run = subprocess.run
    run(['git', 'clone', '--bare', str(checkout), str(remote)], check=True, capture_output=True)

    def local_transport(args, **kwargs):
        args = list(args)
        if args[0] == 'git' and any(op in args for op in ('fetch', 'push', 'ls-remote')):
            args = [str(remote) if x in ('origin', tool._url()) else x for x in args]
            args[1:1] = ['-c', 'protocol.file.allow=always']
            # Tool sets protocol.file.allow=never later; replace it only for this transport.
            args = ['protocol.file.allow=always' if x == 'protocol.file.allow=never' else x
                    for x in args]
        return run(args, **kwargs)

    monkeypatch.setattr(subprocess, 'run', local_transport)
    return remote


def test_sync_stale_pr_and_sha_bound_inspection(tool, checkout, git_transport, respx_mock):
    import httpx

    def git(*args):
        return subprocess.check_output(['git', '-C', str(checkout), *args], text=True).strip()

    git('switch', '-c', 'work')
    (checkout / 'work.py').write_text('pass\n')
    head = tool.commit(str(checkout), ['work.py'], 'work')['head_sha']
    git('switch', 'main')
    (checkout / 'main.py').write_text('pass\n')
    main = tool.commit(str(checkout), ['main.py'], 'advance main')['head_sha']
    git('push', str(git_transport), 'main', 'work:refs/pr/1')
    git('switch', 'work')
    tool.sync(str(checkout))
    assert git('branch', '--show-current') == 'work'
    assert git('rev-parse', 'HEAD') == head
    assert git('rev-parse', 'origin/main') == main
    with pytest.raises(ValueError, match='merge origin/main first'):
        tool.open_pr(str(checkout), 'Stale PR', 'Must integrate main')

    metadata = respx_mock.get('https://huggingface.co/api/datasets/example/collaboration/discussions/1')
    metadata.return_value = httpx.Response(200, json={
        'isPullRequest': True, 'title': 'Work', 'status': 'open', 'events': [],
        'diff': 'wrong Hub diff',
    })
    result = tool.inspect_pr(str(checkout), 1)
    assert result['head_sha'] == head
    assert '+pass' in result['diff']['text']
    assert 'work.py' in result['stat']['text']
    assert 'main.py' not in result['diff']['text']
    assert 'diff' not in metadata.calls.last.request.url.params
    assert tool.inspect_pr(str(checkout), 1, paths=['main.py'])['diff']['text'] == ''
    # Advancing the remote cannot make a review of the previously inspected SHA pass.
    git('push', '--force', str(git_transport), 'main:refs/pr/1')
    with pytest.raises(ValueError, match='PR head changed'):
        tool.review_pr(1, head, 'approve', 'Inspected the previous changes')


def test_close_pr_retries_only_429(tool, respx_mock, monkeypatch):
    import httpx

    sleeps = []
    monkeypatch.setattr(tool.time, 'sleep', sleeps.append)
    route = respx_mock.get('https://huggingface.co/api/datasets/example/collaboration/discussions/1')
    route.side_effect = [httpx.Response(429, headers={'Retry-After': '3'}),
                         httpx.Response(200, json={'isPullRequest': True, 'status': 'closed'})]
    assert tool.close_pr(1, 'Abandoned') == {'pr': 1, 'state': 'closed'}
    assert len(sleeps) == 1 and 3 <= sleeps[0] <= 4
    route.return_value = httpx.Response(503)
    route.side_effect = None
    with pytest.raises(RuntimeError, match='HTTP 503'):
        tool.close_pr(1, 'Abandoned')
    assert len(sleeps) == 1


def test_close_pr_posts_reason_with_429_retry(tool, respx_mock, monkeypatch):
    import json

    import httpx

    sleeps = []
    monkeypatch.setattr(tool.time, 'sleep', sleeps.append)
    respx_mock.get('https://huggingface.co/api/datasets/example/collaboration/discussions/1').respond(
        200, json={'isPullRequest': True, 'status': 'open'})
    route = respx_mock.post('https://huggingface.co/api/datasets/example/collaboration/discussions/1/status')
    route.side_effect = [httpx.Response(429, headers={'Retry-After': '1'}),
                         httpx.Response(200, json={'newStatus': {
                             'id': 'event', 'type': 'status-change',
                             'createdAt': '2026-11-13T00:00:00Z', 'data': {'status': 'closed'},
                         }})]
    assert tool.close_pr(1, 'Superseded')['state'] == 'closed'
    assert len(sleeps) == 1
    assert json.loads(route.calls.last.request.content) == {'status': 'closed', 'comment': 'Superseded'}


@pytest.mark.parametrize('state', ['open', 'draft', 'merged'])
def test_pr_listing_filters_and_caps_transport(tool, respx_mock, state):
    records = [{'num': n, 'title': 'Work', 'author': {'name': 'owner'}, 'status': state,
                'createdAt': '2026-11-13T00:00:00Z', 'isPullRequest': True,
                'repo': {'name': 'example/collaboration', 'type': 'dataset'}} for n in range(3)]
    route = respx_mock.get('https://huggingface.co/api/datasets/example/collaboration/discussions')
    route.respond(200, json={'count': 10000, 'start': 0, 'discussions': records})
    result = tool.list_prs(state=state, limit=2)
    assert len(result['pull_requests']) == 2
    assert result['truncated']
    assert route.call_count == 1
    assert route.calls.last.request.url.params['type'] == 'pull_request'
    assert route.calls.last.request.url.params['status'] == ('open' if state == 'open' else 'all')


@pytest.mark.parametrize('retry_after, expected_calls', [('0', 4), ('120', 1)])
def test_rate_limit_retries_are_bounded(tool, respx_mock, monkeypatch, retry_after, expected_calls):
    sleeps = []
    monkeypatch.setattr(tool.time, 'sleep', sleeps.append)
    route = respx_mock.get('https://huggingface.co/api/datasets/example/collaboration/discussions/1')
    route.respond(429, headers={'Retry-After': retry_after})
    with pytest.raises(RuntimeError):
        tool.close_pr(1, 'Abandoned')
    assert route.call_count == expected_calls
    assert len(sleeps) == expected_calls - 1
