"""Namespace boundary checks. Cluster dry run requires the kernel checks; never launches jobs."""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from flock import isolation
from flock.worker.adapters.pi import PiMapper

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def kernel_namespaces():
    executable = shutil.which("unshare")
    result = subprocess.run([executable, "--user", "--map-root-user", "--pid", "--fork", "--mount-proc", "/bin/true"],
                            capture_output=True, timeout=5, check=False) if executable else None
    if result is None or result.returncode:
        if os.environ.get("FLOCK_REQUIRE_ISOLATION") == "1":
            pytest.fail("run4 requires working user/PID/mount namespaces in this image")
        pytest.skip("this host denies unprivileged namespaces; required by the cluster dry run")


def test_bot_and_pi_invocations_have_separate_roots_and_no_unsafe_fallback(tmp_path):
    tree, pin, home = tmp_path / "tree", tmp_path / "pins/cedar", tmp_path / "pi/cedar"
    bot = isolation.command([sys.executable, "-c", "pass"], tmp_path / "bot-root", cwd=tree,
                            readonly=[tree], python_only=True)
    assert all(flag in bot for flag in ["--user", "--pid", "--mount-proc", "--net", "--kill-child=SIGKILL"])
    config = json.loads(bot[bot.index(str(Path(isolation.__file__).resolve())) + 1])
    assert config["writable"] == [] and config["readonly_root"]
    pi = isolation.command([sys.executable, "-c", "pass"], tmp_path / "pi-root", cwd=tree,
                           readonly=[pin], writable=[tree, home], network=True)
    assert "--net" not in pi and "--mount-proc" in pi
    config = json.loads(pi[pi.index(str(Path(isolation.__file__).resolve())) + 1])
    assert config["writable"] == [str(tree), str(home)]
    assert str(pin.parent) not in config["readonly"]


def test_two_level_symlinked_interpreter_keeps_every_hop_inside_the_root(tmp_path, monkeypatch):
    """A uv venv: bin/python -> .../cpython-3.12-linux (a link) -> cpython-3.12.14-linux. The
    chroot binds the real install and recreates each hop, so the venv's path resolves."""
    pythons = tmp_path / "uv/python"
    real = pythons / "cpython-3.12.14-linux-x86_64-gnu"
    (real / "bin").mkdir(parents=True)
    (real / "bin/python3.12").write_text("")
    (pythons / "cpython-3.12-linux-x86_64-gnu").symlink_to(real.name)  # relative hop
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").symlink_to(pythons / "cpython-3.12-linux-x86_64-gnu/bin/python3.12")
    (venv / "bin/python3").symlink_to("python")
    monkeypatch.setattr(isolation.sys, "executable", str(venv / "bin/python3"))
    monkeypatch.setattr(isolation.sys, "prefix", str(venv))
    monkeypatch.setattr(isolation.sys, "base_prefix", str(real))
    monkeypatch.setattr(isolation.shutil, "which", lambda name: "/usr/bin/unshare")
    argv = [str(venv / "bin/python3"), "-I", "-S", "-c", "pass"]
    config = isolation.plan(argv, cwd=tmp_path, python_only=True)
    hops = dict(config["symlinks"])
    assert hops[str(venv / "bin/python3")] == "python"
    assert hops[str(venv / "bin/python")] == str(pythons / "cpython-3.12-linux-x86_64-gnu/bin/python3.12")
    assert hops[str(pythons / "cpython-3.12-linux-x86_64-gnu")] == real.name
    assert str(real) in config["readonly"] and str(venv) in config["readonly"]

    # Replaying the plan in a scratch root (no namespace) resolves the original spelling.
    scratch = tmp_path / "scratch"
    for path in config["readonly"]:
        if path.startswith(str(tmp_path)):
            (scratch / Path(path).relative_to("/")).mkdir(parents=True, exist_ok=True)
    (scratch / Path(real, "bin").relative_to("/")).mkdir(parents=True, exist_ok=True)
    (scratch / Path(real, "bin/python3.12").relative_to("/")).write_text("")
    for link, target in config["symlinks"]:
        inside = scratch / Path(link).relative_to("/")
        inside.parent.mkdir(parents=True, exist_ok=True)
        inside.symlink_to(scratch / Path(target).relative_to("/") if os.path.isabs(target) else target)
    assert (scratch / Path(venv, "bin/python3").relative_to("/")).resolve() == \
        (scratch / Path(real, "bin/python3.12").relative_to("/")).resolve()


def test_unmounted_symlink_target_fails_with_the_path_named(tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere/python3.12"
    elsewhere.parent.mkdir()
    elsewhere.write_text("")
    link = tmp_path / "bin/python"
    link.parent.mkdir()
    link.symlink_to(elsewhere)
    monkeypatch.setattr(isolation.sys, "executable", str(link))
    monkeypatch.setattr(isolation.sys, "prefix", str(link.parent))
    monkeypatch.setattr(isolation.sys, "base_prefix", str(link.parent))
    with pytest.raises(RuntimeError, match=str(elsewhere)):
        isolation.plan([str(link), "-c", "pass"], cwd=tmp_path, python_only=True)


def test_failed_isolation_never_executes_bot_without_it(tmp_path, monkeypatch):
    from . import test_chatbot4  # world import path

    sandbox = sys.modules["sandbox"]
    assert test_chatbot4.Chatbot4World
    fake = tmp_path / "unshare"
    fake.write_text("#!/bin/sh\nexit 125\n")
    fake.chmod(0o700)
    monkeypatch.setattr(isolation.shutil, "which", lambda _: str(fake))
    result = sandbox.run(ROOT / "worlds/chatbot4/seed", ["Hello"])
    assert not result["ok"] and result["status"] == "exit 125" and not result["replies"]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux prctl")
def test_dumpability_blocks_a_same_uid_child_from_supervisor_credentials():
    script = r'''
import subprocess, sys
from flock.isolation import protect_supervisor
protect_supervisor()
child = r"""
import os
for name in ('environ', 'mem', 'root'):
    path = '/proc/%d/%s' % (os.getppid(), name)
    try:
        if name == 'root': os.readlink(path)
        else:
            with open(path, 'rb'): pass
    except PermissionError: continue
    raise SystemExit('supervisor proc access was allowed')
"""
result = subprocess.run([sys.executable, '-c', child], env={}, capture_output=True)
assert result.returncode == 0, result.stdout + result.stderr
'''
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, timeout=5)


@pytest.mark.parametrize("message", ["maximum context length is 131072 tokens; you requested 134290",
                                      "context length 150300 exceeds 131072", "invalid request 50201",
                                      "you requested 429 tokens, only 200 available"])
def test_context_length_numbers_do_not_trigger_transport_retries(message):
    mapper = PiMapper()
    mapper.on_record({"type": "agent_start"})
    mapper.on_record({"type": "message_end", "message": {"role": "assistant", "stopReason": "error", "errorMessage": message}})
    [event] = mapper.on_record({"type": "agent_settled"})
    assert event.status == "error" and event.data.get("reason") != "model_transport"


@pytest.mark.parametrize("message", ["429 Too Many Requests", "HTTP 503: unavailable", "HTTP 502: bad gateway"])
def test_http_errors_remain_retryable(message):
    mapper = PiMapper()
    mapper.on_record({"type": "agent_start"})
    mapper.on_record({"type": "message_end", "message": {"role": "assistant", "stopReason": "error", "errorMessage": message}})
    [event] = mapper.on_record({"type": "agent_settled"})
    assert event.data["reason"] == "model_transport"


def test_namespace_hides_supervisor_and_sibling_files_and_keeps_live_pin_updates(tmp_path, kernel_namespaces):
    workspace, home, pin = tmp_path / "agents/cedar", tmp_path / "pi/cedar", tmp_path / "pins/cedar"
    for path in (workspace, home, pin, tmp_path / "pins/birch"):
        path.mkdir(parents=True)
    (pin / "pin.json").write_text("old")
    sibling = tmp_path / "pins/birch/pin.json"
    sibling.write_text("other-agent-key")
    secret = tmp_path / "world-secret"
    secret.write_text("world-token")
    script = f'''
import os, pathlib, sys
assert os.getpgrp() == os.getpid(), 'sandbox inherited an outside process group'
for hidden in [{str(sibling)!r}, {str(secret)!r}, '/proc/{os.getpid()}/environ']:
    assert not pathlib.Path(hidden).exists(), hidden
try: os.kill({os.getpid()}, 0)
except ProcessLookupError: pass
else: raise AssertionError('supervisor PID is visible')
status = pathlib.Path('/proc/self/status').read_text()
for key in ['CapEff', 'CapPrm', 'CapBnd', 'CapAmb']:
    assert int(status.split(key+':')[1].split()[0], 16) == 0
assert status.split('NoNewPrivs:')[1].split()[0] == '1'
try: pathlib.Path({str(pin / 'pin.json')!r}).write_text('forged')
except OSError: pass
else: raise AssertionError('pin is writable')
pathlib.Path('own-file').write_text('allowed')
print('ready', flush=True)
sys.stdin.readline()
print(pathlib.Path({str(pin / 'pin.json')!r}).read_text(), flush=True)
'''
    argv = isolation.command([sys.executable, "-I", "-c", script], tmp_path / "root", cwd=workspace,
                             readonly=[pin], writable=[workspace, home], network=True, python_only=True)
    proc = subprocess.Popen(argv, env={}, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        # Bounded readiness wait, including when mount/capability setup fails.
        import selectors
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdout, selectors.EVENT_READ)
            assert selector.select(5), "namespace probe did not answer"
        line = proc.stdout.readline()
        assert line == "ready\n", proc.communicate(timeout=5)
        replacement = pin / "new.json"
        replacement.write_text("updated")
        replacement.replace(pin / "pin.json")
        out, err = proc.communicate("go\n", timeout=5)
        assert proc.returncode == 0 and out == "updated\n", err
        assert (workspace / "own-file").read_text() == "allowed"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_bot_network_and_files_are_read_only(tmp_path, kernel_namespaces):
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "code.py").write_text("pass")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        script = f'''
import pathlib, socket
for path in ['new-file', '/tmp/new-file', {str(tree / 'code.py')!r}]:
    try: pathlib.Path(path).write_text('bad')
    except OSError: pass
    else: raise AssertionError('bot filesystem is writable')
with socket.socket() as sock:
    assert sock.connect_ex(('127.0.0.1', {port})) != 0
for line in open('/proc/self/mountinfo'):
    point, opts = line.split()[4], line.split()[5].split(',')
    if point == '/proc' or point.startswith('/dev/'):
        continue
    assert 'ro' in opts, 'writable mount in the bot sandbox: ' + point
print('isolated')
'''
        argv = isolation.command([sys.executable, "-I", "-c", script], tmp_path / "root", cwd=tree,
                                 readonly=[tree], python_only=True)
        result = subprocess.run(argv, env={}, capture_output=True, timeout=5, text=True, check=False)
        assert result.returncode == 0 and result.stdout == "isolated\n", result.stderr


def test_real_bot_runs_inside_namespaces(tmp_path, kernel_namespaces):
    from . import test_chatbot4

    assert test_chatbot4.Chatbot4World
    result = sys.modules["sandbox"].run(ROOT / "worlds/chatbot4/seed", ["Hello", "Bye"])
    assert result["ok"] and result["replies"] == ["Hello!", "Hello!"], result


async def test_pi_start_uses_namespace_wrapper_and_explicit_environment(tmp_path, monkeypatch):
    from flock.models import TypeSpec
    from flock.worker.adapters import pi

    monkeypatch.setenv("FLOCK_PROCESS_ISOLATION", "1")
    monkeypatch.setenv("HF_TOKEN", "worker-secret")
    workspace, home, pin = tmp_path / "cedar", tmp_path / "home", tmp_path / "pins/cedar/pin.json"
    workspace.mkdir()
    adapter = pi.PiAdapter(workspace, agent="cedar", home=home, pi_bin="/usr/bin/false",
                           model_config=pi.PiModelConfig("http://fake/v1", "MODEL_KEY", "m", "fake"))
    calls = []
    async def spawn(*args, **kwargs):
        calls.append((args, kwargs))
        raise RuntimeError("stop before exec")
    monkeypatch.setattr(pi.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(isolation, "SANDBOX_TMP", "/nonexistent-sandbox-tmp")  # tmp_path is in /tmp
    try:
        with pytest.raises(RuntimeError, match="stop before exec"):
            await adapter.start(workspace, None, TypeSpec(harness="pi", model="m"), env={
                "MODEL_KEY": "provider-key", "HF_TOKEN": "explicit-secret", "FLOCK_TOOLS_PIN": str(pin),
                "FLOCK_WORLD_TOKEN": "world-secret", "UNRELATED_SECRET": "secret"})
        args, kwargs = calls[0]
        assert "--mount-proc" in args and "--pid" in args
        assert "--net" not in args
        assert kwargs["start_new_session"]
        assert kwargs["env"]["MODEL_KEY"] == "provider-key"
        assert kwargs["env"]["FLOCK_TOOLS_PIN"] == str(pin)
        assert not ({"HF_TOKEN", "FLOCK_TOKEN", "FLOCK_WORLD_TOKEN", "UNRELATED_SECRET"} & kwargs["env"].keys())
        # HOME, TMPDIR and caches live on the agent's own disk-backed dir, not the tmpfs root.
        env = kwargs["env"]
        assert env["HOME"] == str(home / "home") and env["TMPDIR"] == str(home / "tmp")
        for var in ("XDG_CACHE_HOME", "UV_CACHE_DIR", "npm_config_cache", "PIP_CACHE_DIR"):
            assert env[var].startswith(str(home / "home") + "/")
        for d in (home / "home", home / "tmp"):
            assert d.is_dir() and (d.stat().st_mode & 0o777) == 0o700
        config = json.loads(args[args.index(str(Path(isolation.__file__).resolve())) + 1])
        assert str(home.resolve()) in config["writable"] and config["private_dirs"] == []
        # the sandbox's /tmp is the agent's disk-backed TMPDIR; the tmpfs root is capped
        assert config["tmp"] == str((home / "tmp").resolve()) and config["tmpfs_size"] == "256m"
    finally:
        await adapter.stop()


async def test_real_pi_starts_and_resumes_inside_namespaces(tmp_path, monkeypatch, kernel_namespaces):
    from . import fake_openai
    from .test_worker_pi import PI_BIN, make_adapter, pi_spec, until_turn_ended

    if not PI_BIN:
        pytest.skip("Pi CLI is required for isolated integration")
    monkeypatch.setenv("FLOCK_PROCESS_ISOLATION", "1")
    server = fake_openai.serve(0, str(tmp_path / "model.jsonl"))
    adapter, workspace = make_adapter(tmp_path, server, monkeypatch)
    try:
        sid = await adapter.start(workspace, None, pi_spec(), env={"FLOCK_TEST_PI_KEY": "fake-key"})
        await adapter.send("bash: echo namespace-check")
        events = await until_turn_ended(adapter)
        assert events[-1].status == "completed"
        await adapter.stop()
        assert await adapter.start(workspace, sid, pi_spec(), env={"FLOCK_TEST_PI_KEY": "fake-key"}) == sid
        await adapter.send("bash: echo namespace-resume")
        events = await until_turn_ended(adapter)
        assert events[-1].status == "completed"
    finally:
        await adapter.stop()
        server.shutdown()


def test_namespace_wrapper_ignores_direct_spot_usr1(tmp_path, monkeypatch):
    fake = tmp_path / "unshare"
    fake.write_text('#!/bin/sh\nkill -USR1 "$$"\nprintf "survived\\n"\n')
    fake.chmod(0o700)
    monkeypatch.setattr(isolation.shutil, "which", lambda _: str(fake))
    argv = isolation.command([sys.executable, "-c", "pass"], tmp_path / "root", cwd=tmp_path,
                             python_only=True)
    result = subprocess.run(argv, env={}, capture_output=True, text=True, timeout=5, check=False)
    assert result.returncode == 0 and result.stdout == "survived\n", result.stderr


def test_readonly_bind_is_a_recursive_bind_then_a_remount_with_the_locked_flags(tmp_path, monkeypatch):
    """A user namespace rejects a non-recursive bind of a tree with locked submounts
    (EINVAL) and a read-only remount that drops a locked flag (EPERM)."""
    calls = []

    class Libc:
        class mount:
            argtypes = None

            def __new__(cls, source, target, kind, flags, data):
                calls.append((source, target, kind, flags, data))
                return 0

    source = tmp_path / "runtime"
    source.mkdir()
    root = tmp_path / "root"
    monkeypatch.setattr(isolation, "_libc", lambda: Libc)
    recursive = []
    monkeypatch.setattr(isolation, "_readonly_recursive", lambda target: recursive.append(target) or True)
    locked = os.ST_NOSUID | os.ST_NODEV | os.ST_NOEXEC | os.ST_RELATIME | os.ST_NODIRATIME
    monkeypatch.setattr(isolation.os, "statvfs", lambda p: type("V", (), {"f_flag": locked})())
    isolation._bind(root, str(source), readonly=True)
    target = os.fsencode(root / source.relative_to("/"))
    assert calls[0] == (os.fsencode(source), target, None, isolation.MS_BIND | isolation.MS_REC, None)
    expected = (isolation.MS_BIND | isolation.MS_REMOUNT | isolation.MS_RDONLY | isolation.MS_NOSUID
                | isolation.MS_NODEV | isolation.MS_NOEXEC | isolation.MS_RELATIME | isolation.MS_NODIRATIME)
    assert calls[1] == (None, target, None, expected, None) and len(calls) == 2
    assert recursive == [root / source.relative_to("/")]  # then mount_setattr on submounts

    # strictatime source (no atime flag): the remount must say so, or the kernel resets it
    calls.clear()
    monkeypatch.setattr(isolation.os, "statvfs", lambda p: type("V", (), {"f_flag": os.ST_NOSUID})())
    isolation._bind(root, str(source), readonly=True)
    assert calls[1][3] & isolation.MS_STRICTATIME and not calls[1][3] & isolation.MS_RELATIME

    # writable binds are recursive too, with no remount
    calls.clear()
    isolation._bind(root, str(source), readonly=False)
    assert [c[3] for c in calls] == [isolation.MS_BIND | isolation.MS_REC]


def test_mount_errors_name_the_target_and_flags(tmp_path, monkeypatch):
    import ctypes

    class Libc:
        class mount:
            argtypes = None

            def __new__(cls, *args):
                return -1

    monkeypatch.setattr(isolation, "_libc", lambda: Libc)
    monkeypatch.setattr(ctypes, "get_errno", lambda: 22)
    with pytest.raises(isolation.IsolationMountError) as e:
        isolation._mount("/usr", tmp_path / "root/usr", flags=isolation.MS_BIND | isolation.MS_REC)
    assert e.value.errno == 22
    assert "target=" + str(tmp_path / "root/usr") in str(e.value) and "flags=0x5000" in str(e.value)


ZONEINFO_MOUNTINFO = r"""22 1 0:21 / / rw,relatime - overlay overlay rw
40 22 0:5 / /tmp/r/root rw,nosuid,nodev - tmpfs tmpfs rw,size=262144k
41 40 0:21 /usr /tmp/r/root/usr ro,nosuid,nodev,relatime - overlay overlay rw
42 41 259:1 /usr/share/zoneinfo/Etc/UTC /tmp/r/root/usr/share/zoneinfo/Etc/UTC ro,relatime - ext4 /dev/nvme0n1p1 rw
43 40 0:30 /work/cedar /tmp/r/root/work/cedar rw,relatime - ext4 /dev/x rw
44 40 0:5 /null /tmp/r/root/dev/null rw,nosuid - devtmpfs udev rw
45 40 0:40 / /tmp/r/root/proc rw,nosuid,nodev,noexec - proc proc rw
46 40 0:41 /my\040dir /tmp/r/root/opt/my\040dir ro - ext4 /dev/x rw
"""


def test_mountinfo_check_allows_planned_mounts_and_their_ro_children():
    mounts = isolation.parse_mountinfo(ZONEINFO_MOUNTINFO)
    assert ("/tmp/r/root/opt/my dir", {"ro"}) in mounts
    root = "/tmp/r/root"
    ro, rw = [f"{root}/usr", f"{root}/opt/my dir"], [f"{root}/work/cedar"]
    extra = [f"{root}/proc", f"{root}/dev/null"]
    isolation.check_mounts(mounts, root, ro, rw, extra)  # the zoneinfo child is ro: fine

    writable_child = ZONEINFO_MOUNTINFO.replace(
        "/tmp/r/root/usr/share/zoneinfo/Etc/UTC ro,relatime", "/tmp/r/root/usr/share/zoneinfo/Etc/UTC rw,relatime")
    with pytest.raises(isolation.IsolationMountError, match="zoneinfo/Etc/UTC is mounted writable"):
        isolation.check_mounts(isolation.parse_mountinfo(writable_child), root, ro, rw, extra)
    stray = ZONEINFO_MOUNTINFO + "47 40 0:42 / /tmp/r/root/home/secrets rw - ext4 /dev/x rw\n"
    with pytest.raises(isolation.IsolationMountError, match="unexpected mount /tmp/r/root/home/secrets"):
        isolation.check_mounts(isolation.parse_mountinfo(stray), root, ro, rw, extra)


def test_bind_never_touches_an_existing_target_on_a_readonly_bind(tmp_path, monkeypatch):
    """/etc/localtime resolves into /usr, already bound read-only: its mount point exists
    and touch() (utime) would fail with EROFS."""
    monkeypatch.setattr(isolation, "_mount", lambda *a, **k: None)
    monkeypatch.setattr(isolation, "_readonly_recursive", lambda target: True)
    monkeypatch.setattr(isolation, "readonly_remount_flags", lambda target: 0)
    source = tmp_path / "zoneinfo/UTC"
    source.parent.mkdir()
    source.write_text("tz")
    root = tmp_path / "root"
    target = root / source.relative_to("/")
    target.parent.mkdir(parents=True)
    target.write_text("tz")

    def refuse(*a, **k):
        raise PermissionError(30, "Read-only file system")

    monkeypatch.setattr(Path, "touch", refuse)
    monkeypatch.setattr(Path, "mkdir", lambda self, *a, **k: refuse()
                        if self.exists() and not k.get("exist_ok") else None)
    isolation._bind(root, str(source), readonly=True)  # no touch, no mkdir of existing paths


def test_setup_errors_name_the_step_and_path(tmp_path, monkeypatch):
    monkeypatch.setattr(isolation, "_mount", lambda *a, **k: None)
    source = tmp_path / "data"
    source.mkdir()
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o555)
    try:
        with pytest.raises(isolation.IsolationSetupError) as e:
            isolation._bind(locked / "root", str(source), readonly=False)
        assert "mount point creation failed" in str(e.value) and str(locked / "root") in str(e.value)
        with pytest.raises(isolation.IsolationSetupError, match="required path is missing: path="):
            isolation._bind(tmp_path / "root", str(tmp_path / "missing"), readonly=False)

        # the entry point prints the step and path, never argv
        config = {"root": str(locked / "root"), "cwd": "/", "readonly": [], "writable": [],
                  "readonly_root": True, "private_dirs": [], "symlinks": []}
        result = subprocess.run([sys.executable, "-I", isolation.__file__, json.dumps(config),
                                 "/bin/true", "argv-secret"], capture_output=True, text=True,
                                timeout=10, check=False)
        assert result.returncode == 125
        assert "root creation failed" in result.stderr and str(locked / "root") in result.stderr
        assert "argv-secret" not in result.stderr
    finally:
        locked.chmod(0o755)


def test_tmpfs_root_has_a_size_cap_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.delenv(isolation.TMPFS_SIZE_ENV, raising=False)
    assert isolation.tmpfs_size() == isolation.TMPFS_SIZE_DEFAULT == "256m"
    for value, expected in (("512m", "512m"), ("1G", "1g"), ("1048576", "1048576"),
                            ("0", "256m"), ("unlimited", "256m"), ("50%", "256m"), ("-1m", "256m")):
        assert isolation.tmpfs_size({isolation.TMPFS_SIZE_ENV: value}) == expected, value
    monkeypatch.setenv(isolation.TMPFS_SIZE_ENV, "128m")
    argv = isolation.command([sys.executable, "-c", "pass"], tmp_path / "root", cwd=tmp_path,
                             python_only=True)
    assert json.loads(argv[argv.index(str(Path(isolation.__file__).resolve())) + 1])["tmpfs_size"] == "128m"


def test_sandbox_tmp_is_the_callers_disk_dir_unless_it_would_hide_a_planned_path(tmp_path, monkeypatch):
    work, disk_tmp = tmp_path / "work", tmp_path / "home/tmp"
    for d in (work, disk_tmp):
        d.mkdir(parents=True)
    # tmp_path itself is under /tmp here: a /tmp bind would hide the workdir, so none
    if os.path.realpath(tmp_path).startswith("/tmp/"):
        plan = isolation.plan([sys.executable], cwd=work, writable=[work], python_only=True, tmp=disk_tmp)
        assert plan["tmp"] is None
    # with nothing planned under the sandbox's /tmp, it is the disk dir
    monkeypatch.setattr(isolation, "SANDBOX_TMP", "/nonexistent-sandbox-tmp")
    plan = isolation.plan([sys.executable], cwd=work, writable=[work], python_only=True, tmp=disk_tmp)
    assert plan["tmp"] == os.path.realpath(disk_tmp)
    assert isolation.plan([sys.executable], cwd=work, writable=[work], python_only=True)["tmp"] is None


def test_enter_mounts_a_capped_tmpfs_and_binds_tmp_from_disk(tmp_path, monkeypatch):
    """The setup order without namespaces: a size-capped tmpfs root, the planned binds, then
    the caller's disk dir over /tmp (writable, and expected by the fail-closed mount check)."""
    work, disk_tmp, root = tmp_path / "work", tmp_path / "home/tmp", tmp_path / "root"
    for d in (work, disk_tmp):
        d.mkdir(parents=True)
    mounts, checked = [], []
    monkeypatch.setattr(isolation, "_mount", lambda *a, **k: mounts.append((a, k)))
    monkeypatch.setattr(isolation, "check_mounts", lambda *a: checked.append(a))
    monkeypatch.setattr(isolation.os, "getpgrp", os.getpid)

    def chroot(path):
        raise RuntimeError("stop before chroot")

    monkeypatch.setattr(isolation.os, "chroot", chroot)
    config = {"root": str(root), "cwd": str(work), "readonly": [], "writable": [str(work)],
              "readonly_root": False, "private_dirs": [], "symlinks": [], "tmp": str(disk_tmp),
              "tmpfs_size": "64m"}
    with pytest.raises(RuntimeError, match="stop before chroot"):
        isolation._enter(config, ["/bin/true"])
    assert mounts[0] == (("tmpfs", root, "tmpfs", isolation.MS_NOSUID | isolation.MS_NODEV,
                          "mode=755,size=64m"), {})
    binds = [(a[0], a[1]) for a, k in mounts if k.get("flags") == isolation.MS_BIND | isolation.MS_REC]
    assert (str(disk_tmp), root / "tmp") in binds
    assert binds.index((str(disk_tmp), root / "tmp")) > binds.index((str(work), root / work.relative_to("/")))
    writable = checked[0][3]
    assert os.path.realpath(root / "tmp") in writable

    # no tmp dir and an older config without a size: /tmp stays on the (default-capped) tmpfs
    mounts.clear()
    checked.clear()
    del config["tmpfs_size"]
    root = tmp_path / "root2"  # (no real tmpfs here: a fresh directory)
    config.update(tmp=None, root=str(root))
    with pytest.raises(RuntimeError, match="stop before chroot"):
        isolation._enter(config, ["/bin/true"])
    assert mounts[0][0][4] == "mode=755,size=256m"
    assert not any(a[1] == root / "tmp" for a, _ in mounts)
    assert os.path.realpath(root / "tmp") not in checked[0][3]

    config.update(tmpfs_size="0,nr_inodes=0", root=str(tmp_path / "root3"))  # never an injection
    with pytest.raises(isolation.IsolationSetupError, match="tmpfs size is invalid"):
        isolation._enter(config, ["/bin/true"])


def test_mountinfo_check_accepts_the_tmp_bind_only_when_planned():
    root = "/run/flock-root"
    info = (f"30 22 0:50 / {root} rw - tmpfs tmpfs rw\n"
            f"31 30 259:1 /state/pi/a/tmp {root}/tmp rw,relatime - ext4 /dev/x rw\n")
    mounts = isolation.parse_mountinfo(info)
    isolation.check_mounts(mounts, root, [], [f"{root}/tmp"])
    with pytest.raises(isolation.IsolationMountError, match="unexpected mount"):
        isolation.check_mounts(mounts, root, [], [])


def _outside_tmp_dir() -> Path | None:
    import tempfile

    for base in (ROOT, Path.home()):
        real = os.path.realpath(base)
        if real == "/tmp" or real.startswith("/tmp/") or not os.access(base, os.W_OK):
            continue
        return Path(tempfile.mkdtemp(prefix=".isolation-test-", dir=base))
    return None


def test_sandbox_tmp_writes_land_on_disk_and_the_root_is_capped(tmp_path, monkeypatch, kernel_namespaces):
    """In namespaces (the cluster dry run): a literal /tmp write lands in the caller's disk dir,
    and the tmpfs root reports the configured size cap."""
    base = _outside_tmp_dir()
    if base is None:
        pytest.skip("needs a writable directory outside /tmp (a /tmp bind would hide /tmp paths)")
    try:
        work, disk_tmp = base / "work", base / "home/tmp"
        for d in (work, disk_tmp):
            d.mkdir(parents=True)
        monkeypatch.setenv(isolation.TMPFS_SIZE_ENV, "32m")
        script = '''
import os, pathlib
pathlib.Path("/tmp/probe").write_text("on-disk")
st = os.statvfs("/")
print(st.f_blocks * st.f_frsize)
'''
        argv = isolation.command([sys.executable, "-I", "-c", script], base / "root", cwd=work,
                                 writable=[work], network=True, python_only=True, tmp=disk_tmp)
        result = subprocess.run(argv, env={}, capture_output=True, timeout=10, text=True, check=False)
        assert result.returncode == 0, result.stderr
        assert (disk_tmp / "probe").read_text() == "on-disk"
        assert int(result.stdout) == 32 * 1024 * 1024
    finally:
        shutil.rmtree(base, ignore_errors=True)
