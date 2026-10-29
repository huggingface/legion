"""CPU-only orchestration/recovery tests. No Slurm jobs or model servers are launched."""
import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "experiments/agent-escape/b300"


@pytest.fixture
def serving():
    spec = importlib.util.spec_from_file_location("qwen_serve", DIRECTORY / "qwen_serve.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def config(tmp_path):
    (tmp_path / "logs").mkdir()
    for host in ("node-a", "node-b"):
        (tmp_path / "logs" / f"{host}-devices.json").write_text(
            json.dumps([f"GPU-{i}" for i in range(8)]))
    return {"context": 32768, "sequences": 128, "batch_tokens": 8192,
            "hosts": ["node-a", "node-b"], "run": str(tmp_path), "job_id": "123",
            "image": "/image.sqsh", "local": "/scratch/test/job-123",
            "model_local": "/scratch/test/model"}


class Process:
    def __init__(self, result=None):
        self.result = result
        self.returncode = result

    def poll(self):
        return self.result

    def wait(self, timeout=None):
        self.returncode = 0
        return 0


class Finished(Exception):
    pass


def test_shell_syntax():
    subprocess.run(["bash", "-n", str(DIRECTORY / "serve-qwen38-27b.slurm")], check=True)


@pytest.mark.parametrize("nodes", [1, 2, 4])
@pytest.mark.parametrize("gpus", [4, 8])
def test_routing_registers_every_independent_gpu(serving, config, nodes, gpus):
    config["gpus"] = gpus
    config["hosts"] = [f"node-{i}" for i in range(nodes)]
    backends = serving.instances(config["hosts"], gpus)
    assert len(backends) == gpus * nodes
    assert len({item["url"] for item in backends}) == len(backends)
    for host in config["hosts"]:
        assert [b["port"] for b in backends if b["hostname"] == host] == list(range(8100, 8100 + gpus))
    router = serving.router_command(config)
    assert all(item["url"] in router for item in backends)
    assert router[router.index("--policy") + 1] == "cache_aware"
    assert router[router.index("--intra-node-data-parallel-size") + 1] == "1"
    assert "--disable-retries" in router


def test_gpu_assignment_and_key_only_in_environment(serving, config, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c,GPU-d,GPU-e,GPU-f,GPU-g,GPU-h")
    devices = serving.gpu_devices()
    replicas = [serving.Replica(config, "node-a", i, d, "private-token")
                for i, d in enumerate(devices)]
    assert [r.env["CUDA_VISIBLE_DEVICES"] for r in replicas] == devices
    assert len({r.log for r in replicas}) == 8
    for replica in replicas:
        assert replica.env["VLLM_API_KEY"] == "private-token"
        assert "private-token" not in " ".join(replica.command + serving.router_command(config))
        for option in ("--tensor-parallel-size", "--data-parallel-size"):
            assert replica.command[replica.command.index(option) + 1] == "1"


@pytest.mark.parametrize("devices", ["0,1", "0,1,2,3,4,5,6,6", ""])
def test_invalid_allocated_gpus_rejected(serving, monkeypatch, devices):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", devices)
    with pytest.raises(ValueError, match="8 distinct"):
        serving.gpu_devices()


def test_replica_restart_does_not_touch_siblings(serving, config, monkeypatch):
    created, stopped = [], []

    def spawn(*args, **kwargs):
        proc = Process()
        created.append(proc)
        return proc

    monkeypatch.setattr(serving, "spawn", spawn)
    monkeypatch.setattr(serving, "stop_processes", lambda ps: stopped.extend(ps))
    replicas = [serving.Replica(config, "node-a", i, str(i), "key") for i in range(8)]
    for replica in replicas:
        replica.tick(0, False)
    failed = created[3]
    failed.result = 1
    for replica in replicas:
        replica.tick(1, True)
    assert stopped == [failed]
    assert len(created) == 8
    replicas[3].tick(10, False)
    assert len(created) == 8
    replicas[3].tick(11, False)
    assert len(created) == 9
    assert all(r.proc is created[i] for i, r in enumerate(replicas) if i != 3)
    replicas[3].tick(12, True)
    replicas[3].tick(612, True)
    assert replicas[3].failures == 0


def test_retry_cap_and_hung_instance(serving, config, monkeypatch):
    monkeypatch.setattr(serving, "spawn", lambda *a, **kw: Process())
    stopped = []
    monkeypatch.setattr(serving, "stop_processes", lambda ps: stopped.extend(ps))
    replica = serving.Replica(config, "node-a", 0, "0", "key")
    replica.tick(0, False)
    replica.tick(1, True)
    for now in (2, 3, 4):
        replica.tick(now, False)
    assert len(stopped) == 1  # three missed checks kill a hung previously healthy process
    for _ in range(4):
        replica.tick(replica.next_start, False)
        replica.proc.result = 1
        replica.tick(replica.next_start + 1, False)
    assert replica.failures == 5
    replica.tick(10000, False)
    assert replica.proc is None and len(stopped) == 5


def test_replica_startup_timeout(serving, config, monkeypatch):
    monkeypatch.setattr(serving, "spawn", lambda *a, **kw: Process())
    monkeypatch.setattr(serving, "stop_processes", lambda ps: None)
    replica = serving.Replica(config, "node-a", 0, "0", "key")
    replica.tick(0, False)
    replica.tick(serving.STARTUP, False)
    assert replica.proc is None and replica.failures == 1


def test_spawn_error_uses_the_same_retry_budget(serving, config, monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("temporary process limit")

    monkeypatch.setattr(serving, "spawn", unavailable)
    replica = serving.Replica(config, "node-a", 0, "0", "key")
    replica.tick(0, False)
    assert replica.failures == 1 and replica.next_start == 10
    monkeypatch.setattr(serving, "spawn", lambda *a, **kw: Process())
    replica.tick(10, False)
    assert replica.proc is not None


def monitor_setup(serving, monkeypatch):
    monkeypatch.setattr(serving, "http_status", lambda *a, **kw: 200)
    monkeypatch.setattr(serving, "verify_auth", lambda *a: None)
    return [Process(), Process()], Process()


def test_readiness_waits_for_all_then_keeps_endpoint_during_recovery(
    serving, config, monkeypatch
):
    children, router = monitor_setup(serving, monkeypatch)
    states = iter([[True] * 15 + [False], [True] * 16,
                   [False] + [True] * 15, [True] * 16])
    monkeypatch.setattr(serving, "health", lambda urls: next(states))
    cycles = []

    def sleep(_):
        cycles.append((Path(config["run"]) / "endpoint.json").exists())
        if len(cycles) == 4:
            raise Finished

    monkeypatch.setattr(serving.time, "sleep", sleep)
    with pytest.raises(Finished):
        serving.monitor(config, children, router, "key", float("inf"))
    assert cycles == [False, True, True, True]
    endpoint = json.loads((Path(config["run"]) / "endpoint.json").read_text())
    assert len(endpoint["instances"]) == 16
    assert endpoint["base_url"] == "http://node-a:8000/v1"
    assert endpoint["instances"][0]["gpu_device"] == "GPU-0"
    assert "key" not in endpoint


def test_startup_failure_never_publishes(serving, config, monkeypatch):
    children, router = monitor_setup(serving, monkeypatch)
    monkeypatch.setattr(serving, "health", lambda urls: [False] * len(urls))
    with pytest.raises(RuntimeError, match="Startup exceeded"):
        serving.monitor(config, children, router, "key", 0)
    assert not (Path(config["run"]) / "endpoint.json").exists()


def test_all_instances_unhealthy_exits_after_grace(serving, config, monkeypatch):
    children, router = monitor_setup(serving, monkeypatch)
    times = iter([1, 2, 602])
    states = iter([[True] * 16, [False] * 16, [False] * 16])
    monkeypatch.setattr(serving.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(serving.time, "sleep", lambda _: None)
    monkeypatch.setattr(serving, "health", lambda urls: next(states))
    with pytest.raises(RuntimeError, match="No healthy instances"):
        serving.monitor(config, children, router, "key", 3600)


def test_router_death_is_fatal(serving, config, monkeypatch):
    children, router = monitor_setup(serving, monkeypatch)
    router.result = 1
    with pytest.raises(RuntimeError, match="Router or first-node"):
        serving.monitor(config, children, router, "key", 3600)


@pytest.mark.parametrize("accept_unauthorized", [False, True])
def test_authentication_against_fake_http_server(serving, accept_unauthorized):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.headers.get("Authorization"))
            self.send_response(200 if requests[-1] == "Bearer private" else 401)
            self.end_headers()

        def do_POST(self):
            requests.append(self.headers.get("Authorization"))
            self.send_response(200 if accept_unauthorized else 401)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        if accept_unauthorized:
            with pytest.raises(RuntimeError, match="authentication"):
                serving.verify_auth(url, "private")
        else:
            serving.verify_auth(url, "private")
            assert requests == ["Bearer private", None, "Bearer incorrect-key"]
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.parametrize("nodes", [1, 2])
@pytest.mark.parametrize("gpus", [4, 8])
def test_launch_fake_slurm_and_signal_cleanup(serving, tmp_path, monkeypatch, nodes, gpus):
    monkeypatch.setenv("SLURM_GPUS_PER_NODE", str(gpus))
    run = tmp_path / "run with spaces"
    snapshot = tmp_path / "snapshot with spaces"
    run.mkdir()
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    image = tmp_path / "image.sqsh"
    image.touch()
    monkeypatch.setenv("QWEN_IMAGE", str(image))
    monkeypatch.setenv("SLURM_JOB_NODELIST", "nodes")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("USER", "tester")
    calls, spawned, stopped = [], [], []

    def check_output(command, **kwargs):
        assert command == ["scontrol", "show", "hostnames", "nodes"]
        return "\n".join(f"node-{i}" for i in range(nodes))

    def spawn(command, **kwargs):
        calls.append(command)
        proc = Process()
        spawned.append(proc)
        return proc

    def monitor(*args):
        (run / "endpoint.json").write_text("{}")
        serving.interrupted(signal.SIGTERM, None)

    monkeypatch.setattr(serving.subprocess, "check_output", check_output)
    monkeypatch.setattr(serving, "spawn", spawn)
    monkeypatch.setattr(serving, "monitor", monitor)
    monkeypatch.setattr(serving, "stop_processes", lambda ps: stopped.extend(ps))
    monkeypatch.setattr(serving.signal, "signal", lambda *a: None)
    with pytest.raises(SystemExit) as error:
        serving.launch(str(snapshot), str(run))
    assert error.value.code == 143
    assert len(calls) == nodes + 2  # staging step, one supervisor per node, one router
    assert all(command[0] == "srun" for command in calls)
    assert all(f"--gpus-per-task={gpus}" in command for command in calls)
    assert f"--nodes={nodes}" in calls[0]
    assert all("--nodes=1" in command for command in calls[1:])
    assert "--nodelist=node-0" in calls[-1]
    assert calls[-1][-2:] == ["router", "/launch/config.json"]
    assert calls[0][-1] == str(run / "config.json")  # a single argv item, despite spaces
    assert stopped == spawned[1:]
    assert not (run / "endpoint.json").exists()
    assert (run / "api-key").stat().st_mode & 0o777 == 0o600
    key = (run / "api-key").read_text().strip()
    assert key not in json.dumps(calls)
    assert key not in (run / "config.json").read_text()


def test_process_group_cleanup(serving, tmp_path):
    marker = tmp_path / "stopped"
    code = ("import signal,time,pathlib; "
            f"signal.signal(signal.SIGTERM, lambda *a: (pathlib.Path({str(marker)!r})"
            ".write_text('stopped'), exit(0))); print('ready', flush=True); time.sleep(60)")
    proc = subprocess.Popen([sys.executable, "-c", code], start_new_session=True,
                            stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "ready"
        serving.stop_processes([proc])
        assert proc.poll() == 0 and marker.read_text() == "stopped"
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def test_stage_reuses_model_cache(serving, tmp_path, monkeypatch):
    config = {"model_local": str(tmp_path / "models" / "revision"),
              "local": str(tmp_path / "job"), "snapshot": "/snapshot"}
    copies = []
    monkeypatch.setattr(serving.subprocess, "run", lambda cmd, **kw: copies.append(cmd))
    serving.stage(config)
    serving.stage(config)
    assert len(copies) == 1 and copies[0][:3] == ["rsync", "-aL", "--"]
    assert (tmp_path / "job" / "gpu-7").is_dir()


def test_four_allocated_devices_do_not_use_other_gpus(serving, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-e,GPU-f,GPU-g,GPU-h")
    assert serving.gpu_devices(4) == ["GPU-e", "GPU-f", "GPU-g", "GPU-h"]
    with pytest.raises(ValueError, match="8 distinct"):
        serving.gpu_devices(8)
