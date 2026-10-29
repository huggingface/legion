"""Slurm coordinator and node supervisor for independent Qwen replicas (stdlib only)."""
from __future__ import annotations

import concurrent.futures
import fcntl
import hashlib
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "Qwen/Qwen3.8-27B"
IMAGE = "/fsx/anton/shared/images/vllm-openai-v0.32.0.sqsh"
STARTUP = 3600
OUTAGE = 600
POLL = 5


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def http_status(url, key=None, payload=None):
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode() if payload is not None else None, headers=headers
    )
    # Cluster traffic must not go through a login node's HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=2) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code
    except (OSError, urllib.error.URLError):
        return 0


def health(urls):
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, len(urls))) as pool:
        return list(pool.map(lambda url: http_status(url + "/health") == 200, urls))


def stop_processes(processes):
    # Each child has its own process group; terminate vLLM's engine descendants too.
    for proc in processes:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 20
    while any(proc.poll() is None for proc in processes) and time.monotonic() < deadline:
        time.sleep(0.1)
    for proc in processes:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def interrupted(signum, frame):
    raise SystemExit(128 + signum)


def spawn(command, *, env=None, log=None):
    if log is None:
        return subprocess.Popen(command, env=env, start_new_session=True)
    with open(log, "ab", buffering=0) as stream:
        return subprocess.Popen(command, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True)


def instances(hosts):
    return [{"hostname": host, "gpu_index": gpu, "port": 8100 + gpu,
             "url": f"http://{host}:{8100 + gpu}"} for host in hosts for gpu in range(8)]


def srun(hosts):
    return ["srun", "--overlap", f"--nodes={len(hosts)}", f"--ntasks={len(hosts)}",
            "--ntasks-per-node=1", "--gpus-per-task=8", "--kill-on-bad-exit=0",
            "--nodelist=" + ",".join(hosts)]


def container(config, hosts):
    run, local = config["run"], config["local"]
    # Mount paths use Pyxis's comma/colon syntax, checked before submission work starts.
    mounts = f'{config["model_local"]}:/model:ro,{local}:/job,{run}:/launch:ro'
    mounts += f",{run}/logs:/logs"
    return srun(hosts) + [f'--container-image={config["image"]}',
                          "--no-container-mount-home", f"--container-mounts={mounts}"]


def stage(config):
    model = Path(config["model_local"])
    model.parent.mkdir(parents=True, exist_ok=True)
    with (model.parent / "stage.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (model / ".complete").exists():
            model.mkdir(exist_ok=True)
            subprocess.run(["rsync", "-aL", "--", config["snapshot"] + "/", str(model) + "/"],
                           check=True)
            (model / ".complete").touch()
    local = Path(config["local"])
    for name in ["tmp", "router"] + [f"gpu-{i}" for i in range(8)]:
        (local / name).mkdir(parents=True, exist_ok=True)


def gpu_devices():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        visible = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
        ).strip().replace("\n", ",")
    devices = [device.strip() for device in visible.split(",") if device.strip()]
    if len(devices) != 8 or len(set(devices)) != 8:
        raise ValueError(f"Expected eight distinct allocated GPUs, got {devices!r}")
    return devices


def model_command(config, index):
    return ["vllm", "serve", "/model", "--served-model-name", MODEL,
            "--host", "0.0.0.0", "--port", str(8100 + index), "--dtype", "bfloat16",
            "--tensor-parallel-size", "1", "--data-parallel-size", "1",
            "--max-model-len", str(config["context"]), "--gpu-memory-utilization", "0.9",
            "--max-num-seqs", str(config["sequences"]),
            "--max-num-batched-tokens", str(config["batch_tokens"]),
            "--reasoning-parser", "qwen3", "--enable-auto-tool-choice",
            "--tool-call-parser", "qwen3_coder", "--language-model-only",
            "--default-chat-template-kwargs", '{"reasoning_effort":"low"}',
            "--enable-prefix-caching"]


class Replica:
    def __init__(self, config, host, index, device, key):
        self.command = model_command(config, index)
        self.env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, VLLM_API_KEY=key,
                        HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                        TMPDIR="/job/tmp", VLLM_CACHE_ROOT=f"/job/gpu-{index}/vllm",
                        TRITON_CACHE_DIR=f"/job/gpu-{index}/triton")
        self.log = f"/logs/{host}-gpu-{index}.log"
        self.proc = None
        self.failures = 0
        self.next_start = 0
        self.started = 0
        self.healthy_since = None
        self.was_healthy = False
        self.misses = 0

    def tick(self, now, healthy):
        if self.proc is None:
            if self.failures < 5 and now >= self.next_start:
                try:
                    self.proc = spawn(self.command, env=self.env, log=self.log)
                except OSError as error:
                    print(f"{self.log}: could not start: {error}", flush=True)
                    self.failed(now)
                    return
                self.started = now
                self.was_healthy = False
                self.misses = 0
            return
        if healthy and self.proc.poll() is None:
            self.was_healthy = True
            self.misses = 0
            if self.healthy_since is None:
                self.healthy_since = now
            if now - self.healthy_since >= 600:
                self.failures = 0
        else:
            self.healthy_since = None
            self.misses += 1
        failed = self.proc.poll() is not None
        failed |= self.was_healthy and self.misses >= 3
        failed |= not self.was_healthy and now - self.started >= STARTUP
        if failed:
            self.failed(now)

    def failed(self, now):
        if self.proc is not None:
            stop_processes([self.proc])
        self.proc = None
        self.healthy_since = None
        self.failures += 1
        self.next_start = now + min(120, 10 * 2 ** (self.failures - 1))
        action = "retry budget exhausted" if self.failures >= 5 else "restart scheduled"
        print(f"{self.log}: failure {self.failures}/5; {action}", flush=True)


def node(config, host):
    devices = gpu_devices()
    key = Path("/launch/api-key").read_text().strip()
    replicas = [Replica(config, host, i, device, key) for i, device in enumerate(devices)]
    atomic_json(f"/logs/{host}-devices.json", devices)
    try:
        while True:
            states = health([f"http://127.0.0.1:{8100 + i}" for i in range(8)])
            for replica, healthy in zip(replicas, states):
                replica.tick(time.monotonic(), healthy)
            time.sleep(POLL)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        stop_processes([replica.proc for replica in replicas if replica.proc is not None])


def router_command(config):
    urls = [item["url"] for item in instances(config["hosts"])]
    return [sys.executable, "-m", "vllm_router.launch_router", "--host", "0.0.0.0",
            "--port", "8000", "--worker-urls", *urls, "--policy", "cache_aware",
            "--intra-node-data-parallel-size", "1", "--disable-retries",
            "--worker-startup-timeout-secs", str(STARTUP),
            "--request-timeout-secs", "14400", "--health-check-interval-secs", "5",
            "--health-check-timeout-secs", "2", "--health-failure-threshold", "3",
            "--health-success-threshold", "2"]


def router(config):
    subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                    "--target", "/job/router", "vllm-router==0.1.16"], check=True)
    env = dict(os.environ, PYTHONPATH="/job/router", TMPDIR="/job/tmp", RUST_LOG="warn")
    # Do not supply the router's backend API key: forward the caller's header unchanged.
    command = router_command(config)
    os.execvpe(command[0], command, env)


def verify_auth(base, key):
    if http_status(base + "/v1/models", key) != 200:
        raise RuntimeError("Authenticated router model discovery failed")
    payload = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
    for supplied in (None, "incorrect-key"):
        if http_status(base + "/v1/chat/completions", supplied, payload) != 401:
            raise RuntimeError("Router inference authentication check failed")


def monitor(config, children, router_proc, key, deadline):
    run = Path(config["run"])
    backends = instances(config["hosts"])
    urls = [item["url"] for item in backends]
    base = f'http://{config["hosts"][0]}:8000'
    published = False
    outage_since = None
    while True:
        now = time.monotonic()
        if router_proc.poll() is not None or children[0].poll() is not None:
            raise RuntimeError("Router or first-node supervisor exited")
        states = health(urls)
        ready = all(states) and http_status(base + "/v1/models", key) == 200
        if not published:
            if any(proc.poll() is not None for proc in children):
                raise RuntimeError("A node supervisor exited during startup")
            if ready:
                verify_auth(base, key)
                for item in backends:
                    devices = json.loads((run / "logs" / f'{item["hostname"]}-devices.json')
                                         .read_text())
                    item["gpu_device"] = devices[item["gpu_index"]]
                atomic_json(run / "endpoint.json", {"model": MODEL, "base_url": base + "/v1",
                            "job_id": config["job_id"], "instances": backends})
                print(f"Ready: {base}/v1 ({len(backends)} instances)", flush=True)
                published = True
            elif now >= deadline:
                raise RuntimeError("Startup exceeded one hour; not all instances/router ready")
        else:
            outage_since = None if any(states) else (outage_since or now)
            if outage_since is not None and now - outage_since >= OUTAGE:
                raise RuntimeError("No healthy instances for ten minutes")
        atomic_json(run / "status.json", {"ready": published, "healthy": sum(states),
                    "total": len(states), "checked_at": time.time()})
        time.sleep(POLL)


def launch(snapshot, run):
    snapshot, run = Path(snapshot).resolve(), Path(run).resolve()
    image = Path(os.environ.get("QWEN_IMAGE", IMAGE)).resolve()
    if not run.is_dir() or not (snapshot / "config.json").is_file() or not image.is_file():
        raise ValueError("Missing run directory, model config, or container image")
    if any(path.name != "slurm.log" for path in run.iterdir()):
        raise ValueError("Use an empty run directory (slurm.log may already exist)")
    for path in (snapshot, run, image):
        if any(char in str(path) for char in ",:\n"):
            raise ValueError("Paths cannot contain Pyxis mount separators (comma/colon/newline)")
    hosts = subprocess.check_output(
        ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]], text=True
    ).split()
    if not hosts or len(hosts) != len(set(hosts)):
        raise ValueError("Expected a nonempty list of distinct Slurm hosts")
    job = os.environ["SLURM_JOB_ID"]
    scratch = Path("/scratch") / os.environ["USER"] / "qwen-serving"
    digest = hashlib.sha256(str(snapshot).encode()).hexdigest()
    config = {"snapshot": str(snapshot), "run": str(run), "image": str(image),
              "hosts": hosts, "job_id": job, "local": str(scratch / "jobs" / job),
              "model_local": str(scratch / "models" / digest)}
    for name, variable, default in (("context", "MAX_MODEL_LEN", 32768),
                                    ("sequences", "MAX_NUM_SEQS", 128),
                                    ("batch_tokens", "MAX_BATCHED_TOKENS", 8192)):
        config[name] = int(os.environ.get(variable, default))
        if config[name] <= 0:
            raise ValueError(f"{variable} must be positive")
    (run / "logs").mkdir()
    shutil.copyfile(__file__, run / "qwen_serve.py")
    atomic_json(run / "config.json", config)
    key = secrets.token_urlsafe(32)
    with (run / "api-key").open("x") as stream:
        stream.write(key + "\n")
    (run / "api-key").chmod(0o600)
    helper = str(run / "qwen_serve.py")
    children = []
    deadline = time.monotonic() + STARTUP
    try:
        staging = spawn(srun(hosts) + ["python3", helper, "stage", str(run / "config.json")])
        children.append(staging)
        staging.wait(timeout=max(1, deadline - time.monotonic()))
        if staging.returncode:
            raise RuntimeError("Model staging failed")
        children.clear()
        for host in hosts:
            command = container(config, [host]) + ["python3", "/launch/qwen_serve.py", "node",
                                                   "/launch/config.json", host]
            children.append(spawn(command, log=run / "logs" / f"{host}-supervisor.log"))
        command = container(config, hosts[:1]) + ["python3", "/launch/qwen_serve.py", "router",
                                                 "/launch/config.json"]
        router_proc = spawn(command, log=run / "logs" / "router.log")
        children.append(router_proc)
        monitor(config, children[:-1], router_proc, key, deadline)
    finally:
        (run / "endpoint.json").unlink(missing_ok=True)
        # Ignore repeated scheduler signals while reaping all child process groups.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        stop_processes(children)


def main():
    os.umask(0o077)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    mode, *args = sys.argv[1:]
    if mode == "launch":
        launch(*args)
    else:
        config = json.loads(Path(args[0]).read_text())
        if mode == "stage":
            stage(config)
        elif mode == "node":
            node(config, args[1])
        elif mode == "router":
            router(config)
        else:
            raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
