"""CgroupMem on a fake cgroupfs, and the slope/ETA maths with an injected clock."""

from pathlib import Path

import pytest

from flock.worker.cgroup import MB, CgroupMem, fs_type


def _w(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def v2(root: Path, *, leaf_max: str = "max", parent_max: str = "11811160064") -> Path:
    _w(root / "proc/self/cgroup", "0::/job/step\n")
    base = root / "sys/fs/cgroup"
    _w(base / "cgroup.controllers", "memory\n")
    _w(base / "job/memory.max", parent_max + "\n")
    leaf = base / "job/step"
    _w(leaf / "memory.max", leaf_max + "\n")
    _w(leaf / "memory.current", str(5000 * MB))
    _w(leaf / "memory.stat", f"anon {1000 * MB}\nfile {3000 * MB}\nshmem {200 * MB}\n"
       f"slab_unreclaimable {50 * MB}\nkernel_stack {10 * MB}\npagetables {40 * MB}\n")
    _w(leaf / "memory.events", "low 0\nhigh 0\nmax 3\noom 1\noom_kill 2\n")
    _w(leaf / "memory.pressure", "some avg10=1.50 avg60=0.20 avg300=0.00 total=1\n"
       "full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n")
    return leaf


def test_v2_parent_limit_and_fields(tmp_path):
    v2(tmp_path)
    cg = CgroupMem(tmp_path, env={})
    assert cg.version == 2 and cg.path == tmp_path / "sys/fs/cgroup/job/step"
    assert cg.armed and cg.limit_mb == 11264  # 11 GiB on the parent, the leaf says max
    s = cg.sample()
    assert s["current_mb"] == 5000 and s["nonreclaim_mb"] == 1300  # file pages left out
    assert s["psi_some_avg10"] == 1.5 and s["oom_kills"] == 2
    assert s["slope_mb_h"] is None and s["eta_limit_s"] is None


def test_v2_smallest_limit_wins(tmp_path):
    v2(tmp_path, leaf_max=str(2048 * MB))
    assert CgroupMem(tmp_path, env={}).limit_mb == 2048


def test_no_limit_not_armed_unless_env(tmp_path):
    v2(tmp_path, parent_max="max")
    assert not CgroupMem(tmp_path, env={}).armed
    cg = CgroupMem(tmp_path, env={"FLOCK_MEMORY_LIMIT_MB": "4096"})
    assert cg.armed and cg.limit_mb == 4096
    both = {"SLURM_MEM_PER_NODE": "7168", "FLOCK_MEMORY_LIMIT_MB": "4096"}
    assert CgroupMem(tmp_path, env=both).limit_mb == 7168  # Slurm's job limit first
    assert not CgroupMem(tmp_path / "nothing", env={}).armed


def test_host_namespace_path_falls_back_to_the_base(tmp_path):
    v2(tmp_path)
    _w(tmp_path / "proc/self/cgroup", "0::/kubepods/burstable/pod1/abc\n")
    _w(tmp_path / "sys/fs/cgroup/memory.max", str(4096 * MB))
    cg = CgroupMem(tmp_path, env={})
    assert cg.path == tmp_path / "sys/fs/cgroup" and cg.limit_mb == 4096


def test_v1_fallback(tmp_path):
    _w(tmp_path / "proc/self/cgroup", "4:cpu,cpuacct:/x\n3:memory:/slurm/job7\n0::/\n")
    base = tmp_path / "sys/fs/cgroup/memory"
    _w(base / "memory.limit_in_bytes", str(1 << 63))
    _w(base / "slurm/memory.limit_in_bytes", str(1 << 63))
    d = base / "slurm/job7"
    _w(d / "memory.limit_in_bytes", str(8192 * MB))
    _w(d / "memory.usage_in_bytes", str(3000 * MB))
    _w(d / "memory.stat", f"rss 1 \ntotal_rss {700 * MB}\ntotal_shmem {100 * MB}\ntotal_cache 9\n")
    _w(d / "memory.oom_control", "oom_kill_disable 0\nunder_oom 0\noom_kill 1\n")
    cg = CgroupMem(tmp_path, env={})
    assert cg.version == 1 and cg.limit_mb == 8192
    s = cg.sample()
    assert s["current_mb"] == 3000 and s["nonreclaim_mb"] == 800 and s["oom_kills"] == 1
    assert s["psi_some_avg10"] is None


def test_slope_and_eta(tmp_path):
    leaf = v2(tmp_path, leaf_max=str(2300 * MB))
    now = [0.0]
    cg = CgroupMem(tmp_path, env={}, clock=lambda: now[0])
    s = None
    for i in range(0, 61, 5):  # +100 MB every 5 min of non-reclaimable anon: 1200 MB/h
        now[0] = i * 60.0
        _w(leaf / "memory.stat", f"anon {(1000 + 100 * i // 5) * MB}\n")
        s = cg.sample()
    assert s["nonreclaim_mb"] == 2200 and s["slope_mb_h"] == 1200
    assert s["eta_limit_s"] == 300  # 100 MB left at 1200 MB/h
    now[0] = 2 * 3600.0 + 1  # samples older than an hour drop out of the window
    cg.sample()
    assert len(cg.samples) == 1 and cg.slope_mb_h() is None


def test_fs_type_longest_mount():
    mounts = "overlay / overlay rw 0 0\n/dev/nvme1 /work ext4 rw 0 0\ntmpfs /work/tmp tmpfs rw 0 0\n"
    assert fs_type(Path("/work/agents"), mounts) == "ext4"
    assert fs_type(Path("/work/tmp/x"), mounts) == "tmpfs"
    assert fs_type(Path("/opt"), mounts) == "overlay"


def test_agent_memory_of_a_live_process_tree(tmp_path):
    import os
    import subprocess
    import sys

    from flock.worker.cgroup import _descendants, agent_memory

    (tmp_path / "tmp").mkdir()
    (tmp_path / "tmp" / "f").write_bytes(b"x" * MB)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        m = agent_memory(os.getpid(), _descendants, tmp_path / "tmp")
    finally:
        child.kill()
        child.wait()
    assert m["pi_pss_mb"] > 0 and m["tmp_mb"] == 1.0 and m["tmpfs_root_mb"] is not None
    assert m["tools_pss_mb"] > 0 and m["tools_by_comm"]
    assert agent_memory(None, _descendants, tmp_path / "tmp") == {"tmp_mb": 1.0}


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="Linux /proc")
def test_kill_all_kills_only_the_same_live_processes():
    import os
    import subprocess
    import sys

    from flock.worker.cgroup import _descendants, _kill_all

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        [(pid, start)] = [p for p in _descendants(os.getpid()) if p[0] == child.pid]
        assert _kill_all([(pid, "not-its-start-time")]) == 0  # a reused pid is left alone
        assert _kill_all([(pid, start)]) == 1 and child.wait(timeout=5) == -9
    finally:
        child.kill()
        child.wait()


def test_oom_score_adj_is_raised_for_the_sandbox(tmp_path):
    """isolation._enter raises the sandbox's OOM score (run in a child: never raise ours; 1000,
    since lowering a score that is already higher needs privilege)."""
    import subprocess
    import sys

    import pytest

    code = ("from pathlib import Path; from flock.isolation import set_oom_score_adj as s; "
            "print(s(1000), Path('/proc/self/oom_score_adj').read_text().strip())")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout.split()
    if out[0] != "True":
        pytest.skip("/proc/self/oom_score_adj is not writable here")
    assert out[1] == "1000"
    from flock.isolation import set_oom_score_adj

    assert not set_oom_score_adj(path=str(tmp_path / "missing" / "oom_score_adj"))
