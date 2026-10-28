"""CgroupMem on a fake cgroupfs, and the slope/ETA maths with an injected clock."""

from pathlib import Path

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
    assert not CgroupMem(tmp_path / "nothing", env={}).armed


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

    from flock.worker.adapters.pi import _descendants
    from flock.worker.cgroup import agent_memory

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
