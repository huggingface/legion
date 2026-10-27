"""Linux process isolation for run4. Setup errors abort; there is no unsandboxed fallback.

unshare creates a user/PID/mount namespace and private procfs. The trusted setup
then builds a small root with read-only runtimes and only the caller's data,
chroots and drops capabilities before executing any agent-supplied code.
"""
from __future__ import annotations

import ctypes
import json
import os
import shutil
import sys
from pathlib import Path

MS_RDONLY, MS_NOSUID, MS_NODEV, MS_NOEXEC = 1, 2, 4, 8
MS_REMOUNT, MS_BIND = 32, 4096


def enabled() -> bool:
    return os.environ.get("FLOCK_PROCESS_ISOLATION") == "1"


def _libc():
    return ctypes.CDLL(None, use_errno=True)


def protect_supervisor() -> None:
    """Deny same-UID ptrace and /proc/<pid>/{environ,mem,root} on worker/world."""
    if not sys.platform.startswith("linux") or _libc().prctl(4, 0, 0, 0, 0) != 0:
        raise RuntimeError("PR_SET_DUMPABLE=0 is required for run4")


def runtime_paths(command: list[str], *, python_only=False) -> list[str]:
    paths = ["/usr", "/bin", "/sbin", "/lib", "/lib64", sys.prefix, sys.base_prefix]
    # Config files needed by Python, Node, TLS, DNS and git; never a host home or /fsx root.
    paths += ["/etc/ssl", "/etc/alternatives", "/etc/hosts", "/etc/resolv.conf",
              "/etc/nsswitch.conf", "/etc/passwd", "/etc/group", "/etc/localtime"]
    if not python_only:
        paths.append(str(Path(__file__).resolve().parents[2]))  # editable flock install
        for binary in (command[0], "node"):
            found = shutil.which(binary)
            if not found:
                continue
            target = Path(found).resolve()
            modules = next((p for p in target.parents if p.name == "node_modules"), None)
            paths.append(str(modules or target.parent))
    return sorted({str(Path(p).absolute()) for p in paths if Path(p).exists()}, key=lambda p: (len(p), p))


def symlink_chain(path: str | Path) -> tuple[list[tuple[str, str]], str]:
    """Resolve `path` one hop at a time: every symlink met on the way (as `(link, target)`,
    the target exactly as stored) and the final real path. Components that do not exist are
    kept as they are. A loop or more than 40 hops raises."""
    links: list[tuple[str, str]] = []
    pending = list(Path(os.path.abspath(path)).parts[1:])
    current = Path("/")
    while pending:
        name = pending.pop(0)
        if name in ("", "."):
            continue
        if name == "..":
            current = current.parent
            continue
        candidate = current / name
        if not candidate.is_symlink():
            current = candidate
            continue
        target = os.readlink(candidate)
        links.append((str(candidate), target))
        if len(links) > 40:
            raise RuntimeError(f"isolation: too many symlinks resolving {path}")
        if os.path.isabs(target):
            current = Path("/")
            pending = list(Path(target).parts[1:]) + pending
        else:
            pending = list(Path(target).parts) + pending
    return links, str(current)


def _within(path: str, roots: list[str]) -> bool:
    return any(path == r or path.startswith(r.rstrip("/") + "/") for r in roots)


def plan(argv: list[str], *, cwd: Path, readonly=(), writable=(), python_only=False,
         private_dirs=()) -> dict:
    """Mounts and symlinks for the chroot. Every path is bound at its real location, and each
    symlink hop on the way to it (and to the interpreter, the command and node) is recreated
    inside, so the original spelling resolves there too (a uv venv's python links through
    `cpython-3.12-...` to `cpython-3.12.14-...`)."""
    links: dict[str, str] = {}

    def real(p) -> str:
        hops, final = symlink_chain(p)
        links.update(hops)
        return final

    ro = [real(p) for p in runtime_paths(argv, python_only=python_only) + [str(p) for p in readonly]]
    rw = [real(p) for p in writable]
    probes = [sys.executable, argv[0] if os.path.isabs(argv[0]) else shutil.which(argv[0])]
    if not python_only:
        probes.append(shutil.which("node"))
    for probe in filter(None, probes):
        final = real(probe)
        if not _within(final, ro + rw):
            raise RuntimeError(f"isolation cannot represent {probe}: it resolves to {final}, "
                               "outside every mounted path")
    real(cwd)
    def ordered(paths: list[str]) -> list[str]:  # parents before nested mounts
        return sorted(dict.fromkeys(paths), key=lambda p: (len(p), p))

    return {"readonly": ordered(ro), "writable": list(dict.fromkeys(rw)),
            "symlinks": sorted(links.items(), key=lambda kv: (len(kv[0]), kv[0])),
            "private_dirs": [str(Path(p).absolute()) for p in private_dirs]}


def command(argv: list[str], root: Path, *, cwd: Path, readonly=(), writable=(), network=False,
            python_only=False, private_dirs=()) -> list[str]:
    """Build a trusted invocation. Only explicitly mounted paths survive the chroot."""
    unshare = shutil.which("unshare")
    if not unshare:
        raise RuntimeError("unshare is required for run4 process isolation")
    config = {"root": str(root.absolute()), "cwd": str(cwd.absolute()),
              "readonly_root": not network,
              **plan(argv, cwd=cwd, readonly=readonly, writable=writable, python_only=python_only,
                     private_dirs=private_dirs)}
    # Slurm sends USR1 to every process on reclaim. Only worker/world supervisors
    # handle it; an unshare wrapper must survive long enough for their bounded drain.
    return ["/bin/sh", "-c", 'trap "" USR1; exec "$@"', "flock-isolate",
            unshare, "--user", "--map-root-user", "--pid", "--fork", "--kill-child=SIGKILL",
            "--mount", "--propagation", "private", "--mount-proc", *(["--net"] if not network else []), "--",
            sys.executable, "-I", str(Path(__file__).resolve()), json.dumps(config), *argv]


def _mount(source, target, kind=None, flags=0, data=None):
    def encoded(value):
        return os.fsencode(value) if value is not None else None
    libc = _libc()
    libc.mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                           ctypes.c_ulong, ctypes.c_char_p]
    if libc.mount(encoded(source), encoded(target), encoded(kind), flags, encoded(data)) != 0:
        raise OSError(ctypes.get_errno(), "isolation mount failed")


def _bind(root: Path, source: str, *, readonly: bool):
    path = Path(source)
    if not path.exists():
        raise RuntimeError("required isolation path is missing")
    target = root / path.relative_to("/")
    target.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        target.mkdir(exist_ok=True)
    else:
        target.touch(exist_ok=True)
    # No recursive binds: a nested host mount must not smuggle extra files/procfs in.
    _mount(source, target, flags=MS_BIND)
    if readonly:
        flags = MS_BIND | MS_REMOUNT | MS_RDONLY | MS_NOSUID | MS_NODEV
        # An unprivileged namespace cannot clear locked flags inherited from its
        # parent mounts (notably noexec on /etc files and atime settings).
        inherited = os.statvfs(source).f_flag
        for flag, mount_flag in ((os.ST_NOEXEC, MS_NOEXEC), (os.ST_NOATIME, 1024),
                                 (os.ST_NODIRATIME, 2048), (os.ST_RELATIME, 1 << 21)):
            if inherited & flag:
                flags |= mount_flag
        _mount(None, target, flags=flags)


def _drop_privileges():
    libc = _libc()
    # NOROOT + locked: exec as namespace uid 0 must never regain capabilities.
    if libc.prctl(28, 3, 0, 0, 0) or libc.prctl(38, 1, 0, 0, 0):
        raise RuntimeError("cannot lock sandbox privileges")
    # All currently defined capabilities fit below 64. EINVAL marks unsupported bits.
    for cap in range(64):
        if libc.prctl(24, cap, 0, 0, 0) != 0 and ctypes.get_errno() != 22:
            raise RuntimeError("cannot drop sandbox capability bounding set")
    class Header(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]
    class Data(ctypes.Structure):
        _fields_ = [("effective", ctypes.c_uint32), ("permitted", ctypes.c_uint32),
                    ("inheritable", ctypes.c_uint32)]
    if libc.capset(ctypes.byref(Header(0x20080522, 0)), (Data * 2)()) != 0:
        raise RuntimeError("cannot clear sandbox capabilities")


def _enter(config, argv):
    # Never share a process group with a supervisor outside the PID namespace.
    if os.getpgrp() != os.getpid():
        os.setsid()
    root = Path(config["root"])
    root.mkdir(parents=True, exist_ok=True)
    _mount("tmpfs", root, "tmpfs", MS_NOSUID | MS_NODEV, "mode=755,size=256m")
    for path in config["readonly"]:
        _bind(root, path, readonly=True)
    for path in config["writable"]:
        _bind(root, path, readonly=False)
    for link, target in config.get("symlinks", ()):
        inside = root / Path(link).relative_to("/")
        if os.path.lexists(inside):
            continue  # already visible through a bind mount
        inside.parent.mkdir(parents=True, exist_ok=True)
        inside.symlink_to(target)
    for path in config["private_dirs"]:
        (root / Path(path).relative_to("/")).mkdir(parents=True, exist_ok=True)
    for path in ("tmp", "proc", "dev"):
        (root / path).mkdir(exist_ok=True)
    for path in ("/dev/null", "/dev/zero", "/dev/urandom", "/dev/random"):
        _bind(root, path, readonly=False)
    (root / "dev/fd").symlink_to("/proc/self/fd")
    _mount("proc", root / "proc", "proc", MS_NOSUID | MS_NODEV | MS_NOEXEC)
    if config["readonly_root"]:
        _mount(None, root, flags=MS_REMOUNT | MS_RDONLY | MS_NOSUID | MS_NODEV)
    os.chroot(root)
    os.chdir(config["cwd"])
    _drop_privileges()
    os.execvpe(argv[0], argv, dict(os.environ))


if __name__ == "__main__":
    try:
        _enter(json.loads(sys.argv[1]), sys.argv[2:])
    except Exception:  # noqa: BLE001 - fail closed, without leaking arguments or paths
        # Paths, argv and errors could contain credentials. Fail closed without their values.
        print("run4 process isolation failed during setup", file=sys.stderr)
        raise SystemExit(125) from None
