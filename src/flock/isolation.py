"""Linux process isolation for run4. Setup errors abort; there is no unsandboxed fallback.

unshare creates a user/PID/mount namespace and private procfs. The trusted setup
then builds a small root with read-only runtimes and only the caller's data,
chroots and drops capabilities before executing any agent-supplied code.
"""
from __future__ import annotations

import contextlib
import ctypes
import json
import os
import re
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path

MS_RDONLY, MS_NOSUID, MS_NODEV, MS_NOEXEC = 1, 2, 4, 8
MS_REMOUNT, MS_BIND = 32, 4096
MS_NOATIME, MS_NODIRATIME, MS_REC, MS_RELATIME, MS_STRICTATIME = 1024, 2048, 16384, 1 << 21, 1 << 24
# statvfs flags of a mount that an unprivileged namespace cannot clear (they are locked) ->
# the mount(2) flags a read-only remount must repeat.
_LOCKED = ((os.ST_RDONLY, MS_RDONLY), (os.ST_NOSUID, MS_NOSUID), (os.ST_NODEV, MS_NODEV),
           (os.ST_NOEXEC, MS_NOEXEC), (os.ST_NOATIME, MS_NOATIME),
           (os.ST_NODIRATIME, MS_NODIRATIME), (os.ST_RELATIME, MS_RELATIME))


DEV_NODES = ("/dev/null", "/dev/zero", "/dev/urandom", "/dev/random")
MOUNT_ATTR_RDONLY, AT_FDCWD, AT_RECURSIVE, SYS_MOUNT_SETATTR = 0x1, -100, 0x8000, 442


class IsolationSetupError(OSError):
    """A sandbox setup step failed: names the step and path (never argv or the env)."""


class IsolationMountError(IsolationSetupError):
    """A mount(2) call of the sandbox setup failed: names the target and flags, no argv."""


@contextlib.contextmanager
def _step(step: str, path) -> Iterator[None]:
    """Turn a plain OSError of one setup step into an IsolationSetupError naming it."""
    try:
        yield
    except IsolationSetupError:
        raise
    except OSError as e:
        raise IsolationSetupError(e.errno, f"isolation {step} failed: "
                                           f"{e.strerror or type(e).__name__}: path={path}") from None


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
        errno = ctypes.get_errno()
        raise IsolationMountError(errno, f"isolation mount failed: {os.strerror(errno)}: "
                                         f"source={source} target={target} type={kind} "
                                         f"flags={flags:#x}")


def readonly_remount_flags(target) -> int:
    """MS_REMOUNT|MS_BIND|MS_RDONLY plus nosuid/nodev, and every locked flag the bound mount
    already has (statvfs of the bind target): a user namespace cannot clear them, and
    leaving them out fails with EPERM. No atime flag on the source means strictatime."""
    inherited = os.statvfs(target).f_flag
    flags = MS_BIND | MS_REMOUNT | MS_RDONLY | MS_NOSUID | MS_NODEV
    for flag, mount_flag in _LOCKED:
        if inherited & flag:
            flags |= mount_flag
    if not inherited & (os.ST_NOATIME | os.ST_RELATIME):
        flags |= MS_STRICTATIME
    return flags


def _readonly_recursive(target) -> bool:
    """mount_setattr(AT_RECURSIVE, MOUNT_ATTR_RDONLY): make every submount of a recursive
    bind read-only too (a remount only changes the top one). False if the kernel lacks it
    (before 5.12); the mountinfo check still refuses a writable submount."""
    libc = _libc()
    attr = (ctypes.c_uint64 * 4)(MOUNT_ATTR_RDONLY, 0, 0, 0)
    libc.syscall.restype = ctypes.c_long
    rc = libc.syscall(ctypes.c_long(SYS_MOUNT_SETATTR), ctypes.c_int(AT_FDCWD),
                      ctypes.c_char_p(os.fsencode(target)), ctypes.c_uint(AT_RECURSIVE),
                      ctypes.byref(attr), ctypes.c_size_t(ctypes.sizeof(attr)))
    # Any failure (ENOSYS before 5.12, EPERM on a locked mount): the mountinfo check that
    # follows the binds still refuses a writable submount.
    return rc == 0


def _unescape(field: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), field)


def parse_mountinfo(text: str) -> list[tuple[str, set[str]]]:
    """(mount point, per-mount options) of each /proc/self/mountinfo line."""
    out = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 6:
            out.append((_unescape(fields[4]), set(fields[5].split(","))))
    return out


def check_mounts(mounts: list[tuple[str, set[str]]], root: str, readonly: list[str],
                 writable: list[str] = (), extra: list[str] = ()) -> None:
    """Fail closed after the binds. Every mount under `root` must be a planned target (or
    `extra`: proc, the /dev nodes), or a submount a recursive bind of one brought along
    (e.g. a container's zoneinfo file bound over /usr's); and every mount at or under a
    read-only target, outside a writable one, must be `ro`."""
    def under(path: str, roots) -> bool:
        return any(path == r or path.startswith(r.rstrip("/") + "/") for r in roots)

    planned = [*readonly, *writable, *extra]
    for point, opts in mounts:
        if point == root or not under(point, [root]):
            continue
        if not under(point, planned):
            raise IsolationMountError(1, f"isolation: unexpected mount {point} in the sandbox root")
        if under(point, readonly) and not under(point, writable) and "ro" not in opts:
            raise IsolationMountError(30, f"isolation: {point} is mounted writable under a "
                                          "read-only path")


def _bind(root: Path, source: str, *, readonly: bool):
    path = Path(source)
    if not path.exists():
        raise IsolationSetupError(2, f"isolation bind failed: required path is missing: path={source}")
    target = root / path.relative_to("/")
    # Create the mount point only when missing: it may already exist on a read-only bind
    # (/etc/localtime -> /usr/share/zoneinfo/... under the /usr bind), where touch()'s utime
    # fails with EROFS.
    with _step("mount point creation", target):
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            if not target.is_dir():
                target.mkdir()
        elif not target.exists():
            target.touch()
    # Recursive: a user namespace refuses (EINVAL) a non-recursive bind of a tree that has
    # locked submounts, since it would uncover what they hide. Only explicitly listed paths
    # are bound, so a submount comes in only under a path that was asked for.
    _mount(source, target, flags=MS_BIND | MS_REC)
    if readonly:
        # Step two: read-only only works as a remount of the bind, with the locked flags;
        # then the submounts the recursive bind brought along, where the kernel can.
        _mount(None, target, flags=readonly_remount_flags(target))
        _readonly_recursive(target)


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
    with _step("root creation", root):
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
        with _step("symlink creation", inside):
            inside.parent.mkdir(parents=True, exist_ok=True)
            inside.symlink_to(target)
    for path in config["private_dirs"]:
        with _step("private directory creation", path):
            (root / Path(path).relative_to("/")).mkdir(parents=True, exist_ok=True)
    for path in ("tmp", "proc", "dev"):
        with _step("directory creation", root / path):
            (root / path).mkdir(exist_ok=True)
    for path in DEV_NODES:
        _bind(root, path, readonly=False)
    with _step("symlink creation", root / "dev/fd"):
        (root / "dev/fd").symlink_to("/proc/self/fd")
    _mount("proc", root / "proc", "proc", MS_NOSUID | MS_NODEV | MS_NOEXEC)
    inside = lambda p: os.path.realpath(root / Path(p).relative_to("/"))
    check_mounts(parse_mountinfo(Path("/proc/self/mountinfo").read_text()), os.path.realpath(root),
                 [inside(p) for p in config["readonly"]], [inside(p) for p in config["writable"]],
                 [inside(p) for p in ("/proc", *DEV_NODES)])
    if config["readonly_root"]:
        _mount(None, root, flags=MS_REMOUNT | MS_RDONLY | MS_NOSUID | MS_NODEV)
    with _step("chroot", root):
        os.chroot(root)
    with _step("chdir", config["cwd"]):
        os.chdir(config["cwd"])
    _drop_privileges()
    os.execvpe(argv[0], argv, dict(os.environ))


if __name__ == "__main__":
    try:
        _enter(json.loads(sys.argv[1]), sys.argv[2:])
    except IsolationSetupError as e:  # step, path and flags only, never argv or the env
        print(f"run4 process isolation failed during setup: {e.strerror}", file=sys.stderr)
        raise SystemExit(125) from None
    except Exception:  # noqa: BLE001 - fail closed, without leaking arguments or paths
        # Paths, argv and errors could contain credentials. Fail closed without their values.
        print("run4 process isolation failed during setup", file=sys.stderr)
        raise SystemExit(125) from None
