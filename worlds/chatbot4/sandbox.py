"""Contract runner: isolated interpreter, bounded resources, no bot I/O except stdio/imports."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from flock.isolation import command as isolate_command

SHIM = r'''
import sys, os, resource, runpy
root = os.path.realpath(sys.argv[1])
resource.setrlimit(resource.RLIMIT_AS, (512 << 20, 512 << 20))
resource.setrlimit(resource.RLIMIT_CPU, (300, 300))
resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
sys.dont_write_bytecode = True
sys.path.insert(0, root)
allowed = set(sys.stdlib_module_names) | {'bot'}
blocked = {'ctypes', '_ctypes', '_posixsubprocess', 'subprocess', 'socket', '_socket', 'multiprocessing'}
def audit(event, args):
    if event == 'import':
        name = args[0].split('.')[0]
        if name not in allowed or name in blocked:
            raise PermissionError('only standard-library bot imports are allowed')
    if event == 'open':
        path, mode, flags = args
        if not isinstance(path, (str, bytes, os.PathLike)):
            raise PermissionError('file access is disabled')
        path = os.path.realpath(os.fsdecode(path))
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            raise PermissionError('file writes are disabled')
        roots = (root + os.sep, os.path.realpath(sys.base_prefix) + os.sep)
        if not path.startswith(roots) or not path.endswith(('.py', '.pyc', '.so')):
            raise PermissionError('file access is disabled except module loading')
    if event.startswith(('socket.', 'subprocess.', 'ctypes.', 'os.kill', 'signal.')) or event in {
        'os.system', 'os.exec', 'os.fork', 'os.forkpty', 'os.posix_spawn', 'os.kill', 'os.killpg',
        'os.remove', 'os.rename', 'os.mkdir', 'os.rmdir', 'os.chmod', 'os.link',
        'os.symlink', 'os.truncate', 'os.putenv', 'os.unsetenv'}:
        raise PermissionError('bot side effects are disabled')
import signal, _signal
# Signals to other processes: os.kill/killpg and signal.pthread_kill raise audit events
# (blocked above); pidfd_send_signal does not, so remove every sender from the modules.
for module in (os, sys.modules.get('posix'), signal, _signal):
    for name in ('kill', 'killpg', 'pthread_kill', 'pidfd_send_signal'):
        if module is not None and hasattr(module, name):
            delattr(module, name)
sys.addaudithook(audit)
runpy.run_module('bot', run_name='__main__')
'''


def run(root: Path, lines: list[str], timeout: float = 300) -> dict:
    if not lines or len(lines) > 200 or any(not isinstance(x, str) or '\n' in x or '\r' in x
                                           or len(x) > 16000 for x in lines):
        return {"ok": False, "status": "supply 1–200 single input lines (at most 16000 chars each)"}
    with (tempfile.TemporaryDirectory() as tmp,
          open(Path(tmp) / "out", "w+b") as out,
          open(Path(tmp) / "err", "w+b") as err):
        argv = isolate_command([sys.executable, "-I", "-S", "-c", SHIM, str(root.resolve())],
                               Path(tmp) / "root", cwd=root.resolve(), readonly=[root.resolve()],
                               python_only=True)
        proc = subprocess.Popen(argv, cwd=tmp,
                                env={}, stdin=subprocess.PIPE, stdout=out, stderr=err,
                                start_new_session=True)
        try:
            proc.communicate(('\n'.join(lines) + '\n').encode(), timeout=timeout)
            status = "ok" if proc.returncode == 0 else f"exit {proc.returncode}"
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            status = "timeout"
        out.seek(0)
        err.seek(0)
        replies = out.read(1 << 20).decode(errors="replace").splitlines()
        ok = status == "ok" and len(replies) == len(lines) and all(x.strip() for x in replies)
        return {"ok": ok, "status": status, "replies": replies,
                "stderr": err.read(4096).decode(errors="replace"), "timeout_s": timeout}
