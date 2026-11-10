"""Serve one native msgboard on localhost and an agent Unix socket."""

import signal
import socket
import sys
from pathlib import Path

import uvicorn
from msgboard.server import Board, create_app


def main():
    directory, host, port, socket_path = sys.argv[1:]
    board = Board(Path(directory))

    class Server(uvicorn.Server):
        def handle_exit(self, sig, frame):
            board.begin_shutdown()
            super().handle_exit(sig, frame)

    def exit_on_signal(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, exit_on_signal)
    signal.signal(signal.SIGINT, exit_on_signal)
    try:
        with socket.socket() as tcp, socket.socket(socket.AF_UNIX) as unix:
            tcp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            tcp.bind((host, int(port)))
            unix.bind(socket_path)
            # The containing host directory is private; the socket is mounted alone.
            Path(socket_path).chmod(0o666)
            config = uvicorn.Config(
                create_app(board),
                log_level="warning",
                timeout_graceful_shutdown=5,
                server_header=False,
            )
            Server(config).run(sockets=[tcp, unix])
    finally:
        board.close()


if __name__ == "__main__":
    main()
