"""``python -m canvas`` : run the service with uvicorn (CANVAS_HOST, CANVAS_PORT)."""

import os

import uvicorn

from .app import create_app


def main() -> None:
    host = os.environ.get("CANVAS_HOST", "0.0.0.0")
    port = int(os.environ.get("CANVAS_PORT", "8765"))
    uvicorn.run(
        create_app(),
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
        timeout_keep_alive=30,
    )


if __name__ == "__main__":
    main()
