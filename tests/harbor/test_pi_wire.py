"""Optional native Pi regression: FLOCK_PI_CLI points to its installed CLI JS."""

import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from flock.harbor.harnesses.pi import request_extension
from flock.models import Inference


@pytest.mark.skipif(
    not os.environ.get("FLOCK_PI_CLI"), reason="native Pi not configured"
)
@pytest.mark.parametrize(
    "extension, expected_role, thinking, extra_body", [
        (False, "developer", "low", {}),
        (True, "system", "low", {}),
        (True, "system", "off", {"chat_template_kwargs": {"enable_thinking": False}}),
        (True, "system", "off", {"reasoning": {"enabled": False}}),
    ]
)
def test_native_pi_reasoning_request(tmp_path, extension, expected_role, thinking, extra_body):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            requests.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunk = {
                "id": "test",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "OK"},
                        "finish_reason": "stop",
                    }
                ],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = tmp_path / "config"
        config.mkdir()
        (config / "models.json").write_text(
            json.dumps(
                {
                    "providers": {
                        "harbor-endpoint": {
                            "baseUrl": f"http://127.0.0.1:{server.server_port}/v1",
                            "apiKey": "fake-key",
                            "api": "openai-completions",
                            "models": [{"id": "test-model", "reasoning": thinking != "off"}],
                        }
                    }
                }
            )
        )
        if extension:
            extensions = config / "extensions"
            extensions.mkdir()
            (extensions / "flock-request.js").write_text(request_extension(
                Inference(reasoning_effort=thinking, extra_body=extra_body)))
        result = subprocess.run(
            [
                "node",
                os.environ["FLOCK_PI_CLI"],
                "--print",
                "--mode",
                "json",
                "--no-session",
                "--no-tools",
                "--provider",
                "harbor-endpoint",
                "--model",
                "test-model",
                "--thinking",
                thinking,
                "Reply OK.",
            ],
            cwd=tmp_path,
            env={
                "PATH": os.environ["PATH"],
                "HOME": str(tmp_path),
                "PI_CODING_AGENT_DIR": str(config),
            },
            capture_output=True,
            text=True,
            timeout=45,
        )
        assert result.returncode == 0, result.stderr + result.stdout
        assert len(requests) == 1
        payload = requests[0]
        assert payload["messages"][0]["role"] == expected_role
        assert payload["messages"][-1] == {
            "role": "user",
            "content": [{"type": "text", "text": "Reply OK."}],
        }
        if thinking == "low":
            assert payload["reasoning_effort"] == "low"
        assert all(payload[key] == value for key, value in extra_body.items())
        events = [
            json.loads(line)
            for line in result.stdout.splitlines()
            if line.startswith("{")
        ]
        messages = [
            event["message"] for event in events if event["type"] == "message_end"
        ]
        assert messages[-1]["content"] == [{"type": "text", "text": "OK"}]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
