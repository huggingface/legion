"""preflight.py: Hub errors become failed checks, and the archive path matches the world's."""
from __future__ import annotations

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError

from .test_run4_experiment import module

ENV = {"RUN3_BUCKET": "hf://buckets/ns/b/", "RUN": "run4b", "HF_TOKEN": "private-marker"}


def online(env, api):
    ok = httpx.Response(200, text="", request=httpx.Request("GET", "http://x"))
    rows = module("preflight").check(env, online=True, get=lambda *a, **k: ok, api=api)
    return next(row for row in rows if row["check"] == "hub_metadata")


class Api:
    def __init__(self, error=None):
        self.error, self.paths = error, []

    def repo_info(self, *args, **kwargs):
        if self.error:
            raise self.error

    def get_bucket_paths_info(self, bucket_id, paths):
        self.paths.append((bucket_id, paths))
        return []


def hub_error(status):
    request = httpx.Request("GET", "https://huggingface.co/api/datasets/x")
    try:  # huggingface_hub 2.x builds its errors on httpx2 responses
        import httpx2
        response = httpx2.Response(status, request=httpx2.Request("GET", str(request.url)))
    except ImportError:
        response = httpx.Response(status, request=request)
    return HfHubHTTPError("401 Unauthorized private-marker", response=response)


@pytest.mark.parametrize("error", [hub_error(401), RuntimeError("private-marker")])
def test_hub_errors_are_recorded_not_raised(error):
    row = online(ENV, Api(error))
    assert row["ok"] is False and "private-marker" not in str(row)


def test_bucket_path_follows_the_world_run_prefix():
    api = Api()
    assert online(ENV, api)["ok"]
    assert api.paths == [("ns/b", ["run4b/world.tar.gz"])]
    api = Api()
    assert online({**ENV, "CHATBOT_BUCKET": "hf://buckets/ns/other/custom"}, api)["ok"]
    assert api.paths == [("ns/other", ["custom/world.tar.gz"])]
    row = online({"RUN3_BUCKET": "hf://buckets/ns/b"}, Api())
    assert not row["ok"] and "RUN" in row["detail"]
