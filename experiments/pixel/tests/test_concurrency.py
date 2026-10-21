"""200 agents painting at once through the real ASGI stack: no lost writes, consistent score."""

from __future__ import annotations

import json
import threading

import numpy as np
from conftest import H, hx, make_target
from fastapi.testclient import TestClient

from canvas.app import create_app
from canvas.config import Settings

N_AGENTS = 200
CALLS = 3
S = 96  # side: 200 agents x 27 px fit in the non-transparent interior


def test_200_agents_concurrent_paint(tmp_path):
    p = tmp_path / "t.png"
    t = make_target(p, w=S, h=S, seed=3)
    app = create_app(Settings(target=p, rate_limit=0, data_dir=tmp_path / "data"))
    world = app.state.world
    hot = (5, 5)  # every agent also paints this one pixel correctly: exactly one is useful
    # Disjoint pixel sets: agent i owns flat indices i*9*CALLS ... (9 own + 1 hot per call).
    own = {}
    flat = np.arange(1, S * S)
    flat = flat[(flat % S != 0) & (flat // S != 0) & (flat != hot[1] * S + hot[0])]
    for i in range(N_AGENTS):
        own[i] = flat[i * 9 * CALLS : (i + 1) * 9 * CALLS]
    assert len(own[N_AGENTS - 1]) == 9 * CALLS
    errors = []
    barrier = threading.Barrier(N_AGENTS)

    with TestClient(app) as c:

        def agent(i):
            try:
                barrier.wait()
                for k in range(CALLS):
                    ids = own[i][k * 9 : (k + 1) * 9]
                    pts = [
                        {"x": int(j % S), "y": int(j // S), "hex": hx(t[j // S, j % S])}
                        for j in ids
                    ]
                    pts.append({"x": hot[0], "y": hot[1], "hex": hx(t[hot[1], hot[0]])})
                    c.post("/claim", json={"n": 2}, headers=H(f"ag{i}"))
                    r = c.post("/paint", json={"pixels": pts, "turn": str(k)}, headers=H(f"ag{i}"))
                    if r.status_code != 200:
                        errors.append(r.text)
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))

        ths = [threading.Thread(target=agent, args=(i,)) for i in range(N_AGENTS)]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        assert not errors, errors[:3]
        s = c.get("/score").json()
        log = c.get("/log", params={"limit": 5000}).text.splitlines()

    total_calls = N_AGENTS * CALLS
    assert s["seq"] == total_calls
    assert len(log) == total_calls
    seqs = sorted(json.loads(ln)["seq"] for ln in log)
    assert seqs == list(range(1, total_calls + 1))
    # No lost writes: every owned pixel and the hot pixel now match the target.
    painted = np.concatenate(list(own.values()))
    cf = world.canvas.reshape(-1, 3)
    tf = t.reshape(-1, 3)
    assert np.array_equal(cf[painted], tf[painted])
    assert (cf[hot[1] * S + hot[0]] == tf[hot[1] * S + hot[0]]).all()
    # Consistent accounting.
    assert s["paints"]["useful"] == len(painted) + 1
    assert s["paints"]["duplicate"] == total_calls - 1
    assert s["paints"]["wrong"] == 0 and s["paints"]["overwritten"] == 0
    # Done = right colour AND (painted, or not work): white work pixels on the white blank
    # canvas start "not done" even though the colours agree.
    done = (cf == tf).all(-1) & (world.painted | ~world.work_mask)
    assert s["matched"] == int(done.sum())
    assert sum(a["pixels"] for a in s["agents"].values()) == total_calls * 10
    assert len(s["agents"]) == N_AGENTS
    # Replay of the final position equals the live canvas.
    assert np.array_equal(world.frame(total_calls), world.canvas)
