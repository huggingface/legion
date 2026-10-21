"""Side-by-side replay core (progress/app.js): two logs at once, aligned by wall time since first
paint or by % done, give the service's frames at the matching log positions (needs node)."""

from __future__ import annotations

import bisect
import gzip
import json
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from conftest import make_target

from canvas.config import Settings
from canvas.world import World

APP = Path(__file__).resolve().parent.parent / "progress" / "app.js"

JS = """
const PX = require(process.argv[2]); const fs = require('fs');
(async () => {
  const [w, h] = [12, 10];
  const spec = JSON.parse(process.argv[4]);
  const out = [];
  for (const [i, path] of process.argv[3].split(',').entries()) {
    const log = PX.prepare(await PX.gunzipText(fs.readFileSync(path)), w, spec.work[i]);
    const r = new PX.Replayer(log, w, h, [255, 255, 255], 7, 2);
    const res = { dur: PX.duration(log), time: {}, pct: {} };
    for (const s of spec.secs) {
      const k = PX.indexAtTime(log, s);
      res.time[s] = { k, f: Array.from(r.frame(k)), a: Array.from(r.agentFrame(k)), ro: PX.readout(log, k) };
    }
    for (const p of spec.pcts) {
      const k = PX.indexAtDone(log, p);
      res.pct[p] = { k, f: Array.from(r.frame(k)) };
    }
    out.push(res);
  }
  console.log(JSON.stringify(out));
})();
"""


def _run(tmp_path, name, agents, gap, seed):
    p = tmp_path / f"{name}.png"
    make_target(p, w=12, h=10)
    w = World(Settings(target=p, claim_order="scattered", claim_seed=seed))
    t0, n = 1_000_000.0, 0
    rng = random.Random(seed)
    while True:
        a = f"{name}-{n % agents:04d}"
        cl = w.claim(a, 3, now=t0 + n * gap)
        if not cl["pixels"]:
            break
        px = [(q["x"], q["y"], *bytes.fromhex(q["hex"])) for q in cl["pixels"]]
        if rng.random() < 0.1:  # an occasional wrong pixel, repainted later
            px[0] = (px[0][0], px[0][1], 1, 2, 3)
        w.paint(a, px, now=t0 + n * gap)
        n += 1
    log = tmp_path / f"{name}.jsonl.gz"
    log.write_bytes(gzip.compress(("\n".join(ln for _, ln in w._memlog) + "\n").encode()))
    return w, log


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_two_logs_aligned_by_time_and_done(tmp_path):
    slow, log_a = _run(tmp_path, "run1", agents=3, gap=2.0, seed=1)  # long run
    fast, log_b = _run(tmp_path, "run1k", agents=9, gap=0.25, seed=2)  # short run
    secs = [0, 1, 3, 7.5, 20, 1e6]
    pcts = [0, 0.25, 0.5, 0.999, 1.0]
    script = tmp_path / "sbs.js"
    script.write_text(JS)
    spec = {"secs": secs, "pcts": pcts, "work": [slow.work_total, fast.work_total]}
    r = subprocess.run(
        ["node", str(script), str(APP), f"{log_a},{log_b}", json.dumps(spec)],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    out = json.loads(r.stdout)
    for w, res in zip((slow, fast), out, strict=True):
        ts = list(w.r_ts.view()[1:])
        done = np.cumsum(
            [
                sum(o == "useful" for o in _outs(w, i))
                - sum(o == "overwritten" for o in _outs(w, i))
                for i in range(w.seq)
            ]
        )
        for s in secs:
            k = bisect.bisect_right(ts, ts[0] + s)  # entries with ts <= first paint + s
            got = res["time"][str(s) if s != 1e6 else "1000000"]
            assert got["k"] == k, s
            assert np.array_equal(
                np.array(got["f"], np.uint8).reshape(10, 12, 4)[..., :3], w.frame(k)
            )
            assert np.array_equal(
                np.array(got["a"], np.uint8).reshape(10, 12, 4)[..., :3], w.replay_agents(k)
            )
        assert res["time"]["1000000"]["k"] == w.seq  # the shorter run freezes at its last frame
        for p in pcts:
            want = int(np.ceil(p * w.work_total))
            k = 0 if want <= 0 else int(np.argmax(np.maximum.accumulate(done) >= want)) + 1
            got = res["pct"][str(p) if p not in (0, 1.0) else str(int(p))]
            assert got["k"] == k, p
            assert np.array_equal(
                np.array(got["f"], np.uint8).reshape(10, 12, 4)[..., :3], w.frame(k)
            )
        end = res["time"]["1000000"]["ro"]
        assert end["done"] == pytest.approx(100.0) and end["active"] >= 1


def _outs(w, i):
    return [p[3] for p in json.loads(w._memlog[i][1])["px"]]
