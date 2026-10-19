"""progress/app.js replays the gzipped paint log to the same frames as the service (needs node)."""

from __future__ import annotations

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
  const text = await PX.gunzipText(fs.readFileSync(process.argv[3]));
  const [w, h] = [+process.argv[4], +process.argv[5]];
  const r = new PX.Replayer(PX.index(PX.parseLog(text), w), w, h, [0, 0, 0], 7, 2);
  const out = {};
  for (const t of JSON.parse(process.argv[6])) out[t] = Array.from(r.frame(t));
  console.log(JSON.stringify(out));
})();
"""


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_js_replay_matches_service(tmp_path):
    p = tmp_path / "t.png"
    make_target(p, w=12, h=10)
    w = World(Settings(target=p, claim_mode="enforced"))
    rng = random.Random(1)
    w.claim("holder", 3)  # enforced: some paints are refused and must not be replayed
    for i in range(60):
        px = [
            (rng.randrange(12), rng.randrange(10), *(rng.randrange(256) for _ in range(3)))
            for _ in range(rng.randrange(1, 5))
        ]
        w.paint(f"a{i % 3}", px)
    assert w.totals["refused"] > 0
    log = tmp_path / "log.jsonl.gz"
    log.write_bytes(gzip.compress(("\n".join(ln for _, ln in w._memlog) + "\n").encode()))
    script = tmp_path / "check.js"
    script.write_text(JS)
    ts = [0, 1, 6, 7, 8, 30, 59, 60]
    r = subprocess.run(
        ["node", str(script), str(APP), str(log), "12", "10", json.dumps(ts)],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    frames = json.loads(r.stdout)
    for t in ts:
        rgba = np.array(frames[str(t)], dtype=np.uint8).reshape(10, 12, 4)
        assert np.array_equal(rgba[..., :3], w.frame(t)), t
