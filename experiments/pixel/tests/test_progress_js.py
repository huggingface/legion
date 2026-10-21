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
from canvas.world import World, agent_colour, root_of

APP = Path(__file__).resolve().parent.parent / "progress" / "app.js"

JS = """
const PX = require(process.argv[2]); const fs = require('fs');
(async () => {
  const text = await PX.gunzipText(fs.readFileSync(process.argv[3]));
  const [w, h] = [+process.argv[4], +process.argv[5]];
  const r = new PX.Replayer(PX.index(PX.parseLog(text), w), w, h, [255, 255, 255], 7, 2);
  const out = {};
  for (const t of JSON.parse(process.argv[6]))
    out[t] = { f: Array.from(r.frame(t)), m: Array.from(r.maskFrame(t)), a: Array.from(r.agentFrame(t)) };
  out.colours = JSON.parse(process.argv[7]).map(PX.agentColour);
  out.roots = JSON.parse(process.argv[7]).map(PX.rootOf);
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
    names = [f"painter-{c}-{i:04d}" for c in "abc" for i in range(334)] + ["ünï"]
    names += [
        f"leader-{i:04d}-painter-{c}-{j:04d}" for i in range(3) for c in "ab" for j in range(3)
    ]
    r = subprocess.run(
        ["node", str(script), str(APP), str(log), "12", "10", json.dumps(ts), json.dumps(names)],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    frames = json.loads(r.stdout)
    for t in ts:
        rgba = np.array(frames[str(t)]["f"], dtype=np.uint8).reshape(10, 12, 4)
        assert np.array_equal(rgba[..., :3], w.frame(t)), t
        mask = np.array(frames[str(t)]["m"], dtype=bool).reshape(10, 12)
        assert np.array_equal(mask, w.replay_mask(t)), t
        amap = np.array(frames[str(t)]["a"], dtype=np.uint8).reshape(10, 12, 4)
        assert np.array_equal(amap[..., :3], w.replay_agents(t)), t
    assert [tuple(c) for c in frames["colours"]] == [agent_colour(n) for n in names]
    assert frames["roots"] == [root_of(n) for n in names]
