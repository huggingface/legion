---
title: flock pixel runs
colorFrom: gray
colorTo: blue
sdk: static
pinned: false
---

# flock pixel progress page

Static, dependency-free page (`index.html` + `app.js`) that shows pixel runs. It is published as
the private Space `cmpatino/flock-pixel`. This file is also that Space's card, which is why it starts
with front matter.

## What it shows

- **Run selector.** Runs come from `runs.json`; the latest is shown by default, and `?run=<id>`
  deep-links to one.
- **Live view**, while a run is `running`. Every 10 s the page reloads `{canvas_url}/canvas.png`
  and fetches `{canvas_url}/score` and `/status`. It shows done %, exact %, pixels left, MSE/PSNR,
  outcomes, req/s, claims outstanding, active agents and the top 10 agents by useful pixels.
  The canvas sends `Access-Control-Allow-Origin: *` on GET routes only.
- **View selector:** canvas, painted mask, or agent map (every painted pixel in its last
  painter's colour; live from `/agents.png`, in the replay from the log, same colours as the
  server: `agentColour` mirrors `canvas/world.py::agent_colour`, tested).
- **Compare tab** (`?tab=compare`):
  - Pick two runs; the latest two are the default.
  - Overlaid series on minutes since start: px/s, paint calls/s, active agents, idle %, step gap
    p50, and model p50 when the driver records it.
  - A summary table: duration, px/s mean and peak, paint calls/s, agents, GPUs, cost.
  - Final canvases and agent maps side by side, from `final/<id>/agents.png`.
- **Side by side tab** (`?tab=sbs`): the replay of two runs at once.
  - Left and right runs are picked from `runs.json`; the default is run1-20261019 vs run1k-20261019.
  - One scrubber drives both panels. Alignment is either wall time since each run's first paint
    (the default; the slider spans the longer run, and the shorter one stays on its final frame),
    or % done.
  - Play at 1x, 10x or 60x, an agent-map toggle for both panels, and "show target instead" on
    the right (`target.png`, shipped with the page).
  - A readout under each panel: time since first paint, done %, px/s over the last 10 s of log
    time, and agents with a paint in the last 10 s.
  - Both logs are decoded in the browser (`DecompressionStream`) and reduced to typed arrays.
    Frames use the same keyframe scheme (every 2000 entries, LRU of 4 per panel).
  - Measured in node on the two real 62,278-entry logs (4.9 MB gz each): both parsed in 1.2 s, a
    first frame in 8 ms, an agent-map frame in ~12 ms, 31 MB heap plus 58 MB typed arrays retained.
  - The agent map comes from the log, so it works for run 1 too, which has no `agents.png`.
  - `tests/test_sbs_js.py` checks both panels' frames and agent maps against the service, in
    both alignment modes.
- **Painted mask:**
  - Live view: lays the canvas's `/mask.png?style=overlay` (magenta = painted, transparent elsewhere) over the dimmed live image. The server renders it, so the browser does no per-pixel work. It refreshes every 10 s with the canvas.
  - Replay: the same mask at position t, computed from the log with the same keyframe scheme.
- **Time series** from `runs/<id>/progress.jsonl`, drawn with canvas 2D (no library):
  - done % and exact %;
  - canvas req/s and working agents;
  - the latest flock numbers: agents by state, turns, pending prompts, cost.
- **Replay** once `final/<id>/` exists. The page downloads `paint-log.jsonl.gz` and gunzips it
  with `DecompressionStream`. It builds the frame at any seq from keyframes (every 2000 entries,
  LRU of 6) plus the delta, and offers a slider and play. This is the same fold as the service;
  `tests/test_progress_js.py` checks that the frames match.
- **Links:** controller status page, the raw canvas endpoints, the board thread, and the final
  files.

## Files the driver writes to the Space

| path | when |
|---|---|
| `runs.json` | start, every 60 s, end: `[{id, name, swarm, started_at, finished_at, status, canvas_url, controller_url, board_url, final?}]` |
| `runs/<id>/progress.jsonl` | every 60 s: one row per driver tick (`ts, elapsed_s, done_pct, exact_pct, mse, psnr, req_s, claims_outstanding, agents_by_state, turns, pending_prompts, flock_cost_usd, gpu_usd_est, ...`) |
| `final/<id>/canvas.png`, `score.json` (+ `target` info), `paint-log.jsonl.gz` | once at the end; the run becomes `finished` (or `stopped: ...`) |

Each upload is one commit with `huggingface_hub.HfApi.create_commit`, about 60 commits per hour.
An upload failure is printed and the run carries on.

## Driver settings

```sh
export HF_TOKEN=...            # write access to the Space (never printed)
export SPACE_ID=cmpatino/flock-pixel
export FLOCK_URL=... FLOCK_TOKEN=...      # controller API
export FLOCK_PUBLIC_URL=...    # controller status page link shown on the page (default FLOCK_URL)
export BOARD_URL=...           # board thread link (optional)
python experiments/pixel/run/driver.py --swarm pixel-100 --canvas-url <public 8765 URL> \
  --run-id pixel-100-a --run-name "run 1: 100 x LFM2.5-2.6B"
```

The driver needs `huggingface_hub` for the uploads; everything else is stdlib.
`--publish-dir DIR` writes the same files locally instead of, or as well as, uploading them.
To preview, serve that directory with `python -m http.server`.

To republish the page itself: `hf upload cmpatino/flock-pixel experiments/pixel/progress . --repo-type space`.
Leave `runs.json` out of that upload once runs exist (`--exclude runs.json`), so the live list is not reset.
