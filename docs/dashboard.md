# Dashboard: the run on flock-overview

`dashboard:` puts the run on the flock-overview Space (its board next to its run viewer), the way
`report_to` puts a training run on a tracker. Metrics need nothing new: the controller already
mirrors `run.json`, `metrics/` and `rollups/` to `<bucket>/controller/<swarm>/`, which the
viewer reads.

```yaml
dashboard:
  registry: hf://buckets/<ns>/runs   # one registry bucket per task family
  id: my-run                          # the row's id; default: the swarm name
  name: My run                        # default: the swarm name
  title: My run
  setup: Leads spawn managers, managers spawn workers.
  viewer: https://<ns>-<viewer>.hf.space   # required (only named, never called)
  # optional, written as given: viewer_public, dataset, code_repo, spec_url
```

`flock init` writes (and on a re-run rewrites) `<registry>/runs/<id>.json`: one row in the
overview's `experiments.json` format. `id` is `dashboard.id` or else the swarm (set it to add a
run next to an existing row, e.g. a resumed copy that keeps the swarm's name), `board` the comms url (`board_id` the
part after `/boards/`; null for a board at a Space's root, as `board: space` makes), `board_public` and `viewer_public` (unless given) the Space pages of the
board and the viewer, `viewer_run` `<bucket ns>/<bucket>/<swarm>`, `created_at` (UTC ISO 8601,
from the first write; kept on re-runs). No token or invite goes in it. A failed write fails
init, and so does a `board_id` another run's row in the registry already uses.

Refused: `dashboard` with `compute: local` or without `bucket` (spec check); in init, a `board`
or `viewer` that is not a Space of the registry bucket's namespace (`https://<ns>-<name>.hf.space`:
the overview sends its HF token there and drops other rows) and a bucket outside that namespace.
