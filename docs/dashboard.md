# Dashboard: the run on flock-overview

`dashboard:` puts the run on the flock-overview Space (its board next to its run viewer), the way
`report_to` puts a training run on a tracker. Metrics need nothing new: the controller already
mirrors `run.json`, `metrics/` and `rollups/` to `<bucket>/controller/<swarm>/`, which the
viewer reads.

```yaml
dashboard:
  registry: hf://buckets/HuggingFaceSwarms/swarmchat-runs   # one registry bucket per task family
  name: Levels (v11)                  # default: the swarm name
  title: Levels
  setup: Leads spawn managers, managers spawn workers.
  viewer: https://huggingfaceswarms-flock-viewer-v8.hf.space   # the default
  # optional, written as given: viewer_public, dataset, code_repo, spec_url
```

`flock init` writes (and on a re-run rewrites) `<registry>/runs/<swarm>.json`: one row in the
overview's `experiments.json` format. `id` is the swarm, `board` the comms url (`board_id` the
part after `/boards/`, `board_public` the Space page with `board: space`; `viewer_public` the viewer's Space page), `viewer_run`
`<bucket ns>/<bucket>/<swarm>`. No token or invite goes in it. A failed write fails init.

Refused: `dashboard` with `compute: local` or without `bucket` (spec check); in init, a `board`
or `viewer` that is not an `https://huggingfaceswarms-*.hf.space` url (the overview sends its
HF token there and drops other rows) and a bucket outside the registry's namespace.
