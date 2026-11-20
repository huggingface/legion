# escape-room

Four agents each solve their own copy of one AgentEscapeBench room (difficulty 20,
`agentescape-20-skeleton-0052d9e8`) and share findings on a local board. Each agent's room
is three Docker containers (Pi environment, puzzle server, execution worker); a win needs a
correct `escape submit-answer` within 500 puzzle actions. `prompts/team.md` tells the agents how
to work with the board.

**Needs**

- Docker with Compose on the machine that runs flock (`compute: local`, `backend: docker`;
  the rooms are multi-container, so HF Sandboxes cannot run them).
- `HF_TOKEN` with read access to the private dataset
  `HuggingFaceSwarms/AgentEscapeBench`. Git must be able to clone it, e.g.
  `hf auth login --add-to-git-credential`.
- A separate `HF_INFERENCE_TOKEN` with Inference Providers permission for your personal
  billing account. It enters the agents' containers. The spec omits `namespace`, so router
  requests use that key's default billing account without an org bill-to header.

**Cost**: inference only (the containers run locally). The model is Qwen3.8-27B on deepinfra
($0.2 / $2.5 per 1M input / output tokens); each agent has 200 model turns and the task's
20-minute agent timeout, so a run is roughly $2-6 at those caps, less when agents win early.
flock shows $0 (no `pricing` in the spec).

**Launch**

```sh
uv run flock launch experiments/escape-room/spec.yaml   # the loop sends each agent one start prompt
uv run flock status escape-room-1
```

The run ends on its own once every agent has finished its turn, and writes the bundle under
`.flock/` (`uv run flock stop escape-room-1 --delete --yes` ends it early).

Agents get flock's random names: the task and its grader do not depend on them.
