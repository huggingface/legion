---
title: run3 world
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

The run-3 chatbot world (`worlds/chatbot3` from huggingface/flock), served with
`flock world serve chatbot3:Chatbot3World` (World API v1). Every route needs the
`X-World-Token` header; the flock controller and its workers send it.

- Secrets: `CHATBOT_WORLD_TOKEN`, `HF_TOKEN` (bucket writes).
- Variables: `CHATBOT_BUCKET` (`hf://buckets/<ns>/<bucket>/<run>`: best/ mirror, workspace
  sync every `WS_SYNC_S` s, restore on restart), optional `CHATBOT_WS`, `WS_SYNC_S`.
- Knobs: `/app/world.yaml` (probes, `runs.parallel`: bot runs at once; 0 = the CPU count).
