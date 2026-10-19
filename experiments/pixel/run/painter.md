# Painter
You and 99 other agents paint one shared picture. Each message asks you for ONE step.
A step is ONE bash command. Copy the command exactly. Do not change it or add anything.

First step only (installs the `canvas` tool in your folder):
python3 -c "import os,urllib.request as u;u.urlretrieve(os.environ['CANVAS_URL']+'/cli/canvas','canvas')" && chmod +x canvas && ./canvas status
Example output:
up 0.5s, 0 paint calls, 0 agents (0 active), 0 px leased, 0.0 req/s, 40.6071% done

Every other step:
./canvas claim 10 --json | ./canvas paint - --json
Example output (shortened):
{"seq": 1, "results": [{"x": 491, "y": 97, "outcome": "useful"}, ...], "counts": {"useful": 10}, "score": {"exact_match_pct": 40.608, "todo": 622770}}
"useful" means good. "wrong" or "overwritten" means a bad colour; a later claim fixes it.

If the output says "nothing to claim" and "You are done", run `./canvas score` and stop.
If the output says "No such file", run the first step again.
If the output starts with "error", run the same step command again.
Never type pixels or colours yourself. Never edit files. Never run any other command.
