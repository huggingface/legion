# Share original code through HF pull requests

The optional `hf_repo` resource connects agents to an existing **private Hugging
Face dataset repository**. Every worker uses the same repository, including workers
added or restarted later. Agents keep separate checkouts and use native Hub PRs.

Create a private dataset on the Hub and commit an initial README to its `main`
branch before launching. Set `HF_REPO_TOKEN` to a token with read/write access to
that dataset, then configure:

```yaml
resources:
  hf_repo:
    repo_id: your-org/team-code
    token_env: HF_REPO_TOKEN
tools: [hf_repo.*]
```

Use a different dataset for each independent experiment. The resource never
creates, initializes, deletes, or changes the visibility of repositories. A missing,
inaccessible, empty, or public dataset fails startup. Stopping a worker retains
all source and PR history. Each worker exports the repository URL, main SHA, and
PR summaries under its resource results.

The launcher forwards the selected credential to workers; each agent also receives
it for Git and Hub calls. Use a token scoped to the collaboration dataset, separate
from the worker infrastructure token and model inference credential. All agents
share this repository credential: reviews and attribution are conventions, not
access controls. Git URLs and exported metadata contain no tokens.

Git must be installed in the task image. The launcher packages the Python client;
agents do not install it from the network. The resource allows `huggingface.co`
for agent requests. To use a message board too, configure `message_board` and
include `message_board.*` in the explicit tools list.

## Agent workflow

Use a separate directory such as `/tmp/team-code`:

```sh
flock-tool hf_repo.clone --json '{"path":"/tmp/team-code"}'
git -C /tmp/team-code switch -c agents/my-component
# Copy only your original source into this checkout, then:
flock-tool hf_repo.commit --json '{"path":"/tmp/team-code","files":["parser.py"],"message":"Implement CSV parser"}'
flock-tool hf_repo.open_pr --json '{"path":"/tmp/team-code","title":"Implement CSV parser","description":"Original parser; tested quoted fields and empty input."}'
flock-tool hf_repo.list_prs --json '{"state":"open"}'
flock-tool hf_repo.inspect_pr --json '{"pr":1}'
flock-tool hf_repo.checkout_pr --json '{"path":"/tmp/team-code","pr":1}'
# Test the checkout, then supply the full head_sha returned by inspect/checkout:
flock-tool hf_repo.review_pr --json '{"pr":1,"head_sha":"<40-character-SHA>","verdict":"approve","comment":"Checked quoted fields locally."}'
flock-tool hf_repo.merge_pr --json '{"pr":1}'
# Switch to main explicitly in your shell, preserving local work, then:
flock-tool hf_repo.sync --json '{"path":"/tmp/team-code"}'
```

`status` and `diff` inspect local changes; `update_pr(path, pr)` publishes additional
commits. Use normal Git in the shell for branches, rebases and conflict resolution.
Always develop on a working branch, leaving local main unchanged: Hub merges
squash commits, so committing directly on local main makes it diverge from the
merged remote main. Switch back to main before syncing; keep your work branch.
`commit` stages only individually named files, including deletions; it refuses an
already populated index, directories, paths outside the checkout, and obvious
reference executable/environment files. These checks prevent accidents, not
malicious bypass through the shell. Untracked files are not included in `diff`.

Review comments record agent name, verdict and reviewed SHA. They are advisory:
merge does not require approval, and a later update can invalidate an earlier
review. Agents share one HF account, so the native PR author is that account;
the PR description/review records the Flock agent. `inspect_pr` bounds the diff
and comments; use the checkout for a full review. Native Hub merge handles
conflicts; the tool never resolves them automatically.

PR updates must descend from the observed remote head and use a push lease. If
another agent advances the PR, fetch and integrate that work before retrying. No
automatic force-reset or rebase is performed. If PR creation succeeds but publication
fails, `open_pr` returns `published: false`, its PR number and `next_action:
update_pr`; the number is also stored in `git config flock.lastPr`. Recover that
PR rather than creating a duplicate. If the creation request itself times out,
inspect `list_prs(state="all")` on the Hub before trying again. Comments and PR
creation are not blindly retried by the wrapper.

## ProgramBench

See [the collaboration example](../experiments/programbench/README.md).
Agents publish only original source, their own tests and build scripts. Never share
the provided reference executable, hidden tests, upstream source, credentials or
large artifacts. Git LFS upload/download is outside this tool's scope.

A useful division is parser, formatting, CLI/error handling, and integration.
Announce ownership and PR numbers on the board. Each agent copies chosen original
source back into its own task workspace, runs its own `compile.sh`, tests and
commits its submission there. A merged collaboration PR is **not** task completion.
