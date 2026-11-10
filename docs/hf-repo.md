# Share original code through HF pull requests

The optional `hf_repo` resource creates a **private Hugging Face dataset repository
for each run**. Agents work in separate Git checkouts and share original source
through native Hub PRs. This does not change the task dataset or share `/workspace`.

```yaml
resources:
  message_board: {}
  hf_repo:
    namespace: edbeeching
    token_env: HF_TOKEN
tools:
  - message_board.*
  - hf_repo.*
```

Set `namespace` to an account or organization where the selected token can create
private datasets and write commits/PRs. Git must be installed in the task image.
The launcher packages the Python client; agents do not install it from the network.
All agents receive the same repository credential. Use a suitably scoped token;
reviews and agent attribution are conventions, not an authorization boundary.
Git URLs and exported metadata do not contain tokens.

Repositories are named `flock-<run-id>` (long names are shortened with a hash).
Their identity is saved in resource state; repeated startup of the same run reuses
that repository. A name collision without saved state fails instead of adopting an
unrelated repository. Initialization creates only README and ignore rules. No task
files are automatically uploaded. The dataset remains private and is retained
when the run ends, including failures. Results under `resources/hf_repo/` contain
its URL, final main SHA and PR summaries. Remote history remains on the Hub.

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

See [the collaboration example](../examples/programbench.yaml).
Agents publish only original source, their own tests and build scripts. Never share
the provided reference executable, hidden tests, upstream source, credentials or
large artifacts. Git LFS upload/download is outside this tool's scope.

A useful division is parser, formatting, CLI/error handling, and integration.
Announce ownership and PR numbers on the board. Each agent copies chosen original
source back into its own task workspace, runs its own `compile.sh`, tests and
commits its submission there. A merged collaboration PR is **not** task completion.
