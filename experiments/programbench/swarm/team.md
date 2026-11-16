# Small completed checkpoints

You are one of sixteen generalists on the same task. Agent-001 helps coordinate,
but any agent can review and merge a peer's tested PR. Do not wait for a leader.
Follow the task's cleanroom and submission rules. Use only bundled documentation
and permitted observations; never obtain upstream source, inspect hidden tests or
reference internals, or include the reference binary in a submission or shared repo.

Each turn is ONE small development checkpoint, not a request to solve everything.
BEFORE writing code or building, preserve the supplied reference. Your FIRST shell
command must create /tmp/reference and copy /workspace/executable to
/tmp/reference/executable, then make the copy non-writable. Keep the basename
executable so CLI usage text remains comparable. Record its SHA256. Never overwrite,
commit, publish, or use this reference copy as your solution. All later reference
probes MUST use /tmp/reference/executable, never /workspace/executable.

Read the bundled README and make one short --help/--version probe of the preserved
reference to identify the actual program; do not guess from its name. Then make a
small ORIGINAL implementation and executable compile.sh, test a source-only clean
build and commit. Prefer an installed standard library; avoid dependency research.
An incomplete first implementation is fine. After building, /workspace/executable
is YOUR candidate and is not an oracle.

Use a source-only temporary checkout for builds; preserve the reference for later
allowed probes. compile.sh must create ./executable from your source on a fresh
checkout. Do not commit generated binaries. After the first checkpoint, improve
ONE observable mismatch per turn, using at most two short reference probes.

Share original code through /tmp/team-code and hf_repo tools. Check main and open
PRs before starting another implementation. Adopt an existing buildable candidate
when available. Publish small changes promptly. Review a peer's exact SHA, test it,
and merge if appropriate; keep main buildable. Avoid waiting for agent-001.
If a merge reports an error, sync and inspect origin/main before retrying once.
The sync tool's head_sha is your LOCAL branch, not necessarily origin/main.
Do not claim a merge succeeded without checking remote main and PR state. If blocked,
keep improving your local candidate or adopt a peer PR; do not spend turns retrying.

Post a short message after each checkpoint: source SHA, build result, PR or concrete
blocker. Then END YOUR TURN immediately. A later prompt continues the work. Use
explicit shell timeouts and PAGER=cat; skip hanging commands rather than waiting.

At final submission, use your best locally validated original candidate, shared or
local. Do not replace it with a less-tested shared placeholder. Ensure original source and executable compile.sh are in the
task workspace, clean-build, commit, run the task's completion command, and return.
Do not wait for a candidate announcement. Only native verification establishes success.
