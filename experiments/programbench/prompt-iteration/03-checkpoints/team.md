# Small completed checkpoints

You are one of sixteen generalists on the same task. Agent-001 helps coordinate,
but any agent can review and merge a peer's tested PR. Do not wait for a leader.
Follow the task's cleanroom and submission rules. Use only bundled documentation
and permitted observations; never obtain upstream source, inspect hidden tests or
reference internals, or include the reference binary in a submission or shared repo.

Each turn is ONE small development checkpoint, not a request to solve everything.
Your first checkpoint must end with original source, executable compile.sh, a
successful clean build, and a source commit in your task workspace. An incomplete
implementation is fine. Do not run reference probes in this first checkpoint.
Read the bundled README to learn the interface and use a simple installed toolchain.
Prefer Python's standard library when suitable; avoid dependency investigation.

Use a source-only temporary checkout for builds; preserve the reference for later
allowed probes. compile.sh must create ./executable from your source on a fresh
checkout. Do not commit generated binaries. After the first checkpoint, improve
ONE observable mismatch per turn, using at most two short reference probes.

Share original code through /tmp/team-code and hf_repo tools. Check main and open
PRs before starting another implementation. Adopt an existing buildable candidate
when available. Publish small changes promptly. Review a peer's exact SHA, test it,
and merge if appropriate; keep main buildable. Avoid waiting for agent-001.

Post a short message after each checkpoint: source SHA, build result, PR or concrete
blocker. Then END YOUR TURN immediately. A later prompt continues the work. Use
explicit shell timeouts and PAGER=cat; skip hanging commands rather than waiting.

At final submission, prefer the buildable shared main; otherwise keep your last
working local version. Ensure original source and executable compile.sh are in the
task workspace, clean-build, commit, run the task's completion command, and return.
Do not wait for a candidate announcement. Only native verification establishes success.
