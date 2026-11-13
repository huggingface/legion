# Collaborating on ProgramBench through HF PRs

Follow every task cleanroom rule. Observe the reference only through its normal
interface; never disassemble, trace, copy or publish it. Never retrieve upstream
source or hidden tests. Share only your team's original code and observations.

Your task workspace and submission are independent. Clone the collaboration repo
with hf_repo.clone into /tmp/team-code, not /workspace. Create a working branch with git switch -c agents/<your-name>/work before editing;
keep local main unchanged so it can fast-forward after Hub squash merges.
Never copy the whole task
workspace into the clone. Publish selected original source files and build scripts.

Read the general board, claim one component, and post your intended interface.
Build a minimal working implementation early, before exhaustive behavioral probes.
Use bounded shell timeouts for probes, avoid interactive pagers and huge outputs,
and reserve the final third of your time for integration, compilation and testing.

Use hf_repo.commit with explicit files, then hf_repo.open_pr. Post the PR number,
component interface and test evidence to the board. Use hf_repo.update_pr for
follow-up commits; recover partially published PRs instead of opening duplicates.
Inspect and test peers' PRs using hf_repo.inspect_pr and hf_repo.checkout_pr.
Reviews record the exact head SHA and are advisory. Recheck changes before merging.
Pass your clone path to hf_repo.inspect_pr. Use hf_repo.sync to fetch origin/main,
then merge origin/main in your work branch before opening a PR. Never rebase a
published PR. Shell Git has no credentials. Sync fast-forwards main only when checked out.

After each implemented component, check the board for peer PRs and integrate useful
work. Copy selected original code back into /workspace. Every agent must produce
an executable /workspace/compile.sh that builds its own original executable from a
fresh checkout, commit its task changes, and follow the task's final submission
instructions. Publishing or merging a PR does not submit your task. Never claim
benchmark success solely from local comparisons.
