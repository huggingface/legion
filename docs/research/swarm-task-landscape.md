# Open-source task landscape for Flock swarms

Research date: 2026-10-29. Scope: long-running, tool-using tasks, spanning eight sectors.
This is a source review, not an installation test or a benchmark result. Resource and
integration estimates below are our judgments; upstream facts are linked at their point of use.

## Recommendation

Start with **Terminal-Bench 2.0 and CORE-Bench**, then add **TheAgentCompany and BixBench**.
Together they test troubleshooting, research reproduction, organizational workflows, and
biological data analysis. Add **SecondState FAB** for document-heavy finance, **DiscoveryWorld**
for experimental reasoning, **MLE-bench** for iterative optimization, and **Overcooked-AI** as
a controlled coordination diagnostic. These are eight different capabilities, not eight
interchangeable benchmark scores.

For the first deployment, favor tasks executable on CPUs against the existing inference router.
Reserve task-training GPUs for an explicit MLE-bench expansion. Keep games as a small diagnostic
slice: a coordination gain in a kitchen simulation does not establish a gain in scientific work.
Factorio is a promising later shared-world experiment, but its underlying game is not open-source.

## 1. What to borrow from Avalon-M1

[Avalon-M1, §2.3, Figure 4 and Appendix Table 7](https://www.gr.inc/avalonm1.pdf)
reports 147 environments and 2,201,439 available tasks. Most environments were built in-house.
The two distributions below measure different things; task sampling changes with curriculum.

| Sector | Environments | Inventory share | Available tasks | Sampled-task share, Figure 4 |
| --- | ---: | ---: | ---: | ---: |
| Software & IT | 48 | 32.7% | 91,483 | 42% |
| Science & education | 22 | 15.0% | 1,864,593 | 10% |
| Financial services | 21 | 14.3% | 10,116 | 17% |
| Life sciences & healthcare | 16 | 10.9% | 20,552 | 8% |
| Information & media | 15 | 10.2% | 198,425 | 9% |
| Recreation & games | 13 | 8.8% | 5,452 | 6% |
| Professional services | 8 | 5.4% | 10,619 | 5% |
| Industrials & infrastructure | 4 | 2.7% | 199 | 3% |

The paper names open-source inputs including SWE-rebench V2 and Endless Terminals, but its full
collection is not established as downloadable. Our proposal borrows sector diversity and
budget-dependent evaluation, not the original training mixture.

### OpenReward is a discovery route, not an openness guarantee

The [OpenReward registry](https://www.openreward.ai/environments) lists ports of known benchmarks
and additional environments including KellyBench, portfolio, FPL, and toolathlon-gym. The latter
advertises 503 tasks with local PostgreSQL rather than external APIs. A listing alone does not
verify source licensing, downloadable state, evaluator availability, or self-hosting. Before
adopting an entry, require all four. Do not count a CORE-Bench or SWE-rebench port as a new task
family. Registry totals also change independently of the paper's inventory.

## 2. How to read this survey

There are **27 candidates**, numbered once even when they cover several sectors.

- **Local**: upstream documents local execution. It does not mean we have run it here.
- **Conditional**: execution or reuse has material extra terms, services, or artifact gates.
- **Builder**: a simulator/generator needing a frozen task suite or an LLM-facing interface.
- **Hold**: unresolved licensing or maintenance makes it a poor initial dependency.

Code licenses were checked through GitHub metadata, with custom/undetected licenses inspected
separately. A repository license does **not** establish the license of downloaded papers,
containers, game assets, datasets, or bundled applications. “Check task terms” means those
components were not individually audited. Linked READMEs provide installation and paper/data
links. Counts describe the cited release, not a promise about a future checkout.

All compute descriptions exclude model inference. Most listed projects provide single-agent
benchmarks: the swarm decompositions below are proposed adaptations, not upstream claims that
multi-agent execution is already supported. Keep the original task, observations and scoring
where possible, and label altered protocols explicitly.

## 3. Software and terminal work

| # / Candidate | Tasks and scoring | Availability and execution | Swarm opportunity / limitation |
| --- | --- | --- | --- |
| 1. [SWE-bench](https://github.com/SWE-bench/SWE-bench) | Resolve repository issues; regression tests score patches. Choose a named split such as Verified. | Local; MIT harness; repository/data terms separate. CPU containers, image cache and sandboxed shell. | Diagnosis, patch and review roles; separate worktrees, then one submitted patch. Public-task contamination and weak tests can distort gains. |
| 2. [SWE-rebench V2](https://github.com/SWE-rebench/SWE-rebench-V2) | [Paper](https://arxiv.org/abs/2602.23866): 32K+ executable tasks across 20 languages, plus a larger instruction collection. Test-based patch evaluation. | Local; MIT code; check [dataset](https://huggingface.co/datasets/nebius/SWE-rebench-V2) and per-repository terms. Docker images and language-specific toolchains. | Broader languages than a Python-only pilot. Freeze one quality-filtered subset; do not mix executable and instruction-only releases or assume generated tests are sound. |
| 3. [R2E-Gym](https://github.com/R2E-Gym/R2E-Gym) | 8.1K+ problems across 13 repositories; executable tasks, unit tests, and descriptions. | Local; Apache-2.0 code; [data collections](https://huggingface.co/R2E-Gym) and underlying repos need separate review. Containerized execution. | Parallel diagnosis and candidate patches, followed by integration. Distinguish benchmark tests from learned/verifier-based candidate selection; hold out tasks used for tuning. |
| 4. [Terminal-Bench 2.0](https://github.com/harbor-framework/terminal-bench-2) | Container tasks including async-code debugging, database recovery and scheduling; task-specific verifiers and oracle solutions. | Local; Apache-2.0 repository; downloaded components retain their terms. Uses [Harbor](https://github.com/harbor-framework/terminal-bench-2); CPU and task-specific GPU cases. | Strong first adapter: investigators, implementer, tester share artifacts. Version 2.0 is the verified source here; do not label it the paper's 2.1 evaluation. |
| 5. [Endless Terminals](https://github.com/kanishkg/endless-terminals) | Procedural terminal task generator; produces initial/final-state tests and container specifications. | Builder/local; Apache-2.0 code; published [datasets](https://huggingface.co/collections/obiwan96/endless-terminals) need card checks. Apptainer; generation additionally needs an LLM. | Useful scalable curriculum after an initial fixed benchmark. Freeze generated tasks and audit oracle success, specification quality and reward hacking before comparing swarm sizes. |

Prefer Terminal-Bench for initial breadth and SWE-rebench for a later repository-repair track.
These are not substitutes for professional workflows merely because both involve software.

## 4. Science and education

| # / Candidate | Tasks and scoring | Availability and execution | Swarm opportunity / limitation |
| --- | --- | --- | --- |
| 6. [CORE-Bench](https://github.com/siegelz/core-bench) | Reproduce computational papers and answer result questions; 270 tasks from 90 papers, with three assistance levels. | Local; MIT harness; capsule licenses separate. Python/R, Docker; CPU and GPU subsets. Upstream recommends [HAL](https://github.com/princeton-pli/hal-harness); original harness is no longer maintained. | Environment repair, execution and result auditing divide naturally. Start with CPU Medium/Hard; Easy mostly tests extracting supplied results. Multiple levels of one paper are related tasks, not independent samples. |
| 7. [ScienceAgentBench](https://github.com/OSU-NLP-Group/ScienceAgentBench) | 102 original tasks from 44 papers; produce Python programs and assess execution/output quality. A verified split was released in April 2026. | Conditional; MIT code, mostly CC BY 4.0 tasks with exceptions. Full artifacts downloaded separately; upstream asks against public redistribution of unpacked data. Some visualization judging uses GPT-4o. | Analyst/programmer/reviewer roles. Pin verified artifacts, distinguish executable checks from model judging, and avoid accidentally reporting upstream best-of-three as mean success. |
| 8. [DiscoveryWorld](https://github.com/allenai/discoveryworld) | Experimental discovery in a simulated world; scenario, seed, and Easy/Normal/Challenge settings. Task progress and scientific conclusions need distinct assessment. | Local; Apache-2.0 code; inspect asset notices. CPU Python environment with action/observation API. | Strong hypothesis/experiment/interpretation split. Upstream explicitly calls multiple user agents untested. First use a deliberating team controlling one actor; audit textual observations before assuming a text-only model is sufficient. |
| 9. [ScienceWorld](https://github.com/allenai/ScienceWorld) | Text-based science procedures, such as phase changes and classification; environment action/reward loop, variations and simplifications. | Local; Apache-2.0 code; CPU Python/JVM. Version 1.3.0 changes deterministic ordering and can change trajectories relative to 1.2.x. | Cheap procedural control and easy World wrapper. A team controlling one actor is simpler than inventing multiple actors; less representative of open-ended research. |
| 10. [MLE-bench](https://github.com/openai/mle-bench) | 75 Kaggle competitions; local submission grading on reconstructed held-out splits. Low-complexity subset has 22 competitions. | Conditional; [MIT covers code, not external data](https://github.com/openai/mle-bench/blob/main/LICENSE). Dataset access and competition terms vary; CPU/GPU, sizable storage. Upstream lists 158 GB for the full low-complexity subset. | Parallel feature/model experiments plus independent validation; natural continuous score. Allocate a fixed aggregate training-compute budget, not a GPU per added agent without accounting for it. |
| 11. [MLGym](https://github.com/facebookresearch/MLGym) | 13 open-ended AI research tasks spanning vision, NLP, RL and game theory; task-specific performance objectives. | Hold/conditional; CC BY-NC 4.0, not unrestricted open-source. GitHub marked archived at review time. Docker/Podman and GPU support. | Strong experiment-design fit, but licensing/maintenance make MLE-bench preferable for the first optimization track. Dataset terms remain separate. |

## 5. Life sciences and healthcare

| # / Candidate | Tasks and scoring | Availability and execution | Swarm opportunity / limitation |
| --- | --- | --- | --- |
| 12. [BixBench](https://github.com/Future-House/BixBench) | Multi-step bioinformatics using Python/R/Bash and notebook-derived data. Current README reports 205 questions; original paper describes a different release. Open-answer and multiple-choice modes. | Local with artifact dependencies; Apache-2.0 code; [dataset](https://huggingface.co/datasets/futurehouse/BixBench) and source-publication data terms require review. Docker notebook runtime; README instructs HF authentication. | Data inspection, statistical analysis and independent interpretation. Use agentic open-answer mode; audit each selected question's verifier and judge dependency. Do not equate questions with independent capsules. |
| 13. [MedAgentBench](https://github.com/stanfordmlgroup/MedAgentBench) | Execute tasks in a virtual electronic health record through a FHIR server; task-specific evaluation. | Conditional until artifacts checked; MIT repo; Docker FHIR image and separately downloaded `refsol.py`. Patient/task artifact terms not independently verified. | Retrieval and consistency review are useful, but many tasks may be too short for large teams. Use only benchmark records; serialize state-changing actions and reset the database per trial. |

BixBench is the better first fit for long analytical trajectories. MedAgentBench adds a different
capability—correct state changes in a structured application—rather than broader biology coverage.

## 6. Finance

| # / Candidate | Tasks and scoring | Availability and execution | Swarm opportunity / limitation |
| --- | --- | --- | --- |
| 14. [SecondState FAB](https://github.com/SecondState-ai/finance-agents-benchmark) | Synthetic-company due diligence: 50 tasks, 160 documents, 231 criteria. Criterion-level LLM judging; task passes only when every criterion passes. | Local/conditional judge; MIT code, CC BY 4.0 tasks/data/rubrics/results. Python, Docker/Podman; documented judge requires an OpenAI credential. | Split financial, contractual and evidence checks; produce one supported report. Excellent document-collaboration candidate, but one fictional company limits generalization and judge variance must be measured. |
| 15. [Vals Finance Agent Benchmark](https://github.com/vals-ai/finance-agent) | Research company filings and answer financial questions. [Paper](https://arxiv.org/abs/2508.00828) describes 537 questions. | Conditional; MIT harness does not make the full evaluation open. Documented path requires Vals approval/API access, Tavily and SEC API credentials. | Parallel filing retrieval, calculation and evidence verification. Live retrieval complicates repeatability; not the default self-hosted finance choice. Do not confuse this project with SecondState FAB. |
| 16. [FinRL](https://github.com/AI4Finance-Foundation/FinRL) | Financial RL library with train/backtest pipelines, not a packaged LLM task suite. Portfolio and trading objectives can become continuous-scored tasks. | Builder; MIT code, provider-specific market-data terms. CPU for small backtests; training may need GPUs. Download/API requirements depend on chosen provider. | Researchers propose features and strategies, reviewer checks leakage. Create frozen offline data, transaction-cost assumptions and holdouts; no live trading. This would be a new Flock benchmark, not an official FinRL LLM score. |

## 7. Professional services and information/media

| # / Candidate | Tasks and scoring | Availability and execution | Swarm opportunity / limitation |
| --- | --- | --- | --- |
| 17. [TheAgentCompany](https://github.com/TheAgentCompany/TheAgentCompany) | 175 task images spanning engineering, PM, HR, finance and administration. Result/checkpoint evaluation includes deterministic and LLM methods. | Local/conditional NPC/judge; MIT benchmark, separate app/data terms. Docker services include GitLab, Plane, ownCloud and RocketChat; README asks for 30+ GB disk. Environment LLM configuration is separate from agent LLM. | Strongest organizational-work candidate. Delegate evidence gathering, project updates and review. Isolate service state across trials; keep simulated coworkers distinct from evaluated swarm members. |
| 18. [WorkArena / WorkArena++](https://github.com/ServiceNow/WorkArena) | ServiceNow browser workflows. L1 has 33 task types; ++ composes longer planning tasks. Environment validation supports completion checks. | Conditional; [Apache-2.0 code](https://github.com/ServiceNow/WorkArena/blob/main/LICENSE), but ServiceNow instance access is gated. BrowserGym/Playwright and approved instance access. | Policy interpretation, navigation and checking; serialize shared-record updates. Prefer ++ for this task horizon, but it is not fully self-hosted OSS. |
| 19. [WebArena](https://github.com/web-arena-x/webarena) | Tasks in self-hosted websites, with executable evaluation. Covers information retrieval and web application workflows. | Local; Apache-2.0 harness, separate hosted-app/data licenses. Browser/site containers; upstream recommends AgentLab/BrowserGym infrastructure. | Parallel information gathering followed by one state-changing executor. Multiple browser sessions need consistent identities; raw browser pages can consume context quickly. |
| 20. [AppWorld](https://github.com/StonyBrookNLP/appworld) | [Paper](https://arxiv.org/abs/2407.18901): 750 tasks over simulated personal applications. State-based tests check completion and collateral changes. | Conditional; public code Apache-2.0; protected bundles add encrypted-redistribution requirements. Local API/code execution; preserve upstream train/dev/test restrictions. | API-oriented information and scheduling workflows are attractive for text models. Team can inspect separate apps, but transactions need ownership. Do not call all components unmodified Apache-2.0. |
| 21. [Toolathlon](https://github.com/hkust-nlp/Toolathlon) | Long multi-app tool sequences; current README identifies Toolathlon-Verified, plus a sandbox branch. Execution-based task checking. | Hold pending license/setup audit: public code but GitHub reports no repository license. Real-service dependencies vary; public evaluation service is a separate execution option. | Strong cross-app collaboration potential. Inspect sandbox and OpenReward/Toolathlon-GYM variants separately; different task counts or mocked services change what is measured. |

Information/media coverage here mainly means search, document handling and publishing workflows.
It does **not** cover professional video editing or creative-media quality. Those remain gaps;
adding an image benchmark without an objective task outcome would not close them.

## 8. Games, industrial tasks and infrastructure

| # / Candidate | Tasks and scoring | Availability and execution | Swarm opportunity / limitation |
| --- | --- | --- | --- |
| 22. [TextArena](https://github.com/TextArena/TextArena) | 100+ text games across single-player, cooperative and competitive settings; game-specific terminal outcomes/rewards. | Local; MIT code; CPU and a text action loop. Select and pin concrete environments rather than relying on the changing suite total. | Cheap protocol and communication experiments. Choose cooperative/team objectives; ordinary adversarial self-play is not swarm task completion. Many games are short. |
| 23. [OpenSpiel](https://github.com/google-deepmind/open_spiel) | General game framework with cooperative, general-sum and zero-sum games; native utilities/outcomes. | Builder/local; Apache-2.0 code; Python/C++ on CPUs, game-specific assets separate. | Controlled imperfect-information and coordination tests. Needs textual observation/action mapping and an explicit team objective; not a turnkey long-horizon language-agent benchmark. |
| 24. [Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai) | Cooperative kitchen task: coordinate ingredients, cooking and deliveries to maximize reward. Layouts change coordination pressure. | Local; MIT implementation; CPU simulator. This is a research environment, not a requirement to automate the commercial game. | Strong diagnostic for division of labor and blocking. Original small-team setting is unsuitable for pretending there are 32 independent in-world agents; use fixed actors with planner teams and label that protocol. |
| 25. [Factorio Learning Environment](https://github.com/JackHopkins/factorio-learning-environment) | Factory-building via Python REPL actions; production/technology objectives can support sustained progress. | Conditional; [MIT applies to FLE, not Factorio](https://github.com/JackHopkins/factorio-learning-environment/blob/main/LICENSE). Docker/headless setup; current README says game client needed only for optional rendering. Verify server distribution terms separately. | Excellent potential for mining, power, logistics and production specialists. Shared placement/inventory conflicts and selected-version multiplayer behavior need validation; not entirely open-source infrastructure. |
| 26. [OR-Gym](https://github.com/hubbs5/or-gym) | Inventory, supply-chain networks, vehicle routing, packing and related operational simulations; objective rewards. | Builder; MIT code; CPU simulator, older Gym/Ray assumptions in examples. Synthetic scenarios can avoid external data. | Divide demand forecasting, replenishment and routing; measure cost/service tradeoffs. Requires task prompts, fixed instances and baselines; no established LLM swarm leaderboard. |
| 27. [SUMO](https://github.com/eclipse-sumo/sumo) | Traffic simulation with routing and network examples; suitable foundation for signal-control or transport-planning tasks. | Builder; EPL-2.0 repository, maps/demand data separate. CPU simulations; scale depends on road network. | Specialists optimize districts then check network-wide effects. Must define frozen demand, constraints and delay/throughput scoring. This is new benchmark construction, not an off-the-shelf agent evaluation. |

For an initial industrial slice, use Terminal-Bench's actual operations tasks, clearly labeled
as IT operations. Physical logistics/manufacturing coverage requires OR-Gym/SUMO task design or
the conditional Factorio route; the sector is a genuine readiness gap.

## 9. Ranked integration shortlist

Ranking is a judgment balancing objective scoring, replayability, collaboration opportunity,
CPU accessibility and breadth. Effort bands estimate one engineer's adapter/smoke-test work:
**S: 1–3 days**, **M: 4–7 days**, **L: 1–2 weeks**, after artifacts are accessible.
They exclude licensing delays, a production-hardening pass and full experiment execution.
Pilot horizons are proposed caps, not observed runtimes or upstream leaderboard budgets.

| Rank | Candidate | Proposed first experiment | Adapter / effort | Pilot cap and key gate |
| ---: | --- | --- | --- | --- |
| 1 | Terminal-Bench 2.0 | CPU async-debugging and database-recovery tasks; diagnosis, implementation and test roles. | Sandboxed shell/filesystem plus Harbor lifecycle bridge; M. | 60 min/task; oracle succeeds and no-op fails in the exact image. |
| 2 | CORE-Bench | Reproduce a CPU capsule at Medium and Hard; separate dependency repair from result checking. | Isolated capsule workspace and HAL-compatible final report; M. | 90 min/task; dependencies/artifacts reproducible, accepted reference answers verified. |
| 3 | TheAgentCompany | Select project-management/data-analysis tasks involving multiple artifacts; start without NPC-dependent tasks. | Browser/shell tools plus per-trial service reset; L. | 90 min/task; reset isolation and evaluator/NPC model costs identified. |
| 4 | BixBench | One capsule's related analysis questions: explore data, analyze, independently verify interpretation. | Containerized Python/R execution and answer adapter; M. | 90 min/capsule; selected verifier types and task-data access confirmed. |
| 5 | SecondState FAB | Evidence reconciliation across financial and contractual documents, ending in one cited response. | Read-only data-room mount, scratch files and external grading step; S–M. | 60 min/task; freeze judge version, test grading consistency, review document/task terms. |
| 6 | DiscoveryWorld | Proteomics on Normal, then Challenge; team proposes experiments and controls one actor. | World service wrapping observations/actions; M. | 60 min/scenario; text observability and action serialization verified. |
| 7 | MLE-bench | Small tabular competition from low-complexity split; independent features/models and a validation reviewer. | Sandboxed training jobs, artifact registry and offline grader; L. | 120 min/task; fixed aggregate training compute and dataset acceptance. |
| 8 | Overcooked-AI | Several fixed layouts; compare planning teams controlling the same two actors. | World service with deterministic ticks and textual state; M. | Fixed simulator steps plus 30 min wall cap; same macro-actions and actor count for all arms. |

Readiness fallback: if browser/service setup delays TheAgentCompany, bring forward DiscoveryWorld
or a ScienceWorld control. Do not silently replace a benchmark inside an already registered
comparison. BixBench and FAB judging costs are separate from agent inference costs.

## 10. Flock integration implications

The existing [World interface](../world.md) exposes a manifest and tools through an external
HTTP service; agent identity accompanies calls. Use it for authoritative shared state, not for
copying entire third-party frameworks into Flock. The [design](../design.md) assigns experiment
metrics and budgets to the experiment layer rather than to the hosting controller.

Three reusable adapter patterns would cover most of the shortlist:

1. **Artifact work:** task container, immutable inputs, private agent scratch/worktrees,
   shared publication directory, one submitted artifact and out-of-band grader. Fits terminal,
   repository, scientific, ML and document tasks. Merge patches explicitly; simultaneous edits
   to one checkout can measure filesystem races rather than collaboration.
2. **Shared application:** one resettable service instance per trial, independent browser/tool
   sessions and serialized destructive writes. Fits company tasks, web apps and EHRs. Cache
   immutable assets; do not share mutable databases between experimental arms.
3. **Simulation World:** observe/act/reset backed by a deterministic simulator, with one
   authoritative action order and native rewards. Fits discovery, games and operations.
   Preserve per-player information boundaries where the task requires them.

Pi remains the agent harness, but the escape benchmark's tools-only setup is not sufficient for
arbitrary code execution or browsers. Supply explicit sandbox tools or configure appropriate
native access for each adapter. Do not assume a provider name alone makes a vision-dependent
benchmark usable with the current model endpoint.

World calls have a documented 60-second timeout and 1 MB response limit. Long scientific jobs
need submit/status/result tools, bounded logs and artifact references rather than one blocking
call. Keep graders, hidden tests and solutions outside agent-visible mounts. Run reference/oracle
validation under the same image and resource constraints before spending model tokens.

Use node-local storage for live SQLite and mutable execution state, and publish consistent
snapshots/artifacts to the bucket. Existing message-board lifetime and missing-token-accounting
issues are prerequisites to trustworthy comparisons; this research does not modify them or
restart any experiments. Preserve per-conversation routing identities and record router/harness
versions in future run manifests.

## 11. Proposed experimental progression

### Stage A: integration gates

For each candidate, choose three development tasks with different failure modes. Confirm reset,
reference success where supplied, negative-control failure, timeout cleanup, hidden-grader
separation, complete token accounting and one small communicating team. For continuous objectives,
use a known baseline instead of inventing a binary oracle. Reject broken tasks before model
comparison, keeping exclusions and reasons in a manifest.

### Stage B: CPU-first breadth pilot

Select five tasks per family from Terminal-Bench, CORE-Bench, TheAgentCompany, BixBench,
SecondState FAB, DiscoveryWorld and Overcooked-AI: 35 tasks. This samples capabilities rather
than copying Avalon's training percentages. Use development splits where upstream requires them;
keep tuning and final evaluation tasks separate. The family mix covers most sectors but leaves
physical infrastructure and creative media explicitly incomplete.

Compare the same tasks and seeds with:

- One agent.
- Independent teams at N = 2, 4, 8, without communication or shared intermediate artifacts.
- Communicating teams at N = 2, 4, 8, with the same tools, model and allowed task information.

Use three trials per task/configuration: **35 × 7 × 3 = 735 episodes** once all gates pass.
This is a proposed campaign, not a launch request. For the independent arm, freeze a selector
based only on permitted validation evidence, with selector tokens counted; no choosing via hidden
test scores. For stateful tasks, separate independent candidate worlds and select one final
state/artifact. If candidate worlds change simulation or tool-compute access, account for that
resource explicitly and report the limitation rather than claiming a pure communication ablation.

Start with a fixed aggregate token budget B per task shared by all team members. Calibrate B on
development tasks, then freeze it and the wall cap before evaluation. Team size must not multiply
B. A later fixed-wall-time study can allow different token use, but report it as a separate
question. Keep aggregate task CPU/GPU and simulator-step budgets fixed as well. Queue launches
against measured router capacity; agent count is not the same as active-request concurrency.

### Stage C: longer optimization and missing sectors

Add five MLE-bench tasks only after reserving task-training compute separately from model-serving
GPUs. Extend larger agent counts to tasks with enough parallel work; do not automatically sweep
16/32 agents through two-player games. Investigate OR-Gym supply chains or SUMO for a fully open
physical-infrastructure track. Evaluate Factorio separately if proprietary-engine dependence is
acceptable. Retain task-family results rather than hiding differences in a single pooled score.

### Measurement and reporting

Record task/dataset revision, container digest, seed, roles, actor count, agent count, model,
reasoning setting, context/output limits, harness/router revisions, comms settings, budgets,
all tool calls, artifacts and evaluator configuration. Board names should include benchmark,
task ID, N, trial and UTC timestamp, e.g. `tbench-db-wal-recovery-n4-r1-20261101T120000Z`.

Primary outcomes: success or native score at fixed budget, score-versus-time/token curves,
time-to-solution, and total input/output tokens across every agent. Also report normalized
`total_tokens / N`, communication tokens, retries, environment/NPC/judge costs, CPU/GPU time,
and failures. Count coordinator/selector work, and define whether it is one of the N agents.
Use N as total model-driven swarm members by default.

For terminal correctness tasks, partial test counts are diagnostic, not a replacement for the
canonical pass criterion. Grade checkpoints out of band where feasible, without feeding hidden
feedback to agents. For games, preserve native actor count while varying advisory agents only
when explicitly labeled. For research tasks, group related questions/levels by source capsule
or paper when estimating uncertainty.

Show all attempted trials in a status table. Separate task failure, timeout, infrastructure
failure, solved-with-incomplete-accounting and valid solve. Preserve valid correctness outcomes
when usage is missing, but exclude them from token-efficiency estimates with explicit counts.
Do not report tokens-to-solution averaged only over easy successes as the main efficiency metric;
report success at budget and censored/unsolved runs alongside it. Use paired task comparisons
and intervals across tasks, not just error bars over repeated runs of one scenario.

## 12. Remaining verification work before adoption

This report did not install environments, download full task archives, execute oracle solutions,
or benchmark resource use. Those are integration gates, not completed research claims.
Specifically unresolved:

- Exact license/access of each external capsule, competition dataset, medical artifact and app
  image. Repository license metadata alone is insufficient.
- Toolathlon's missing license and whether any sandbox/GYM port resolves licensing as well as
  service dependencies. A separately licensed port does not automatically license upstream data.
- Which BixBench/ScienceAgentBench/company tasks require model judges, vision or NPC inference;
  which can be scored reproducibly with execution-only checks.
- Selected-version support for genuine multiple actors in DiscoveryWorld/Factorio, and the cost
  of browser service reset/isolation on this cluster.
- Runnable equivalents for the paper's in-house environments. Public OpenReward entries are
  leads, not evidence that all 147 paper environments are reproducibly available.

No adapters, datasets or paid workloads were launched as part of this research.
