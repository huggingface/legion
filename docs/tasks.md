# Integrating tasks

[Back to the README](../README.md)

Shepherd consumes ready-to-run Harbor tasks. Adding a task does not require an
AgentEscape adapter or a change to Shepherd's runner.

1. Create a self-contained task following the [Harbor task format](https://docs.harborframework.com/tasks/overview).
   A typical single-container task has this layout:

   ```text
   tasks/my-task/
   ├── instruction.md
   ├── task.toml
   ├── environment/
   │   └── Dockerfile
   ├── solution/
   │   └── solve.sh
   └── tests/
       └── test.sh
   ```

   The instruction describes the objective, `task.toml` configures the task, and
   `tests/test.sh` grades the result and writes a numerical reward under
   `/logs/verifier/`. The oracle solution is optional but useful for validating the
   task. Multi-service tasks can include `environment/docker-compose.yaml`.

2. Validate the task locally with Harbor, including its verifier. If you provide an
   oracle solution, run it before trying a model:

   ```sh
   uv run harbor run --path tasks/my-task --agent oracle --env docker
   ```

   Use the backend you intend to run. The agent image must support Harbor's Pi
   installer and, when using Shepherd tools, the Python tool runner.

3. Publish the task directory under `tasks/` in a Hugging Face dataset repository.
   Include the files it needs inside that directory; references to sibling task
   directories are not portable. Record the published commit SHA.

4. Select it in a copy of an experiment YAML:

   ```yaml
   task:
     repo: https://huggingface.co/datasets/your-org/your-tasks
     revision: <published-commit-sha>
     selector: my-task
   ```

   Omit `task.dataset` for this direct-discovery layout. If the repository instead
   provides a Harbor registry, specify its `name@version` entry. Shepherd resolves
   the selector, downloads the selected task, and gives each agent its own trial.
   No local generation step or `task.path` setting is needed.

Start with one agent to check model access, tool setup, and grading before increasing
`agents.count`.


## AgentEscape environments

The AgentEscape dataset is private and requires access. Each agent gets a separate
puzzle instance; the shared board carries findings between them. These tasks define
three services per agent: the Pi environment, puzzle server, and execution worker.
Other Harbor tasks may use a single container. A successful AgentEscape attempt must
submit its answer through the task's `escape` CLI to earn a reward.
