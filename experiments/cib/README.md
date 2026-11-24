# Collaborative Intelligence Benchmark (CIB)

One Pi agent is evaluated as a teammate in four dimensions of
[Stanford's Collaborative Intelligence Benchmark](https://github.com/zou-group/collaborative-intelligence).
The Harbor task runs the other teammates and judges in separate containers.

**Setup:** follow the [dataset quickstart](https://huggingface.co/datasets/HuggingFaceSwarms/collaborative-intelligence-benchmark#setup)
to clone `cib-harbor` beside this Legion checkout and configure credentials: the evaluated model's key and
the team's (partner and judge keys and a Qwen3-14B endpoint), which are needed even for a solo episode.
You need Linux, Docker with Compose (or rootless Podman), and access to the dataset.
Inference is paid, including partner and judge calls.
On the HF cluster (no Docker on compute nodes), set `environment.backend: podman` in `config.yaml` and follow the
dataset's [Podman setup](https://huggingface.co/datasets/HuggingFaceSwarms/collaborative-intelligence-benchmark/blob/main/adapter/README.md#podman).

**Run one Gemini solo episode** from the Legion checkout. To check the setup first without model calls,
run `collab-intel-harbor-sweep` with the same config, `--dataset-root ../cib-harbor --output results/cib/plan --plan-only`.

```sh
uv run --locked --with '../cib-harbor/adapter[hub]' collab-intel-harbor-run \
  --config experiments/cib/config.yaml
```

The recipe pins a Legion-compatible revision of the dataset. This benchmark uses the dataset's launcher
with Legion's native Harbor runner.
Its native config has no Legion `namespace` field: model billing is configured at the
benchmark's Gemini gateway, independently of HF Jobs and Inference Providers.
Keep one evaluated agent; the task supplies its teammates. The benchmark launcher configures
its model gateway and preserves Stanford's solo eligibility rule and score aggregation.

For the four-dimension sweep, other models and results, use the
[dataset instructions](https://huggingface.co/datasets/HuggingFaceSwarms/collaborative-intelligence-benchmark#run).
