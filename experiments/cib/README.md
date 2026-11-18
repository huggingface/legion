# Collaborative Intelligence Benchmark (CIB)

One Pi agent is evaluated as a teammate in four dimensions of
[Stanford's Collaborative Intelligence Benchmark](https://github.com/zou-group/collaborative-intelligence).
The Harbor task runs the other teammates and judges in separate containers.

**Setup:** follow the [dataset quickstart](https://huggingface.co/datasets/HuggingFaceSwarms/collaborative-intelligence-benchmark#setup)
to clone `cib-harbor` beside this Flock checkout and configure model and team credentials.
You need Linux, Docker with Compose (or rootless Podman), and access to the dataset.
Inference is paid, including partner and judge calls.
On the HF cluster (no Docker on compute nodes), set `environment.backend: podman` in `config.yaml` and follow the
dataset's [Podman setup](https://huggingface.co/datasets/HuggingFaceSwarms/collaborative-intelligence-benchmark/blob/main/adapter/README.md#podman).

**Run one Gemini solo episode** from the Flock checkout:

```sh
uv run --locked --with '../cib-harbor/adapter[hub]' collab-intel-harbor-run \
  --config experiments/cib/config.yaml
```

This benchmark uses the dataset's launcher with Flock's native Harbor runner.
Keep one evaluated agent; the task supplies its teammates. The benchmark launcher configures
its model gateway and preserves Stanford's solo eligibility rule and score aggregation.

For the four-dimension sweep, other models and results, use the
[dataset instructions](https://huggingface.co/datasets/HuggingFaceSwarms/collaborative-intelligence-benchmark#run).
