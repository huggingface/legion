# Billing and storage

`namespace` is the optional account that pays for worker and controller Jobs, hf-sandbox
sandboxes, vLLM servers started with `inference.hf_jobs`, and HF router requests. Runnable
examples omit it for personal-account defaults. You can set it on any compute
backend, including Slurm: it still selects billing for HF sandboxes and inference.
E2B sandboxes are billed to `E2B_API_KEY`'s account, not `namespace`.

With `namespace` omitted or empty, HF infrastructure uses `HF_TOKEN`'s account and HF router
requests use the inference key's default billing account, with no `X-HF-Bill-To` header.

## Personal accounts

Use two separately scoped tokens when agents run on remote infrastructure:

| Credential | Permissions |
|---|---|
| `HF_TOKEN` | Manage Jobs and create/write buckets in your personal account. If deploying a board Space, create/write that Space and its storage too. Separately grant read access to the private task/model repositories used by the run, including any required org-owned datasets. |
| The inference key (`HF_INFERENCE_TOKEN` in the examples) | Make Inference Providers calls for your personal billing account. For `hf_jobs` servers, also read access to the job's personal namespace for the jobs proxy. No infrastructure write permissions are needed for router calls. |

The inference key enters each agent's container. Keep the infrastructure token out of it.
Private task/model read access does not require billing Jobs or inference to the repository's
org. A bucket in another namespace additionally needs bucket write access there.

## Optional org billing

To bill an organization instead, explicitly set its namespace:

```yaml
namespace: my-org
bucket_id: legion-runs
```

This selects the org for HF infrastructure and router billing; router requests include
`X-HF-Bill-To: my-org`. Grant `HF_TOKEN` the corresponding org Jobs/bucket/Space permissions,
and the inference key org-scoped Inference Providers permission (`inference.serverless.write`).
For `hf_jobs` servers, the inference key also needs org read access for the jobs proxy.
Org membership and token permissions both need to allow the selected resources.

Board placement is independent: a bare `BOARD_SPACE` always uses `HF_TOKEN`'s account.
Choose `my-org/<unique-run-board>` explicitly if the board should live in the org too.

Hugging Face documents router organization billing for Team and Enterprise organizations in
its [Inference Providers billing guide](https://huggingface.co/docs/inference-providers/en/pricing#organization-billing).
That is documentation support; an actual org preflight still needs to verify your tokens and
plan. It makes billable inference requests.

## Bucket IDs

Set `bucket_id` to a bare name (`legion-runs`) or a qualified ID (`your-username/legion-runs`), as
in [huggingface_hub](https://huggingface.co/docs/huggingface_hub/guides/buckets).
Bare names use the explicit namespace, else `HF_TOKEN`'s account. Qualified IDs select storage
independently of billing. `hf://buckets/...` URLs and user-provided path prefixes are refused
in this field. The former `bucket` field has been removed.

`legion launch` resolves the ID once, creates a missing bucket private with `HF_TOKEN`, and
stores `<namespace>/<name>` for the controller and workers. Existing buckets retain their
visibility. The source YAML and an unset namespace stay as written. `legion init` checks an
existing bucket's write access; for a missing one it reports that launch will create it.
Legion's own `controller/<swarm>/` and `<swarm>/<type>/<agent>/` folders are unchanged.

`bucket_id` is required for HF Jobs and Slurm workers. Local workers keep agent state on disk;
an optional bucket can still hold the local controller's checkpoint. For Slurm, launch writes
the resolved ID to `hosts/bucket_id` before submitting the controller, so the batch script can
construct its checkpoint URL without another account lookup.

A configured bucket is always prepared, including local and foreground launches. It needs
bucket create/write credentials even with `LEGION_STATE_URL=` (checkpointing off). Leave
`bucket_id` unset for a local run that needs no Hub storage or credentials. The foreground
`python -m legion.controller.localrun` command uses the same bucket preparation as `legion launch`.

A manually org-deployed job controller refuses a blank namespace when `HF_TOKEN` belongs
to another account. For a controller in `my-org`, set `namespace: my-org`
explicitly so worker placement and billing use that org. Matching-account controllers accept
an unset namespace, keep it empty in the stored spec and send no router bill-to header.
See [job deployment](deploy-job.md).
