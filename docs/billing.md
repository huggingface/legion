# Billing and storage

`namespace` is the optional account that pays for worker and controller Jobs, hf-sandbox
sandboxes, vLLM servers started with `inference.hf_jobs`, and HF router requests. Examples
omit it to use the launcher's account for infrastructure. You can set it on any compute
backend, including Slurm: it still selects billing for HF sandboxes and inference.

```yaml
namespace: my-org
bucket_id: flock-runs
```

With `namespace` omitted or empty, infrastructure uses `HF_TOKEN`'s account. Router requests
send no bill-to header, so the inference key's owner pays. An explicit namespace sends
`X-HF-Bill-To: my-org` on both preflight and agent requests. Flock never fills that header from
`whoami(HF_TOKEN)`: the two tokens can belong to different people. There is no
`inference.hf_router.bill_to` or `inference.hf_jobs.namespace` setting.

## Tokens for an organization

Use two separately scoped tokens when agents run on remote infrastructure:

| Credential | Permissions |
|---|---|
| `HF_TOKEN` | Manage Jobs and create/write buckets in the billing org; read private task/model repositories used by the run. If deploying a board Space, create/write that Space and its storage too. |
| The inference key (`HF_INFERENCE_TOKEN` in the examples) | Make Inference Providers calls on the billing org (`inference.serverless.write`) for router bill-to. For `hf_jobs` servers in the org, also read access to the org: the jobs proxy admits only tokens with that access. |

The inference key enters each agent's container. Keep the infrastructure token out of it.
Org membership and token permissions both need to allow the selected resources. A bucket in
another namespace additionally needs bucket write access there.

Hugging Face documents router organization billing for Team and Enterprise organizations in
its [Inference Providers billing guide](https://huggingface.co/docs/inference-providers/en/pricing#organization-billing).
That is documentation support; an actual org preflight still needs to verify your tokens and
plan. It makes billable inference requests.

## Bucket IDs

Set `bucket_id` to a bare name (`flock-runs`) or a qualified ID (`storage-org/flock-runs`), as
in [huggingface_hub](https://huggingface.co/docs/huggingface_hub/guides/buckets).
Bare names use the explicit namespace, else `HF_TOKEN`'s account. Qualified IDs select storage
independently of billing. `hf://buckets/...` URLs and user-provided path prefixes are refused
in this field. The former `bucket` field has been removed.

`flock launch` resolves the ID once, creates a missing bucket private with `HF_TOKEN`, and
stores `<namespace>/<name>` for the controller and workers. Existing buckets retain their
visibility. The source YAML and an unset namespace stay as written. `flock init` checks an
existing bucket's write access; for a missing one it reports that launch will create it.
Flock's own `controller/<swarm>/` and `<swarm>/<type>/<agent>/` folders are unchanged.

`bucket_id` is required for HF Jobs and Slurm workers. Local workers keep agent state on disk;
an optional bucket can still hold the local controller's checkpoint. For Slurm, launch writes
the resolved ID to `hosts/bucket_id` before submitting the controller, so the batch script can
construct its checkpoint URL without another account lookup.

A configured bucket is always prepared, including local and foreground launches. It needs
bucket create/write credentials even with `FLOCK_STATE_URL=` (checkpointing off). Leave
`bucket_id` unset for a local run that needs no Hub storage or credentials. The foreground
`python -m flock.controller.localrun` command uses the same bucket preparation as `flock launch`.

A manually org-deployed job controller refuses a blank namespace when `HF_TOKEN` belongs
to another account: set `namespace: my-org` explicitly so worker placement and billing use
that org. Matching-account controllers accept an unset namespace and keep it empty in the
stored spec and router headers. See [job deployment](deploy-job.md).
