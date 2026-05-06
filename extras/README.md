# `extras/` — doc-only stubs

> **Hard partition.** Nothing under `extras/` is on the runtime path. The
> core package ([`../src/musicbox_churn/`](../src/musicbox_churn/)) does
> not import from here. Tests do not exercise it. CI does not validate
> it. It exists to document **shape**, not to ship code.

The core ships a small, honest pipeline (data → train → batch-score →
monitor). Anything that would imply a larger product than what's actually
here lives under `extras/` with a banner saying so.

| Stub | Purpose | Status |
|---|---|---|
| [feast/](feast/) | Declarative feature definitions for training/inference consistency | doc-only |

Future planned additions (not yet written):
- `api/` — FastAPI online-serving skeleton. Documents the request/response
  shape; would need raw-log access (which we don't have) to recompute
  rolling features at request time. See [../README.md](../README.md)
  § Honest framing.
- `k8s/api-deployment.yaml`, `api-service.yaml`, `train-job.yaml` — k8s
  manifests for the (deliberately not shipped) online API and training
  jobs. The actual core deploy artifact ([../k8s/batch-score-cronjob.yaml](../k8s/batch-score-cronjob.yaml))
  lives in the top-level `k8s/`, not here.
