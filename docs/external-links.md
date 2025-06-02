# External References — Google Cloud (backend-critical)

This document contains implementation-critical Google Cloud specifications that backend agents must respect.
All links are sourced from canonical Google Cloud documentation as of June 2025.

| Topic | Must-know snippet | Canonical URL |
|-------|------------------|---------------|
| Cloud Run Gen 2 execution-env | Add annotation → `run.googleapis.com/execution-environment: "gen2"` to service.yaml; Gen 2 has no system-call emulation and cold-start ≈ 600 ms. | [Cloud Run Execution Environments](https://cloud.google.com/run/docs/configuration/execution-environments) |
| Service→Service auth (Cloud Run) | From caller: `curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" https://YOUR_SERVICE/run` – verify with Cloud Run's built-in Authorization header check. | [Service-to-Service Authentication](https://cloud.google.com/run/docs/authenticating/service-to-service) |
| Pub/Sub push (OIDC) | Push JWT claims → `aud=<pushEndpoint>`, `iss=pubsub.googleapis.com`, `sub=serviceAccount:<SA_EMAIL>`. Validate signature with Google certs bundle. | [Pub/Sub Push Authentication](https://cloud.google.com/pubsub/docs/authenticate-push-subscriptions) |
| Firestore PITR | Retains doc versions 7 days; restore via `gcloud firestore pitr restore --to-time="<RFC3339>"`. Storage overhead billed at 15% of live DB. | [Firestore Point-in-Time Recovery](https://cloud.google.com/firestore/docs/pitr) |
| Vertex AI Gemini quotas | Default per-project (us-central1): 6 req/s, 60K output tokens/min. Request raise via "Generative AI quotas" page. | [Vertex AI Generative AI Quotas](https://cloud.google.com/vertex-ai/generative-ai/docs/quotas) |
| Gemini 2.5 Pro Model | Model ID: `gemini-2.5-pro-preview-05-06`. Public preview with 1M+ input tokens, 65K output tokens. Supports text, code, images, audio, video. | [Gemini 2.5 Pro Model Card](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-pro) |
| Vertex AI Model Garden | Complete overview of available generative models, versions, and capabilities on Vertex AI platform. | [Model Garden Overview](https://cloud.google.com/vertex-ai/generative-ai/docs/models) |
| Model Versions & Lifecycle | Detailed reference for model versioning, lifecycle stages, and deprecation policies for Vertex AI models. | [Model Versions and Lifecycle](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/overview) |
| Identity Platform JWT | Claims always include `auth_time`, `sub = UID`, `firebase.sign_in_provider`. Back-end should trust only `aud=<PROJECT_ID>`. | [Identity Platform JSON Web Tokens](https://cloud.google.com/identity-platform/docs/json-web-tokens) |
| Cloud Storage CMEK | Attach key with: `gsutil kms encryption -k projects/…/cryptoKeys/key gs://YOUR_BUCKET`. Rotate via `gcloud kms keys versions create --location=global --keyring=… --key=key`. | [Cloud Storage Customer-Managed Encryption Keys](https://cloud.google.com/storage/docs/encryption/customer-managed-keys) |
| Workload Identity Federation (CI/CD) | GitHub Actions OIDC → create pool/provider, then `GOOGLE_APPLICATION_CREDENTIALS` not required. Example in blog post. | [Workload Identity Federation](https://cloud.google.com/iam/docs/workload-identity-federation) |
| Secret Manager best-practice | Fetch secrets at start-up, cache in memory; avoid latest alias in production—pin explicit version to guarantee rollbacks. | [Secret Manager Best Practices](https://cloud.google.com/secret-manager/docs/best-practices) |
| Cloud Logging exclusions | Exclude regex: `resource.type="cloud_run_revision" AND jsonPayload.samples` → blocks raw HealthKit values. | [Cloud Logging Exclusions](https://cloud.google.com/logging/docs/exclusions) |

## How to use

1. **Copy the table into your repo.** ✅ (This document)
2. **In each tech-specific doc** (google-cloud-2025.md, security.md, etc.) add a short line:
   "See External References → Cloud Run Gen 2 for the latest flags."
3. **(Optional)** add a CI step that 404-checks every URL above; if Google moves a page, your agent will raise a PR with the new link.

## Integration Status

With these deep-links you now have:

- ✅ **Complete in-repo how-to** (what your markdown already covers)
- ✅ **Canonical vendor knobs & quotas** (the table above)

This closes the last "where do I find the official spec?" question an autonomous backend agent might hit.
