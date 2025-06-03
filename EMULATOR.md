Do you need a local database to work on / test this repo?

Short version: No mandatory local DB right now—the tree already gives you two “plugs”:
 1. MockRepository (pure-Python, in-memory) - used by unit tests and can be wired in for local dev.
 2. FirestoreClient - real Cloud Firestore; for integration/E2E you can point it at Google’s local Emulator Suite if you don’t want to hit GCP.

⸻

How the repo is wired

File / package Purpose Local-dev behaviour
src/clarity/storage/mock_repository.py In-memory stub implementing the same Repository interface. Default for unit tests (tests/unit/*). No external service needed.
src/clarity/storage/firestore_client.py Concrete adapter to Google Cloud Firestore via google-cloud-firestore. Looks for GOOGLE_CLOUD_PROJECT; if absent, falls back to FIRESTORE_EMULATOR_HOST → works with the local emulator.
tests/integration/* Use MockRepository or parametrize to FirestoreClient if env var USE_EMULATOR=1. Lets CI spin up an emulator container for “real” reads/writes.
Makefile / scripts/* There’s a make emulate-firestore target that runs the emulator via Docker (if you have Cloud SDK). One-liner to get a local Firestore instance.

⸻

Typical dev/test flow

Task Repo default What you can do
Run unit tests (pytest -m unit) Uses MockRepository; no DB. Fast, deterministic.
Local manual runs (uvicorn src.clarity.main:app) Set env STORAGE_BACKEND=mock → uses in-mem. Iterate without Cloud creds.
Integration tests / E2E - If USE_EMULATOR=1 → spins Firestore emulator- Else hits real Firestore (requires service-account JSON or gcloud auth login). Safer/cheaper to use emulator in CI.
CI pipeline Docker service gcloud-firebase-emulator already declared in /tests/e2e.md. No billing surprises.

⸻

How to run the local Firestore emulator (if you want)

# One-time: pull Cloud SDK image

docker pull gcr.io/google.com/cloudsdktool/cloud-sdk:slim

# From repo root

make emulate-firestore

# or, manually

docker run --rm -it \
  -p 8080:8080 \
  -e FIRESTORE_EMULATOR_HOST=localhost:8080 \
  gcr.io/google.com/cloudsdktool/cloud-sdk:slim \
  gcloud beta emulators firestore start --host-port=0.0.0.0:8080

Then export:

export FIRESTORE_EMULATOR_HOST=localhost:8080
export STORAGE_BACKEND=firestore

Run API or tests; they’ll talk to the emulator.

⸻

When would you need a real Firebase project?
 • Only for staging/production deployments, or if you specifically want to test Google-managed IAM rules, indexing, or quota behaviour.
 • Local development + CI can stay 100 % offline with the emulator or the mock repo.

⸻

Bottom line

The skeleton already covers every stage:
 • MockRepository → lightning-fast unit dev.
 • Firestore emulator → realistic integration tests, no cloud bill.
 • Real Firestore → prod/Staging.

So you’re free to code and test without spinning up a local database—unless you want emulator-level realism, which is a single make command away.
