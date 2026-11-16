# Github Lab 4 – Changes

**Repo:** https://github.com/dhruvpuri0709/MLOps/tree/main/Labs/Github_Labs/Lab4

**Live API:** https://dhruv-iris-model-api-683887520659.us-east1.run.app/docs

Summary of how I set up GCP and GitHub for this lab, what I changed, and how to run it.

The original lab built a Docker image that **retrained** the model when run, and pushed it to Artifact Registry without running it. I changed it into a full CI/CD pipeline that **serves** the trained model as a public prediction API on **Cloud Run**:

```
push to main → tests → train + upload to GCS → build image → push to Artifact Registry → deploy to Cloud Run → smoke-test the live API
```

## 1. GCP setup

I used the `github-lab3` project and the service account from Github Lab 3.

- **APIs enabled:** Cloud Storage, Artifact Registry and Cloud Run Admin.
- **Service account roles:**
  - **Storage Admin:** upload models and the version file to the bucket.
  - **Artifact Registry Writer:** push Docker images.
  - **Cloud Run Admin:** deploy the service and make it public.
  - **Service Account User:** let Cloud Run run the service as the project's default account.
- **Bucket:** `github-actions-lab3`, reused from Github Lab 3. Models go under `trained_models/`, and the version counter is `model_version.txt`.
- **Artifact Registry:** a Docker repository `my-repo` in `us-east1`.

## 2. GitHub setup

In **Settings → Secrets and variables → Actions**:

| Type | Name | Value |
|---|---|---|
| Secret | `GCP_SA_KEY` | Service account JSON key |
| Secret | `GCP_PROJECT_ID` | `github-lab3` |
| Variable | `GCS_BUCKET_NAME` | `github-actions-lab3` |
| Variable | `VERSION_FILE_NAME` | `model_version.txt` |
| Variable | `GCP_REGION` | `us-east1` |
| Variable | `ARTIFACT_REPO` | `my-repo` |

## 3. Code changes

### New: model API ([`src/app.py`](src/app.py))

A FastAPI app that loads the trained `model.joblib` once at startup and serves it:

| Endpoint | What it does |
|---|---|
| `GET /` | Health check and the deployed model version |
| `POST /predict` | Takes 4 Iris measurements and returns the predicted class and per-class probabilities |
| `GET /docs` | Interactive page for trying predictions in the browser |

Inputs are validated: all four fields are required and must be greater than 0. Invalid requests get a `422` error.

```bash
curl -X POST https://dhruv-iris-model-api-683887520659.us-east1.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.8, "petal_width": 2.2}'
# {"class_id":2,"class_name":"virginica","probabilities":{"setosa":0.0,"versicolor":0.0,"virginica":1.0},"model_version":"4"}
```

### Dockerfile: serves instead of retraining

- Copies only `app.py` and the trained `model.joblib` into the image, and starts `uvicorn` on the port Cloud Run provides.
- Uses a separate, smaller [`requirements-serve.txt`](requirements-serve.txt) with no GCS client or pytest. scikit-learn is pinned to the training version so the model loads correctly.
- The model version is passed in at build time (`--build-arg MODEL_VERSION`), and the API returns it.
- Added a [`.dockerignore`](.dockerignore), so `.env`, `.venv` and tests are never copied into the image.

### Tests

- **New [`test/test_app.py`](test/test_app.py):** 4 API tests using FastAPI's test client: health, two known predictions (setosa and virginica), and rejection of bad input.
- **Fixed existing tests in [`test/test_pytest.py`](test/test_pytest.py):**
  - **`test_get_model_version`** passed by coincidence. The fake file content was set *after* the function call, and `int()` of an unset `MagicMock` is 1, which happened to equal the expected value. The fake content is now set before the call, using values (`'3'`, `'99'`) that can't match by accident.
  - **`test_train_model`** couldn't fail. It checked `hasattr(model, 'predict')`, which is true even for a model that was never trained. It now uses two classes and checks `classes_`, which only exists after `fit()`.
  - **`test_save_model_to_gcs`** left a real `model.joblib` in the project folder. It now runs in pytest's temporary folder.
  - Removed an unnecessary patch, an unused import and misleading comments.
- **Checking the fixes:** I deliberately broke the code three ways (version file ignored, `fit()` removed, missing file downloaded anyway). Each break now makes a test fail.

## 4. Workflow ([`.github/workflows/github_lab4_ci_cd.yml`](../../../.github/workflows/github_lab4_ci_cd.yml))

Runs on pushes to `main` that change Lab 4, or manually from the Actions tab.

| Step | What it does |
|---|---|
| Run tests | `pytest test/`. Nothing is trained or deployed if a test fails. |
| Train and save model | Trains, uploads to GCS and increments `model_version.txt`. Fails if the version wasn't updated. |
| Build and push image | `dhruv-iris-model-api:v<N>` and `:latest` to Artifact Registry |
| Deploy to Cloud Run | Deploys that exact version as a public service, with `--max-instances 2` to cap cost |
| Smoke test | Calls the live `/` and `/predict` and checks the returned model version matches the one just built |
| Summary | Shows the version, image and live URL on the run page |

## How to run

**On GitHub:** push a change under `Labs/Github_Labs/Lab4/`, or go to **Actions → CI/CD - Train, Build and Deploy Model API - Github Lab 4 → Run workflow**. The live URL appears on the run's summary page.

**Locally,** from `Labs/Github_Labs/Lab4`, with a `.env` containing `GCS_BUCKET_NAME` and `VERSION_FILE_NAME`:

```bash
uv venv --python 3.10 && source .venv/bin/activate
uv pip install -r requirements.txt
gcloud auth application-default login

pytest test/                          # 11 tests
python src/train_and_save_model.py    # train, upload, write model.joblib

# Build and run the API container
docker build --build-arg MODEL_VERSION=local -t iris-model-api .
docker run --rm -p 8080:8080 iris-model-api
# open http://localhost:8080/docs
```
