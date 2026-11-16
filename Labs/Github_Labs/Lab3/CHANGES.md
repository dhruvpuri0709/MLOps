# Github Lab 3 – Changes

**Repo:** https://github.com/dhruvpuri0709/MLOps/tree/main/Labs/Github_Labs/Lab3

Summary of how I set up GCP and GitHub for this lab, what I changed in the code and workflow, and how to run it.

## 1. GCP setup

1. **Project:** created a new GCP project.
2. **Service account:** created a service account for GitHub Actions to use, instead of my own Owner account. It has only the **Storage Admin** role, so the credential stored in GitHub can only access Cloud Storage.
3. **JSON key:** generated a key for the service account. It isn't committed to the repo.
4. **Bucket:** created the GCS bucket `github-actions-lab3`.

## 2. GitHub setup

In **Settings → Secrets and variables → Actions**:

| Type | Name | Value |
|---|---|---|
| Secret | `GCP_SA_KEY` | The full service-account JSON key |
| Variable | `GCS_BUCKET_NAME` | `github-actions-lab3` |

- **The key:** the `google-github-actions/auth@v2` step reads `GCP_SA_KEY`, which lets `storage.Client()` authenticate.
- **The bucket name:** the training step passes `GCS_BUCKET_NAME` to the script as an environment variable, and the script reads it with `os.getenv`. No `.env` file is needed.

## 3. Code changes: champion/challenger model promotion

**File: [`train_and_save_model.py`](train_and_save_model.py)**

The original script uploaded every model to the same folder, including models that were worse than earlier ones. Now every model is kept as its own version, and only one at least as good as the current best becomes the **champion**:

```
gs://github-actions-lab3/
├── models/
│   └── <version>/            # one folder per run (UTC timestamp)
│       ├── model.joblib
│       └── metrics.json      # version, accuracy, train/test sizes, created_at
└── champion/                 # current best model
    ├── model.joblib
    └── metrics.json
```

Each run:
1. **Trains and scores** the Random Forest on Iris, measuring accuracy on the test split.
2. **Uploads** the model and `metrics.json` to `models/<version>/`.
3. **Reads** `champion/metrics.json` from the bucket.
4. **Promotes or keeps:** if there's no champion yet, or the new accuracy is at least the champion's, it copies this version into `champion/`. Otherwise it keeps the current champion and logs why.

**Other changes in the script:**
- **Upload errors fail the run.** The original `try/except` only printed upload errors, so a failed upload still showed as a successful run.
- **The comparison metric is one setting.** `PRIMARY_METRIC = "accuracy"` at the top of the file can be changed in one line.
- **Versions use UTC timestamps**, so they're consistent between local runs and GitHub runners.

**Output of the first successful run:**

```
Model accuracy: 1.0000
✅ Model uploaded to gs://github-actions-lab3/models/20261102-161159/model.joblib
No champion yet, promoting this model as the first champion
🏆 Promoted 20261102-161159 to gs://github-actions-lab3/champion/
```

**Iris is an easy dataset:** test accuracy is always 1.0, so every new model ties and gets promoted. To see a challenger rejected, set the accuracy in `champion/metrics.json` in the bucket above 1.0, or switch to a harder dataset.

## 4. Workflow changes

**File: [`.github/workflows/github_lab3_gcp_connection.yaml`](../../../.github/workflows/github_lab3_gcp_connection.yaml)**

- **Bucket name:** the train step now has `env: GCS_BUCKET_NAME: ${{ vars.GCS_BUCKET_NAME }}`.
- **Triggers:**
  - **Push:** runs on every push to `main`. A `paths:` filter limiting it to Lab 3 changes is included but commented out.
  - **Manual:** `workflow_dispatch` lets it be run from the Actions tab.
  - **Schedule:** the daily midnight run is commented out.

## How to run

**On GitHub:** push to `main`, or go to **Actions → Train and save model to GCS → Run workflow**.

**Locally,** from `Labs/Github_Labs/Lab3`:

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

gcloud auth application-default login      # or: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
export GCS_BUCKET_NAME=github-actions-lab3
python train_and_save_model.py
```

To list the uploaded models:

```bash
gcloud storage ls -r gs://github-actions-lab3/
```
