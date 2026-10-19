# Lab 2 – Changes

Summary of what I changed from the original lab, and how to run it.

## 1. Added model calibration

**New file: [`src/calibrate_model.py`](src/calibrate_model.py)**

- Loads the trained Random Forest and keeps it frozen (`FrozenEstimator`). Only a score → probability mapping is fitted on a separate calibration set (`CalibratedClassifierCV`, sigmoid / Platt scaling by default, isotonic optional).
- Compares the **base** and **calibrated** models on a held-out test set:
  - Probability quality: Brier score, log loss, Expected Calibration Error (ECE)
  - Classification quality: ROC AUC, F1, accuracy
- Outputs:
  - `models/model_<ts>_calibrated.joblib`
  - `metrics/<ts>_calibration_metrics.json` (metrics plus reliability-curve points)
  - MLflow run in the `Model Calibration` experiment
  - A results table on the GitHub Actions run summary page

**Example result** (797 test rows, sigmoid):

| Metric | Base | Calibrated |
|---|---|---|
| Brier score ↓ | 0.0667 | 0.0624 |
| Log loss ↓ | 0.3187 | 0.2222 |
| ECE ↓ | 0.0452 | 0.0141 |
| ROC AUC ↑ | 0.9656 | 0.9656 |
| F1 ↑ | 0.9135 | 0.9168 |

The probabilities get more reliable. ROC AUC is unchanged, as expected: calibration is monotonic, so it doesn't change how the model ranks samples.

## 2. Changed `src/train_model.py`

- **Train / calibration / test split (60 / 20 / 20).** `make_classification` produces a *different* problem for every `n_samples` or `random_state`: a model trained on one generated dataset scored about 0.50 accuracy on another. So the script now generates one dataset and splits it, and saves the held-out splits to `data/calibration.pickle` and `data/test.pickle`.
- `n_samples` changed from `randint(0, 2000)`, which could produce 0 or very few rows, to `randint(2000, 4000)`.
- The MLflow tracking store changed from `./mlruns` to `sqlite:///mlflow.db`. The file store is rejected by default in current MLflow (3.x).

## 3. GitHub Actions workflows (`.github/workflows/`)

| Workflow | Trigger | What it does |
|---|---|---|
| `github_lab2_model_calibration_evaluation.yml` **(new)** | Push or PR touching Lab2, or manual | Train → calibrate → compare. Uploads the metrics and calibrated model as artifacts. |
| `github_lab2_model_training_evaluation_on_push.yml` | Push to `main` | Train → evaluate → commit the model and metrics back to the repo |
| `github_lab2_model_training_evaluation.yml` | Manual only | Same as above. The daily cron is disabled. |

- Renamed the two original workflows from `model_calibration*` to `model_training_evaluation*`, since they train and evaluate but don't calibrate.
- Added `permissions: contents: write` so the workflow can push its results. On a fork, the push previously failed with a 403 error.
- Commits from the workflows are now made under my GitHub account instead of the original author's.
- Disabled the daily schedule for Lab 2 and Lab 3, and removed the Lab 4 CI/CD pipeline. Those workflows were failing on this fork because the GCP secrets aren't set up.

## How to run locally

From `Labs/Github_Labs/Lab2`:

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

ts=$(date +%Y%m%d%H%M%S)
python src/train_model.py --timestamp "$ts"        # train base model + save splits
python src/calibrate_model.py --timestamp "$ts"    # calibrate + compare (add --method isotonic to try isotonic)

mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001   # open http://127.0.0.1:5001
```

On GitHub, push a change under `Labs/Github_Labs/Lab2/`, or run **Model Calibration and Evaluation – Github Lab 2** from the Actions tab. The results table appears on the run's summary page.
