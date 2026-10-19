import argparse, json, os, pickle
import joblib
import mlflow
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss, roc_auc_score


def expected_calibration_error(y_true, y_prob, n_bins=10):
    # Weighted average gap between predicted probability and actual frequency, per bin
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bin_ids == b
        if in_bin.any():
            ece += in_bin.mean() * abs(y_prob[in_bin].mean() - y_true[in_bin].mean())
    return ece


def compute_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        # Probability quality (lower is better)
        "Brier_Score": brier_score_loss(y_true, y_prob),
        "Log_Loss": log_loss(y_true, y_prob),
        "ECE": expected_calibration_error(y_true, y_prob),
        # Ranking / classification quality
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "F1_Score": f1_score(y_true, y_pred),
        "Accuracy": accuracy_score(y_true, y_pred),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    parser.add_argument("--method", type=str, default="sigmoid", choices=["sigmoid", "isotonic"],
                        help="sigmoid = Platt scaling (stable on small data), isotonic = step function (needs more data)")
    args = parser.parse_args()
    timestamp = args.timestamp

    # Load the trained (uncalibrated) model produced by train_model.py
    model_filename = f'model_{timestamp}_dt_model.joblib'
    try:
        model = joblib.load(model_filename)
    except FileNotFoundError:
        raise ValueError(f'Failed to find the trained model {model_filename}, run train_model.py first')

    # Held-out splits saved by train_model.py (same distribution as the training data)
    with open('data/calibration.pickle', 'rb') as f:
        X_cal, y_cal = pickle.load(f)
    with open('data/test.pickle', 'rb') as f:
        X_test, y_test = pickle.load(f)

    # Fit only the score -> probability mapping on the calibration set; the forest is not retrained
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=args.method)
    calibrated.fit(X_cal, y_cal)

    # Compare both models on the test set, which neither has seen
    base_prob = model.predict_proba(X_test)[:, 1]
    cal_prob = calibrated.predict_proba(X_test)[:, 1]
    base_metrics = compute_metrics(y_test, base_prob)
    cal_metrics = compute_metrics(y_test, cal_prob)

    # Reliability curve points: in each bin, mean predicted probability vs actual fraction of class 1
    base_true, base_pred = calibration_curve(y_test, base_prob, n_bins=10)
    cal_true, cal_pred = calibration_curve(y_test, cal_prob, n_bins=10)

    results = {
        "timestamp": timestamp,
        "method": args.method,
        "n_calibration": int(len(y_cal)),
        "n_test": int(len(y_test)),
        "base": base_metrics,
        "calibrated": cal_metrics,
        "reliability_curve": {
            "base": {"mean_predicted": base_pred.tolist(), "fraction_positive": base_true.tolist()},
            "calibrated": {"mean_predicted": cal_pred.tolist(), "fraction_positive": cal_true.tolist()},
        },
    }

    os.makedirs('models', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)
    joblib.dump(calibrated, f'models/model_{timestamp}_calibrated.joblib')
    with open(f'metrics/{timestamp}_calibration_metrics.json', 'w') as f:
        json.dump(results, f, indent=4)

    # Report
    lower_is_better = {"Brier_Score", "Log_Loss", "ECE"}
    rows = []
    for name in base_metrics:
        b, c = base_metrics[name], cal_metrics[name]
        better = (c < b) if name in lower_is_better else (c > b)
        rows.append((name, b, c, c - b, "yes" if better else ("same" if c == b else "no")))

    print(f"\nCalibration method: {args.method} | calibration rows: {len(y_cal)} | test rows: {len(y_test)}")
    print(f"{'Metric':<12}{'Base':>10}{'Calibrated':>12}{'Change':>10}  Improved")
    for name, b, c, d, imp in rows:
        print(f"{name:<12}{b:>10.4f}{c:>12.4f}{d:>+10.4f}  {imp}")
    print("Brier, Log Loss, ECE: lower is better. ROC AUC, F1, Accuracy: higher is better.")

    # Show the same table on the GitHub Actions run page
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, 'a') as f:
            f.write(f"## Calibration results ({args.method})\n\n")
            f.write(f"Calibration rows: {len(y_cal)}, test rows: {len(y_test)}\n\n")
            f.write("| Metric | Base | Calibrated | Change | Improved |\n|---|---|---|---|---|\n")
            for name, b, c, d, imp in rows:
                f.write(f"| {name} | {b:.4f} | {c:.4f} | {d:+.4f} | {imp} |\n")
            f.write("\nBrier, Log Loss, ECE: lower is better. ROC AUC, F1, Accuracy: higher is better.\n")

    # Log to MLflow alongside the training runs
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Model Calibration")
    with mlflow.start_run(run_name=f"calibration_{timestamp}"):
        mlflow.log_params({"timestamp": timestamp, "method": args.method,
                           "n_calibration": len(y_cal), "n_test": len(y_test)})
        mlflow.log_metrics({f"base_{k}": v for k, v in base_metrics.items()})
        mlflow.log_metrics({f"calibrated_{k}": v for k, v in cal_metrics.items()})
