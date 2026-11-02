import pandas as pd
import joblib
import io
import json
import os
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from google.cloud import storage

# Metric used to compare a new model (challenger) against the current best (champion)
PRIMARY_METRIC = "accuracy"
CHAMPION_PREFIX = "champion"

def download_data():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)
    return X, y

def preprocess_data(X, y):
    # random_state ensures reproducibility
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def upload_model(bucket, model, blob_name):
    """Saves model directly to GCS using an in-memory buffer."""
    # Use a buffer to avoid writing to local disk
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    bucket.blob(blob_name).upload_from_file(buffer, content_type='application/octet-stream')
    print(f"✅ Model uploaded to gs://{bucket.name}/{blob_name}")

def upload_json(bucket, data, blob_name):
    bucket.blob(blob_name).upload_from_string(json.dumps(data, indent=4), content_type='application/json')

def get_champion_metrics(bucket):
    """Returns the current champion's metrics, or None if there is no champion yet."""
    blob = bucket.blob(f"{CHAMPION_PREFIX}/metrics.json")
    if not blob.exists():
        return None
    return json.loads(blob.download_as_text())

def promote_to_champion(bucket, version):
    """Copies this version's model and metrics into champion/, replacing the previous champion."""
    for filename in ["model.joblib", "metrics.json"]:
        source = bucket.blob(f"models/{version}/{filename}")
        bucket.copy_blob(source, bucket, f"{CHAMPION_PREFIX}/{filename}")
    print(f"🏆 Promoted {version} to gs://{bucket.name}/{CHAMPION_PREFIX}/")

def main():
    # Use environment variables for flexibility
    bucket_name = os.getenv("GCS_BUCKET_NAME", "github-actions-lab3")

    # Execution steps
    X, y = download_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    model = train_model(X_train, y_train)

    # Evaluation
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f'Model accuracy: {accuracy:.4f}')

    # Versioning: every model is kept under models/<version>/
    now = datetime.now(timezone.utc)
    version = now.strftime("%Y%m%d-%H%M%S")
    metrics = {
        "version": version,
        "accuracy": accuracy,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "created_at": now.isoformat(),
    }

    # Upload errors are not caught, so a failed upload fails the GitHub Actions run
    bucket = storage.Client().bucket(bucket_name)
    upload_model(bucket, model, f"models/{version}/model.joblib")
    upload_json(bucket, metrics, f"models/{version}/metrics.json")

    # Champion / challenger: promote only if the new model is at least as good as the current best
    champion = get_champion_metrics(bucket)
    if champion is None:
        print("No champion yet, promoting this model as the first champion")
        promote_to_champion(bucket, version)
    elif metrics[PRIMARY_METRIC] >= champion[PRIMARY_METRIC]:
        print(f"Challenger {PRIMARY_METRIC} {metrics[PRIMARY_METRIC]:.4f} >= "
              f"champion {champion[PRIMARY_METRIC]:.4f} ({champion['version']}), promoting")
        promote_to_champion(bucket, version)
    else:
        print(f"Challenger {PRIMARY_METRIC} {metrics[PRIMARY_METRIC]:.4f} < "
              f"champion {champion[PRIMARY_METRIC]:.4f} ({champion['version']}), keeping current champion")

if __name__ == "__main__":
    main()
