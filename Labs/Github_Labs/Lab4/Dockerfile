FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Only the serving dependencies (no GCS client, no pytest); scikit-learn must match the version used for training
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# The API code and the trained model (model.joblib is produced by the training step before docker build)
COPY src/app.py .
COPY model.joblib .

# Model version passed in at build time, returned by the API
ARG MODEL_VERSION=unknown
ENV MODEL_VERSION=${MODEL_VERSION}

# Serve the model; Cloud Run sets PORT (8080 by default)
EXPOSE 8080
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
