FROM python:3.10-slim

# Build-time argument — the MLflow Run ID of the registered model
ARG RUN_ID

# Make it available as an env var at runtime too
ENV RUN_ID=${RUN_ID}

WORKDIR /app

# Install only what's needed to pull the model at startup
RUN pip install --no-cache-dir mlflow boto3

# Simulate downloading the model artifact for the run
RUN echo "Downloading model for Run ID: ${RUN_ID}"

# Entry point — in a real setup this would serve the model
CMD echo "Serving model for Run ID: ${RUN_ID}"