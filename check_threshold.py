import mlflow
import sys
import os

THRESHOLD = 0.85

# Read Run ID written by train.py
with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

print(f"Checking Run ID: {run_id}")

# Fetch accuracy from MLflow
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)
accuracy = run.data.metrics.get("accuracy", 0.0)

print(f"Accuracy: {accuracy:.4f}  |  Threshold: {THRESHOLD}")

# Gate
if accuracy < THRESHOLD:
    print(f"FAILED: accuracy {accuracy:.4f} is below threshold {THRESHOLD}")
    sys.exit(1)

print("PASSED: accuracy meets threshold. Proceeding to deploy.")
