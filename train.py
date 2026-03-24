import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# ── Load data ──────────────────────────────────────────────────────────────
data = pd.read_csv("data/iris.csv")
# Last column is always the target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Train ──────────────────────────────────────────────────────────────────
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("assignment5")

with mlflow.start_run() as run:
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(clf, "model")

    print(f"Accuracy: {acc:.4f}")
    print(f"Run ID:   {run.info.run_id}")

    # ── Export Run ID for the deploy job ──────────────────────────────────
    with open("model_info.txt", "w") as f:
        f.write(run.info.run_id)