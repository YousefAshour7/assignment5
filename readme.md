# Assignment 5 — Setup Guide

## 1. Create a DagsHub repo
1. Go to https://dagshub.com → **New Repository**
2. Name it `assignment5` and create it
3. From the repo page, click **Remote → Data** — copy the DVC remote URL  
   (looks like `https://dagshub.com/<username>/assignment5.dvc`)

---

## 2. Set up the project locally

```bash
cd assignment5

# Init git & DVC
git init
pip install dvc dvc-s3 scikit-learn pandas mlflow dagshub

dvc init

# Generate the dataset
python create_dataset.py

# Track data with DVC
dvc add data/iris.csv
git add data/iris.csv.dvc data/.gitignore .dvc/

# Point DVC at your DagsHub remote
dvc remote add origin https://dagshub.com/<YOUR_USERNAME>/assignment5.dvc
dvc remote modify origin auth basic
dvc remote modify origin user <YOUR_USERNAME>
dvc remote modify origin password <YOUR_DAGSHUB_TOKEN>

git add .dvc/config
```

---

## 3. Push data to DagsHub

```bash
dvc push
```

---

## 4. Push code to GitHub

Create a GitHub repo named `assignment5`, then:

```bash
git add .
git commit -m "initial commit"
git remote add origin https://github.com/<YOUR_GH_USERNAME>/assignment5.git
git push -u origin main
```

---

## 5. Add GitHub Secrets

Go to your GitHub repo → **Settings → Secrets and variables → Actions → New secret**

| Secret Name            | Value                                         |
|------------------------|-----------------------------------------------|
| `DAGSHUB_USERNAME`     | your DagsHub username                         |
| `DAGSHUB_TOKEN`        | DagsHub token (Settings → Tokens on DagsHub) |
| `MLFLOW_TRACKING_URI`  | `https://dagshub.com/<username>/assignment5.mlflow` |

---

## 6. Trigger a failing run (for the screenshot)

Temporarily edit `check_threshold.py` and change:
```python
THRESHOLD = 0.85
```
to:
```python
THRESHOLD = 0.999
```
Push → the deploy job will fail. Take your screenshot. Then revert to `0.85` for the passing run.