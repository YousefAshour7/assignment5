"""Run this once locally to create data/iris.csv before DVC tracking."""
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris(as_frame=True)
df = iris.frame
df.columns = [*iris.feature_names, "target"]
df.to_csv("data/iris.csv", index=False)
print("data/iris.csv created!")