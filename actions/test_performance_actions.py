import glob

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

from slsdt.slsdt import SLSDT
from slsdt.reader_csv import read_csv


clfs = [DecisionTreeClassifier(), SLSDT()]
instances = glob.glob("actions/*.csv")
results = []

for instance in instances:
    print("Testando: ", instance)

    result = []
    result.append(instance)

    X, y = read_csv(instance, "class")

    result.append(X.shape[0])
    result.append(X.shape[1])
    result.append(np.unique(y).shape[0])

    for clf in clfs:
        cv = StratifiedKFold(n_splits=5)
        result.append(
            cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean()
        )

    results.append(result)

df = pd.DataFrame(
    results,
    columns=[
        "Instance",
        "Number Of Objects",
        "Number Of Features",
        "Number Of Classes",
        "DecisionTreeClassifier",
        "SLSDT",
    ],
)
df = df.round(3)
print(df)
df.to_csv("results/result_performance_slsdt.csv")