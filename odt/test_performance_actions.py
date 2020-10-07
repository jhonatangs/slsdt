import glob

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

from reader_csv import read_csv
from odt import ODT


clfs = [DecisionTreeClassifier(), ODT()]
instances = glob.glob("instances_actions/*.csv")
results = []

for instance in instances:
    print("Testando: ", instance)

    result = []
    result.append(instance)

    X, y = read_csv(instance, "class")

    for clf in clfs:
        cv = StratifiedKFold(n_splits=5)
        result.append(
            cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean()
        )

    results.append(result)

df = pd.DataFrame(results, columns=["Instance", "DecisionTreeClassifier", "DT"])
print(df)
df.to_csv("results/result_performance.csv")
