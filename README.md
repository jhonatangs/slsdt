This repository is for my first scientific initiation project.

# Oblique Decision Tree

Oblique Decision Tree is a algorithm for induction a machine learning method called decision tree using oblique approach.

## Examples

### Install Dependencies

```
pip install -r requirements.txt
```

### Example

```
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold

from reader_csv import read_csv
from odt import ODT

X, y = read_csv("instances_actions/iris.csv", "class")

clf = ODT()
clf.fit(X, y)
print(clf.predict(X) == y)

cv = StratifiedKFold(n_splits=5)
results = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
print(results.mean())
```
