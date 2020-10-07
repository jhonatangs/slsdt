# Oblique Decision Tree


Oblique Decision Tree is a algorithm for generate a machine learning method called decision tree using oblique approach.

## Examples

### Install Dependencies

```
pip install -r requirements.txt
```

### Reader csv

```
from reader_csv import read_csv

# passes the file path and the class index
X, y = read_csv("../instances_actions/iris.csv", "class")
```

### Oblique Decisin Tree

```
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold

from reader_csv import read_csv
from saodt import SAODT

# passes the file path and the class index
X, y = read_csv("../instances_actions/iris.csv", "class")

# can configure parameters
clf = SAODT()

clf.fit(X, y)

# predicts X and compares the results with y
print(clf.predict(X) == y)

# CROSS VALIDATION
# with clf already instantiated
cv = StratifiedKFold(n_splits=5)
results = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
print(results)
print(results.mean())
```