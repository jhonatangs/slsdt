# Oblique Decision Tree


Oblique decision tree is a algorithm for generate a machine learning method called decision tree using oblique approach. For oblique approach was created a auxiliary algorithm called Artificial Feature that adds one virtual feature in a database using simulated annealing metaheuristic.

## Examples

### Install Dependencies

```
pip install -r requirements.txt
```

### Reader csv

```
from reader_csv import read_csv

# passes the file path and the class index
X, y = read_csv("../instances/iris.csv", "class")
```

### Artificial Feature separete

```
import numpy as np

from reader_csv import read_csv
from af import ArtificialFeature

# passes the file path and the class index
X, y = read_csv("../instances/iris.csv", "class")

# can configure parameters
af = ArtificialFeature()
af.fit(X, y)

# return data with virtual feature, weights using and best impurity
X_with_virtual_feature, weights, impurity = af.sa(np.unique(y))
```

### Simulated Annealing Oblique Decisin Tree

```
import numpy as np
from sklearn.model_selection import cross_val_score

from reader_csv import read_csv
from saodt import SAODT

# passes the file path and the class index
X, y = read_csv("../instances/iris.csv", "class")

# can configure parameters
clf = SAODT()
clf.fit(X, y)

# predicts X and compares the results with y
print(clf.predict(X) == y)

# CROSS VALIDATION
# with clf already instantiated
results = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
print(results)
print(results.mean())
```