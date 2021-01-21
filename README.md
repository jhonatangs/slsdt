# SLSDT

Stochastic Local Search Decision Tree

This repository is for my first scientific initiation project.

## About

Decision tree is a predictive modelling aproach used in machine learning, data mining and statistics. In the decision tree each internal node represents a test on a feature and each terminal (or leaf) node represents a class label. Oblique Decision Tree is a variation of traditional decision trees, which allows multivariate tests in its internal nodes in the form of a combination of the features.

Our research, SLSDT, is a method for induction oblique decision trees using a stochastic local search method called Late Acceptance Hill-Climbing (LAHC) to try to find the best combination of features in each internal node.

This project also provides a utility to read csv files and convert to the format accepted by the SLSDT method.

## How to use

1. Install

```bash
pip3 install slsdt
```

2. read_csv

```python
from slsdt.reader_csv import read_csv

X, y = read_csv("some_file.csv", "class_column_name")
```

3. slsdt

```python
from slsdt.slsdt import SLSDT

clf = SLSDT()
clf.fit(X, y)

result = clf.predict(X)

print(result)
print(result == y)
```

## Iris example oblique split

```python
from sklearn import datasets
from slsdt.slsdt import SLSDT

iris = datasets.load_iris()
X = iris.data[:, :2] # we only take the sepal width and sepal length features.
y = iris.target

mark = y != 2

# we only take the 0 (Iris-setosa) and 1 (Iris-versicolor) class labels
X = X[mark]
y = y[mark]

clf = SLSDT()
clf.fit(X, y)
clf.print_tree()

result = clf.predict(X)

print(result)
print(result == y)
```

### Plot iris oblique split

![alt text](https://github.com/jhonatangs/slsdt/blob/main/oblique-split-iris.png "Iris oblique split")

Plot with Matplotlib using the results obtained above.

## How to contribute

-   Leave the :star: if you liked the project
-   Fork this project
-   Cloner your fork: `git clone your-fork-url && cd slsdt`
-   Create a branch with your features: `git checkout -b my-features`
-   Commit your changes: `git commit -m 'feat: My new features'`
-   Send the your branch: `git push origin my-features`

## License

This project is licensed under the EPL 2.0 License - see the [LICENSE](https://github.com/jhonatangs/slsdt/blob/main/LICENSE) file for details.
