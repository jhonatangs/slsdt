.. _`quickstart`:

==========
Quickstart
==========

In this tutorial, we'll assume that slsdt is already installed on your system.
If that's not the case, see :ref:`intro-install`.

Training and testing iris dataset from the `scikit-learn`_ library
===================================================================

Example of how to train and test the slsdt with a simple database.

.. code-block:: python

    from slsdt.slsdt import SLSDT
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split


    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = SLSDT()
    clf.fit(X_train, y_train)
    results = clf.predict(X_test)
    print(f"Accuracy: {sum(results == y_test) / len(y_test)}")


Testing slsdt with cross validation from the `scikit-learn`_ library
====================================================================

Example of how to test slsdt using 5-fold cross validation

.. code-block:: python

    from slsdt.slsdt import SLSDT
    from sklearn.datasets import load_iris
    from sklearn.model_selection import cross_val_score


    clf = SLSDT()
    iris = load_iris()
    print(cross_val_score(clf, iris.data, iris.target, cv=5, scoring="accuracy"))


.. _scikit-learn: https://scikit-learn.org/stable/index.html