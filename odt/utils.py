import math

import numpy as np
from numba import njit


@njit
def entropy(classes):
    """calculates of the entropy (impurity criterion) in the current node

    Arguments:
        classes {numpy.ndarray} -- number of occurrences of each class in the
        current node

    Returns:
        float -- entropy calculated
    """
    total = np.sum(classes)
    entropy = 0.0

    for p in classes:
        if p == 0:
            continue
        else:
            p /= total
            entropy -= p * np.log2(p)

    return entropy


@njit
def gini(classes):
    """calculates of the gini (impurity criterion) in the current node

    Arguments:
        classes {numpy.ndarray} -- number of occurrences of each class in the
        current node

    Returns:
        float -- gini calculated
    """
    total = np.sum(classes)
    sum_freq = 0.0

    for p in classes:
        if p == 0:
            continue
        else:
            p /= total
            sum_freq += p ** 2

    return 1 - sum_freq


@njit
def unique_counts(labels, unique_classes):
    """count number of occurrences of each class

    Args:
        array_classes (numpy.ndarray): all classes
        classes_ (numpy.ndarray): each classes contained in array_classes

    Returns:
        numpy.ndarray: number of ocurrences of each class in array_classes
    """
    count_classes = np.zeros_like(unique_classes)

    for label in labels:
        count_classes[label] += 1

    return count_classes


@njit
def best_in_column(X, y, index, criterion, unique_classes):
    """Return the best impurity in column selected using the impurity
    criterion specified.

    Arguments:
        X {numpy.ndarray} -- dataset for to evaluate the best impurity
        y {numpy.ndarray} -- classes for to evaluate the best impurity
        index {int} -- Index of the column to be analyzed
        criterion {str} -- impurity criterion ("entropy" or "gini")
        classes_ {numpy.ndarray} -- Unique classes

    Returns:
        Tuple -- A tuple containing of the best threshold value and the best
        impurity value
    """
    if index >= X.shape[1]:
        raise ValueError("Invalid Index!")

    best_gain, best_gini = -math.inf, -math.inf
    best_threshold = None
    n_samples = X.shape[0]
    count_classes = unique_counts(y, unique_classes)

    if criterion == "entropy":
        total_impurity = entropy(count_classes)
    else:
        total_impurity = gini(count_classes)

    count_right = count_classes
    count_left = np.zeros_like(unique_classes)
    sum_right = np.sum(count_right)
    sum_left = 0

    column = np.zeros((n_samples, 2))

    for i in range(n_samples):
        column[i, 0] = i
        column[i, 1] = X[i, index]

    column = column[column[:, 1].argsort()]

    i = 0

    while i < n_samples:
        count_right[y[int(column[i, 0])]] -= 1
        count_left[y[int(column[i, 0])]] += 1
        count = 0
        j = i + 1
        if j < n_samples:
            while column[i, 1] == column[j, 1]:
                count_right[y[int(column[j, 0])]] -= 1
                count_left[y[int(column[j, 0])]] += 1
                count += 1
                j += 1
                if j >= n_samples:
                    break
        sum_right -= count + 1
        sum_left += count + 1
        p1 = sum_left / float(n_samples)
        p2 = sum_right / float(n_samples)

        if criterion == "entropy":
            gain = total_impurity - (
                (p1 * entropy(count_left)) + (p2 * entropy(count_right))
            )
            info = -((p1 * np.log2(p1)) + (p2 * np.log2(p2)))
            gain_ratio = gain / info

            if gain_ratio > best_gain and sum_left >= 0 and sum_right >= 0:
                best_gain = gain_ratio
                best_threshold = column[i, 1]
        else:
            gini_split = total_impurity - (
                (p1 * gini(count_left)) + (p2 * gini(count_right))
            )
            if gini_split > best_gini and sum_left >= 0 and sum_right >= 0:
                best_gini = gini_split
                best_threshold = column[i, 1]

        i += count + 1

    if criterion == "entropy":
        return best_threshold, best_gain
    else:
        return best_threshold, best_gini


def check_X_y(X, y):
    """checks if the data set is in the correct shape

    Arguments:
        X {numpy.ndarray} -- original dataset
        y {numpy.ndarray} -- original classes

    Raises:
        ValueError: If the dataset is empty
        ValueError: If the dataset is incorrect shape

    Returns:
        Tuple(numpy.ndarray, numpy.ndarray) -- return samples and classes if shape is correct
    """
    if len(X) <= 0 or len(y) <= 0:
        raise ValueError("Empty Database!")
    if len(np.shape(X)) != 2 or len(np.shape(y)) != 1:
        raise ValueError("Incorrect shape!")

    return X, y


@njit
def add_virtual_feature(X, weights):
    """Add a column containing a virtual feature calculated like a sum
    of the line with weights

    Arguments:
        X {numpy.ndarray} -- Dataset for add a column with virtual feature
        weights {numpy.ndarray} -- Array of the weights for calculation of the virtual feature

    Returns:
        numpy.ndarray -- Dataset containing a column with virtual feature
    """
    n_features = X.shape[1] - 1

    for j in range(X.shape[0]):
        s = 0.0
        for i in range(n_features):
            s += X[j, i] * weights[i]
        X[j, n_features] = s


@njit
def del_last_column(X):
    """delete a last column of the array

    Arguments:
        X {numpy.ndarray} -- Dataset for the remove last column

    Returns:
        numpy.ndarray -- Dataset without last colum
    """

    return X[:, :-1]


@njit
def find_cut_point(X, y, criterion, unique_classes):
    """Search for the best point for tree division

    Args:
        X (numpy.ndarray): Data
        y (numpy.ndarray): Classes
        criterion (str): impurity criterion to be used (entropy/gini)
        classes_ (numpy.ndarray): each classes containing in the node

    Returns:
        Tuple[int, float]: a tuple containing index and value of the
        best cut point
    """
    best_index, best_threshold, best_impurity = (
        -1,
        None,
        -math.inf,
    )
    for i in range(X.shape[1]):
        aux_threshold, aux_gain = best_in_column(X, y, i, criterion, unique_classes)
        if aux_gain > best_impurity:
            best_impurity = aux_gain
            best_threshold = aux_threshold
            best_index = i

    return best_index, best_threshold
