import math

import numpy as np
from numba import njit


@njit
def entropy(count_labels):
    aux_entropy = lambda x: x * np.log2(x)

    return -np.sum(aux_entropy(count_labels[count_labels != 0] / np.sum(count_labels)))


@njit
def gini(count_labels):
    return 1 - np.sum(np.square(count_labels / np.sum(count_labels)))


@njit
def best_in_column(X, y, index, criterion, frequencies_y):
    best_impurity = -math.inf
    best_threshold = None
    n_samples = X.shape[0]

    total_impurity = criterion(frequencies_y)

    count_right = np.copy(frequencies_y)
    count_left = np.zeros_like(frequencies_y)
    sum_right = np.sum(count_right)
    sum_left = 0

    column = np.zeros((n_samples, 2))

    for i in range(n_samples):
        column[i, 0] = i
        column[i, 1] = X[i, index]

    column = column[column[:, 1].argsort()]

    i = 0

    while i < n_samples - 1:
        count_right[y[int(column[i, 0])]] -= 1
        count_left[y[int(column[i, 0])]] += 1
        count = 0
        j = i + 1

        if j < n_samples - 1:
            while column[i, 1] == column[j, 1] and j < n_samples - 1:
                count_right[y[int(column[j, 0])]] -= 1
                count_left[y[int(column[j, 0])]] += 1
                count += 1
                j += 1

        sum_right -= count + 1
        sum_left += count + 1
        p1 = sum_left / float(n_samples)
        p2 = sum_right / float(n_samples)

        impurity = (
            total_impurity - p1 * criterion(count_left) - p2 * criterion(count_right)
        )

        if impurity > best_impurity and sum_left > 0 and sum_right > 0:
            best_impurity = impurity
            best_threshold = column[i, 1]

        i += count + 1

    return best_threshold, best_impurity


@njit
def best_split(X, y, criterion, frequencies_y):
    best_index, best_threshold, best_impurity = (
        -1,
        None,
        -math.inf,
    )
    for i in range(X.shape[1]):
        aux_threshold, aux_gain = best_in_column(X, y, i, criterion, frequencies_y)
        if aux_gain > best_impurity:
            best_impurity = aux_gain
            best_threshold = aux_threshold
            best_index = i

    return best_index, best_threshold


@njit
def make_initial_weights(X, y, criterion, frequencies_y):
    best_index, best_threshold = best_split(X, y, criterion, frequencies_y)

    weights = np.zeros(X.shape[1] + 1)
    weights[-1] = -best_threshold
    weights[best_index] = 1

    return weights


@njit
def apply_weights(record, weights):
    return np.sum(np.multiply(record, weights[:-1])) + weights[-1]


@njit
def calc_impurity(X, y, weights, criterion, frequencies_y, min_samples_leaf):
    count_left = np.zeros_like(frequencies_y)
    count_right = np.zeros_like(frequencies_y)

    for i in range(y.shape[0]):
        if apply_weights(X[i], weights) > 0:
            count_left[y[i]] += 1
        else:
            count_right[y[i]] += 1

    if (
        np.sum(count_left) <= min_samples_leaf
        or np.sum(count_right) <= min_samples_leaf
        or np.sum(count_left) <= 0
        or np.sum(count_right) <= 0
    ):
        return -math.inf

    total_frequencies_y = np.sum(frequencies_y)
    p1 = np.sum(count_left) / total_frequencies_y
    p2 = np.sum(count_right) / total_frequencies_y

    return (
        criterion(frequencies_y)
        - p1 * criterion(count_left)
        - p2 * criterion(count_right)
    )


@njit
def more_zeros(arr1, arr2):
    return np.count_nonzero(arr1 == 0) > np.count_nonzero(arr2 == 0)
