from __future__ import annotations

from enum import Enum

import numpy as np

from slsdt.utils import (
    apply_weights,
    calc_impurity,
    entropy,
    gini,
    make_initial_weights,
    more_zeros,
)


class Movement(Enum):
    INCREASE = 1
    MULTIPLE_INCREASE = 2
    SWAP = 3
    ZERO = 4
    RESET = 5


class Node:
    def __init__(
        self,
        weights=np.empty(0),
        is_leaf=None,
        results=np.empty(0),
        children_left=None,
        children_right=None,
        error=-1,
    ):

        self.weights = weights
        self.is_leaf = is_leaf
        self.results = results
        self.children_left = children_left
        self.children_right = children_right
        self.error = error


class SLSDT:
    """
    A oblique decision tree using LAHC heuristic for find best split in
    each node of the tree
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: int = 8,
        max_samples: int = 10000,
        min_samples_split: int = 4,
        min_samples_leaf: int = 7,
        max_iterations: int = 1000000,
        l: int = 10,
        increase: float = 0.0,
        multiple_increase: float = 0.75,
        swap: float = 0.1,
        zero: float = 0.1,
        reset: float = 0.1,
        seed: int = 42,
    ):

        self.criterion = criterion
        self.max_depth = max_depth
        self.max_samples = max_samples
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_iterations = max_iterations
        self.l = l
        self.increase = increase
        self.multiple_increase = multiple_increase
        self.swap = swap
        self.zero = zero
        self.reset = reset
        self.rng = np.random.default_rng(seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> SLSDT:
        """
        build a oblique decision tree classifier

        Arguments:
            X {numpy.ndarray} -- records for training
            y {numpy.ndarray} -- class labels for training

        Returns:
            SLSDT: return the classifier
        """

        X, y = self.__check_X_y(X, y)

        self.X = X
        self.y = y
        self.unique_classes = np.unique(y)
        self.n_classes = self.unique_classes.shape[0]
        self.n_features = X.shape[1]
        self.__calc_percentage_movements()
        self.criterion = entropy if self.criterion == "entropy" else gini
        self.tree = self.__make_tree(X, y)
        self.__prune(self.tree)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        predict unknown class labels

        Args:
            X (numpy.ndarray): data to be predict

        Returns:
            numpy.ndarray: class labels predicted
        """
        return self.__classify(self.tree, X)

    def get_params(self, deep=True):
        """
        get params of the class

        Args:
            deep (bool, optional): Defaults to True.
        Returns:
            Dict: params of the class
        """
        return {
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "max_samples": self.max_samples,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_iterations": self.max_iterations,
            "l": self.l,
            "increase": self.increase,
            "multiple_increase": self.multiple_increase,
            "swap": self.swap,
            "zero": self.zero,
            "reset": self.reset,
        }

    def set_params(self, **parametes):
        """
        Set params of the class

        Returns:
            SLSDT: return the classifier
        """
        for parameter, value in parametes.items():
            setattr(self, parameter, value)

        return self

    def print_tree(self):
        """prints the decision tree built"""

        self.__aux_print_tree(self.tree, 1)

    @staticmethod
    def __check_X_y(X, y):
        if len(X) <= 0 or len(y) <= 0:
            raise ValueError("Empty Database!")
        if len(np.shape(X)) != 2 or len(np.shape(y)) != 1:
            raise ValueError("Incorrect shape!")

        return X, y

    def __calc_percentage_movements(self):
        self.percentage_movements = [
            (Movement.INCREASE, self.increase),
            (Movement.MULTIPLE_INCREASE, self.multiple_increase),
            (Movement.SWAP, self.swap),
            (Movement.ZERO, self.zero),
            (Movement.RESET, self.reset),
        ]

        self.percentage_movements = sorted(
            self.percentage_movements, key=lambda x: x[1], reverse=True
        )

        sum_percentages = sum([x[1] for x in self.percentage_movements])

        self.percentages_transform = list(
            map(lambda x: x[1] / sum_percentages, self.percentage_movements)
        )

        for i in range(1, len(self.percentages_transform)):
            self.percentages_transform[i] += self.percentages_transform[i - 1]

    def __build_movement(self):
        x = self.rng.random()

        if x <= self.percentages_transform[0]:
            return self.percentage_movements[0][0]
        elif x <= self.percentages_transform[1]:
            return self.percentage_movements[1][0]
        elif x <= self.percentages_transform[2]:
            return self.percentage_movements[2][0]
        elif x <= self.percentages_transform[3]:
            return self.percentage_movements[3][0]
        else:
            return self.percentage_movements[4][0]

    def __make_movement(self, weights, movement):
        weights = np.copy(weights)

        if movement == Movement.INCREASE:
            column_modified = self.rng.integers(weights.shape[0], size=1)[0]
            value_increase = (1 - -1) * self.rng.random() + -1
            weights[column_modified] += value_increase

        elif movement == Movement.MULTIPLE_INCREASE:
            number_columns_modified = self.rng.integers(
                1, weights.shape[0] + 1, size=1
            )[0]
            columns_modified = self.rng.choice(
                weights.shape[0], size=number_columns_modified, replace=False
            )
            for column_modified in columns_modified:
                value_increase = (1 - -1) * self.rng.random() + -1
                weights[column_modified] += value_increase

        elif movement == Movement.SWAP:
            column1, column2 = self.rng.choice(weights.shape[0], size=2, replace=False)
            weights[column1], weights[column2] = np.copy(weights[column2]), np.copy(
                weights[column1]
            )

        elif movement == Movement.ZERO:
            column_modified = self.rng.integers(weights.shape[0], size=1)[0]
            weights[column_modified] = 0

        elif movement == Movement.RESET:
            weights = np.copy(self.initial_weights)

        return weights

    def __lahc(self, X, y, frequencies_y):
        if X.shape[0] > self.max_samples:
            random_indexes = self.rng.choice(
                X.shape[0],
                size=self.max_samples,
                replace=False,
            )

            X = np.copy(X[random_indexes])
            y = np.copy(y[random_indexes])

        weights = make_initial_weights(X, y, self.criterion, frequencies_y)
        self.initial_weights = np.copy(weights)

        weights_final = np.copy(weights)

        cost = calc_impurity(
            X,
            y,
            weights_final,
            self.criterion,
            frequencies_y,
            self.min_samples_leaf,
        )

        cost_final = np.copy(cost)

        costs = [cost for _ in range(self.l)]

        iteration, v = 0, 0

        while iteration < self.max_iterations:
            weights_neighbor = self.__make_movement(weights, self.__build_movement())

            cost_neighbor = calc_impurity(
                X,
                y,
                weights_neighbor,
                self.criterion,
                frequencies_y,
                self.min_samples_leaf,
            )

            if cost_neighbor >= cost or cost_neighbor >= costs[v]:
                weights = np.copy(weights_neighbor)
                cost = np.copy(cost_neighbor)

                if cost > cost_final:
                    weights_final = np.copy(weights)
                    cost_final = np.copy(cost)
                elif cost == cost_final:
                    if more_zeros(weights, weights_final):
                        weights_final = np.copy(weights)
                        cost_final = np.copy(cost)

            costs[v] = cost
            v = (v + 1) % self.l
            iteration += 1

        return weights_final, cost_final

    def __stopping_criterion(self, n_classes, depth, samples_node):
        return (
            self.max_depth == depth
            or n_classes == 1
            or samples_node <= self.min_samples_split
        )

    def __make_tree(self, X, y, depth=1):
        if X.shape[0] == 0 or y.shape[0] == 0:
            return Node()

        classes, count_classes = np.unique(y, return_counts=True)

        error = np.sum(np.delete(count_classes, np.argmax(count_classes)))

        frequencies_y = np.zeros(self.n_classes)
        frequencies_y[classes] = count_classes

        n_classes = classes.shape[0]

        if not self.__stopping_criterion(n_classes, depth, X.shape[0]):
            weights, _ = self.__lahc(X, y, frequencies_y)

            split = np.array([apply_weights(record, weights) > 0 for record in X])

            if (
                np.sum(split) <= self.min_samples_leaf
                or np.sum(~split) <= self.min_samples_leaf
                or np.sum(split) <= 0
                or np.sum(~split) <= 0
            ):
                return Node(is_leaf=True, results=frequencies_y, error=error)

            left = self.__make_tree(X[split], y[split], depth + 1)
            right = self.__make_tree(X[~split], y[~split], depth + 1)

            return Node(
                weights=weights,
                children_left=left,
                children_right=right,
                results=frequencies_y,
                error=error,
            )

        return Node(is_leaf=True, results=frequencies_y, error=error)

    def __prune(self, tree):
        if tree.is_leaf:
            return tree.error

        error_left = self.__prune(tree.children_left)
        error_right = self.__prune(tree.children_right)

        if tree.error <= error_left + error_right:
            tree.children_left = None
            tree.children_right = None
            tree.is_leaf = True
            return tree.error

        else:
            return error_left + error_right

    def __classify(self, node, X):
        if node.is_leaf:
            return np.zeros(X.shape[0]) + np.argmax(node.results)

        split = np.array([apply_weights(record, node.weights) > 0 for record in X])

        y_pred = np.zeros(X.shape[0])

        if split.sum() > 0:
            y_pred[split] = self.__classify(node.children_left, X[split])

        if (~split).sum() > 0:
            y_pred[~split] = self.__classify(node.children_right, X[~split])

        return y_pred

    def __aux_print_tree(self, node, depth):
        print(depth * " ", end="")
        if node.is_leaf:
            print(f"Predict: {node.results}")
        else:
            print(f"Weights: {node.weights}")

        if node.children_left:
            self.__aux_print_tree(node.children_left, depth + 1)
        if node.children_right:
            self.__aux_print_tree(node.children_right, depth + 1)
