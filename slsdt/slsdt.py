from __future__ import annotations

from math import ceil
from typing import Dict

import numpy as np
from graphviz import Digraph


from slsdt.utils import (
    apply_weights,
    calc_impurity,
    calc_penalty,
    entropy,
    gini,
    make_initial_weights,
)
from slsdt.Neighborhood import Neighborhood
from slsdt.Node import Node


class SLSDT:
    """
    Stochastic Local Search Decision Tree is a method for induction Oblique
    Decision Tree using Late Acceptance Hill-Climbing (LAHC) heuristic for
    to try to find best split in each internal node of the tree
    """

    def __init__(
        self,
        criterion: str = "entropy",
        max_depth: int = 8,
        max_samples: int or float = 10000,
        min_samples_split: int or float = 2,
        min_samples_leaf: int or float = 1,
        min_impurity_split: float = 0.1,
        max_iterations: int = 500000,
        l: int = 20,
        increase: float = 0.0,
        multiple_increase: float = 0.75,
        swap: float = 0.1,
        zero: float = 0.1,
        reset: float = 0.1,
        seed: int = 42,
    ) -> None:

        """SLSDT constructor

        Creates a Stochastic Local Decision Tree (SLSDT). The default
        parameters is provides for SMAC.

        Args:
            criterion (str): The function to measure the quality of a split.
                Supported criteria are "gini" for the Gini impurity and "entropy"
                for the Information Gain"
            max_depth (int): The maximum depth of the tree
            max_samples (int or float): The number of samples to consider when
                looking for the best split:\n
                If int, the consider max_samples at each split.\n
                If float, then consider max_samples like
                int(max_samples * n_samples) at each split
            min_samples_split (int or float): The minimum number of samples
                required to split an internal node:\n
                If int, then consider min_samples_split as the minimum number\n
                If float, then consider ceil(min_samples_split * n_samples)
                as the minimum number
            min_samples_leaf (int or float): The minimum number of samples
                required to be a leaf:\n
                If int, then consider min_samples_leaf as the minimum number\n
                If float, then consider ceil(min_samples_leaf * n_samples)
                as the minimum number
            min_impurity_split (float): The minimum impurity required to split
                a node.
            max_iterations (int): The maximum number of the iterations of LAHC
            l (int): The size of the LAHC list
            increase (float): The percentage of the neighborhood called
                "increase" in the LAHC heuristic. Adds a random value from
                the range [-1, 1) in a random position of the solution
            multiple_increase (float): The percentage of the neighborhood
                called "multiple_increase" in the LAHC heuristic. Adds n
                random values  from the range [-1, 1) at n random positions
                of the solution. Where n is in the range [1, n_features]
            swap (float): The percentage of the neighborhood called "swap"
                in the LAHC heuristic. Swaps the value of two random positions.
            zero (float): The percentage of the neighborhood called "zero" in
                the LAHC heuristic. Resets a random position
            reset (float): The percentage of the neighborhood called "reset" in
                the LAHC heuristic. Back the solution for better solution
                axis-parallel
            seed (int): A seed for random functions

        """

        self.criterion = criterion
        self.max_depth = max_depth
        self.max_samples = max_samples
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_split = min_impurity_split
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
        Build a oblique decision tree classifier from the training set (X, y)

        Arguments:
            X (ndarray): The training input samples.
            y (ndarray): The target values (class labels).

        Returns:
            self: Fitted estimator.
        """

        X, y = self.__check_X_y(X, y)

        self.X = X
        self.y = y

        self.min_samples_split = (
            self.min_samples_split
            if isinstance(self.min_samples_split, int)
            else ceil(self.min_samples_split * X.shape[0])
        )

        self.min_samples_leaf = (
            self.min_samples_leaf
            if isinstance(self.min_samples_leaf, int)
            else ceil(self.min_samples_leaf * X.shape[0])
        )

        self.max_samples = (
            self.max_samples
            if isinstance(self.max_samples, int)
            else ceil(self.max_samples * X.shape[0])
        )

        self.unique_classes = np.unique(y)
        self.n_classes = self.unique_classes.shape[0]
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.__calc_percentage_movements()
        self.criterion = entropy if self.criterion == "entropy" else gini
        self.tree = self.__make_tree(X, y)
        self.__prune(self.tree)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class value for X

        Predicted class for each sample in X

        Args:
            X (ndarray): The imput samples to be predict

        Returns:
            ndarray: The predicted classes
        """
        return self.__classify(self.tree, X)

    def text_tree(self) -> None:
        """Prints the tree in text format"""

        self.__aux_text_tree(self.tree, 1)

    def print_tree(self, features_names, target_names, filename="tree") -> Digraph:
        """Prints the tree in image format

        Args:
            features_names (ndarray): Features names
            target_names (ndarray): Unique class labels names
            filename (str, optional): Output file name. Defaults to "tree".

        Returns:
            Digraph: [description]
        """

        g = Digraph(
            "G",
            format="png",
            filename=filename,
            node_attr={"shape": "box"},
        )
        self.__aux_print_tree(self.tree, g, features_names, target_names)
        g.render()
        return g

    def get_params(self, deep=True) -> Dict:
        """
        Get parameters for this estimator

        Args:
            deep (bool, optional): Defaults to True. If True, will return the
                parameters for this estimator and contained subobjects that
                are estimators.
        Returns:
            dict: Parameter names mapped to their values.
        """
        return {
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "max_samples": self.max_samples,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_impurity_split": self.min_impurity_split,
            "max_iterations": self.max_iterations,
            "l": self.l,
            "increase": self.increase,
            "multiple_increase": self.multiple_increase,
            "swap": self.swap,
            "zero": self.zero,
            "reset": self.reset,
        }

    def set_params(self, **parametes) -> SLSDT:
        """
        Set params of this estimator

        Args:
            **parameters (dict): Estimator parameters

        Returns:
            self: Estimator instance
        """
        for parameter, value in parametes.items():
            setattr(self, parameter, value)

        return self

    @staticmethod
    def __check_X_y(X, y):
        if len(X) <= 0 or len(y) <= 0:
            raise ValueError("Empty Database!")
        if len(np.shape(X)) != 2 or len(np.shape(y)) != 1:
            raise ValueError("Incorrect shape!")

        return X, y

    def __calc_percentage_movements(self):
        self.percentage_movements = [
            (Neighborhood.INCREASE, self.increase),
            (Neighborhood.MULTIPLE_INCREASE, self.multiple_increase),
            (Neighborhood.SWAP, self.swap),
            (Neighborhood.ZERO, self.zero),
            (Neighborhood.RESET, self.reset),
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

        if movement == Neighborhood.INCREASE:
            column_modified = self.rng.integers(weights.shape[0] - 1, size=1)[0]
            value_increase = (1 - -1) * self.rng.random() + -1
            weights[column_modified] += value_increase

        elif movement == Neighborhood.MULTIPLE_INCREASE:
            number_columns_modified = self.rng.integers(1, weights.shape[0], size=1)[0]

            columns_modified = self.rng.choice(
                weights.shape[0] - 1, size=number_columns_modified, replace=False
            )

            for column_modified in columns_modified:
                value_increase = (1 - -1) * self.rng.random() + -1
                weights[column_modified] += value_increase

        elif movement == Neighborhood.SWAP:
            column1, column2 = self.rng.choice(
                weights.shape[0] - 1, size=2, replace=False
            )

            weights[column1], weights[column2] = np.copy(weights[column2]), np.copy(
                weights[column1]
            )

        elif movement == Neighborhood.ZERO:
            column_modified = self.rng.integers(weights.shape[0] - 1, size=1)[0]

            weights[column_modified] = 0

        elif movement == Neighborhood.RESET:
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

        cost = (
            calc_impurity(
                X,
                y,
                weights_final,
                self.criterion,
                frequencies_y,
            )
            - calc_penalty(weights_final)
        )

        cost_final = np.copy(cost)

        costs = [cost for _ in range(self.l)]

        iteration, v = 0, 0

        while iteration < self.max_iterations:
            weights_neighbor = self.__make_movement(weights, self.__build_movement())

            cost_neighbor = (
                calc_impurity(
                    X,
                    y,
                    weights_neighbor,
                    self.criterion,
                    frequencies_y,
                )
                - calc_penalty(weights_neighbor)
            )

            if cost_neighbor >= cost or cost_neighbor >= costs[v]:
                weights = np.copy(weights_neighbor)
                cost = np.copy(cost_neighbor)

                if cost > cost_final:
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
            weights, cost = self.__lahc(X, y, frequencies_y)

            split = np.array([apply_weights(record, weights) > 0 for record in X])

            if (
                np.sum(split) <= self.min_samples_leaf
                or np.sum(~split) <= self.min_samples_leaf
                or cost < self.min_impurity_split
            ):
                return Node(
                    is_leaf=True, results=frequencies_y, error=error, samples=X.shape[0]
                )

            left = self.__make_tree(X[split], y[split], depth + 1)
            right = self.__make_tree(X[~split], y[~split], depth + 1)

            return Node(
                weights=weights,
                children_left=left,
                children_right=right,
                results=frequencies_y,
                error=error,
                impurity=cost,
                samples=X.shape[0],
            )

        return Node(
            is_leaf=True, results=frequencies_y, error=error, samples=X.shape[0]
        )

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

    def __aux_text_tree(self, node, depth):
        print(depth * " ", end="")
        if node.is_leaf:
            print(f"Predict: {node.results}")
        else:
            print(f"Weights: {node.weights}")

        if node.children_left:
            self.__aux_text_tree(node.children_left, depth + 1)
        if node.children_right:
            self.__aux_text_tree(node.children_right, depth + 1)

    def __aux_print_tree(self, node, g, features_names, target_names):
        if not node.is_leaf:
            aux = ""
            for i in range(len(node.weights)):
                if i == len(node.weights) - 1:
                    aux += str(abs(node.weights[i]))
                    aux += " > 0"
                elif node.weights[i] != 0:
                    if node.weights[i] == -1:
                        aux += "-"
                    elif node.weights[i] != 1:
                        aux += (
                            str(abs(round(node.weights[i], 2)))
                            if i != 0
                            else str(round(node.weights[i], 2))
                        )
                    aux += str(features_names[i])
                    aux += " + " if node.weights[i + 1] > 0 else " - "

            aux += r"\n"
            aux += (
                str("gain" if self.criterion == entropy else "gini")
                + " = "
                + str(np.round(node.impurity, 2))
                + r"\n"
            )
            aux += "samples = " + str(node.samples) + r"\n"
            aux += "values = " + str(node.results)

            if node.children_left:
                if node.children_left.is_leaf:
                    leaf_left = ""
                    leaf_left += "samples = " + str(node.children_left.samples) + r"\n"
                    leaf_left += "values = " + str(node.children_left.results) + r"\n"
                    leaf_left += "class = " + str(
                        target_names[np.argmax(node.children_left.results)]
                    )
                    g.edge(aux, leaf_left)

                else:
                    aux1 = ""
                    for i in range(len(node.children_left.weights)):
                        if i == len(node.children_left.weights) - 1:
                            aux1 += str(abs(node.children_left.weights[i]))
                            aux1 += " > 0"
                        else:
                            if node.children_left.weights[i] == -1:
                                aux1 += "-"
                            elif node.children_left.weights[i] != 1:
                                aux1 += str(
                                    abs(round(node.children_left.weights[i], 2))
                                    if i != 0
                                    else str(round(node.children_left.weights[i], 2))
                                )
                            aux1 += str(features_names[i])
                            aux1 += (
                                " + "
                                if node.children_left.weights[i + 1] > 0
                                else " - "
                            )
                    aux1 += r"\n"
                    aux1 += (
                        str("gain" if self.criterion == entropy else "gini")
                        + " = "
                        + str(np.round(node.children_left.impurity, 2))
                        + r"\n"
                    )
                    aux1 += "samples = " + str(node.children_left.samples) + r"\n"
                    aux1 += "values = " + str(node.children_left.results)
                    g.edge(aux, aux1)
                self.__aux_print_tree(
                    node.children_left, g, features_names, target_names
                )
            if node.children_right:
                if node.children_right.is_leaf:
                    leaf_right = ""
                    leaf_right += (
                        "samples = " + str(node.children_right.samples) + r"\n"
                    )
                    leaf_right += "values = " + str(node.children_right.results) + r"\n"
                    leaf_right += "class = " + str(
                        target_names[np.argmax(node.children_right.results)]
                    )
                    g.edge(aux, leaf_right)

                else:
                    aux2 = ""
                    for i in range(len(node.children_right.weights)):
                        if i == len(node.children_right.weights) - 1:
                            aux2 += str(abs(node.children_right.weights[i]))
                            aux2 += " > 0"
                        else:
                            if node.children_right.weights[i] == -1:
                                aux2 += "-"
                            elif node.children_right.weights[i] != 1:
                                aux2 += (
                                    str(abs(round(node.children_right.weights[i], 2)))
                                    if i != 0
                                    else str(round(node.children_right.weights[i], 2))
                                )
                            aux2 += str(features_names[i])
                            aux2 += (
                                " + "
                                if node.children_right.weights[i + 1] > 0
                                else " - "
                            )
                    aux2 += r"\n"
                    aux2 += (
                        str("gain" if self.criterion == entropy else "gini")
                        + " = "
                        + str(np.round(node.children_right.impurity, 2))
                        + r"\n"
                    )
                    aux2 += "samples = " + str(node.children_right.samples) + r"\n"
                    aux2 += "values = " + str(node.children_right.results)
                    g.edge(aux, aux2)
                self.__aux_print_tree(
                    node.children_right, g, features_names, target_names
                )
