import numpy as np

from utils import (
    apply_weights,
    calc_impurity,
    entropy,
    gini,
    make_initial_weights,
)


class Movement:
    """Represent a movement to generete a neighbors in a solution of the
    metaheuristic
    const_increase
    multiple_const_increase
    percentage_increase
    multiple_percentage_increase
    double_half
    mathematical
    swap
    reset
    """

    CONST_INCREASE = 1
    MULTIPLE_CONST_INCREASE = 2
    SWAP = 3
    RESET = 4
    MATHEMATICAL = 5


class Node:
    """Represents a node in a decision tree"""

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


class ODT:
    """A oblique decision tree using a metaheuristic for find a array of weights
    for generate virtual feature using mutiples features of database"""

    def __init__(
        self,
        criterion="gini",
        max_depth=None,
        max_samples=10000,
        min_samples_split=2,
        min_samples_leaf=1,
        max_iterations=300000,
        l=10,
        const_increase=0.7,
        multiple_const_increase=0.25,
        swap=0.1,
        reset=0.1,
        mathematical=1,
        seed=42,
    ):

        self.criterion = criterion
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_samples = max_samples
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_iterations = max_iterations
        self.l = l
        self.const_increase = const_increase
        self.multiple_const_increase = multiple_const_increase
        self.swap = swap
        self.reset = reset
        self.mathematical = mathematical
        self.rng = np.random.default_rng(seed)

    def get_params(self, deep=True):
        """get params of the class
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
            "const_increase": self.const_increase,
            "multiple_const_increase": self.multiple_const_increase,
            "swap": self.swap,
            "mathematical": self.mathematical,
            "reset": self.reset,
        }

    def set_params(self, **parametes):
        """Set params of the class

        Returns:
            ODT: return the class
        """
        for parameter, value in parametes.items():
            setattr(self, parameter, value)

        return self

    def fit(self, X, y):
        """build a oblique decision tree classifier

        Arguments:
            X {numpy.ndarray} -- Original dataset for fit
            y {numpy.ndarray} -- Original classes for fit

        Returns:
            SAODT: return the class
        """
        X, y = self.__check_X_y(X, y)

        self.X = X
        self.y = y
        self.unique_classes = np.unique(y)
        self.n_classes_ = self.unique_classes.shape[0]
        self.n_features_ = X.shape[1]
        self.__calc_percentage_movements()
        self.criterion = entropy if self.criterion == "entropy" else gini
        self.tree_ = self.__make_tree(X, y)
        self.__prune(self.tree_)

        return self

    @staticmethod
    def __check_X_y(X, y):
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

    def __calc_percentage_movements(self):
        """calculates the percentage of each movement"""
        self.percentage_movements = [
            (Movement.CONST_INCREASE, self.const_increase),
            (Movement.MULTIPLE_CONST_INCREASE, self.multiple_const_increase),
            (Movement.SWAP, self.swap),
            (Movement.RESET, self.reset),
            (Movement.MATHEMATICAL, self.mathematical),
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
        """build a random movement using percentages of each movement"""

        x = self.rng.random()

        if x <= self.percentages_transform[0]:
            return self.percentage_movements[0][0]
        elif x <= self.percentages_transform[1]:
            return self.percentage_movements[1][0]
        elif x <= self.percentages_transform[2]:
            return self.percentage_movements[2][0]
        elif x <= self.percentages_transform[3]:
            return self.percentage_movements[3][0]
        elif x <= self.percentages_transform[4]:
            return self.percentage_movements[4][0]

    def __make_movement(self, weights, movement):
        """makes a movement in weights, generating a neighbor

        Arguments:
            weights {numpy.ndarray} -- array of the weights
            movement {Movement} -- The movement to be applied in the weights array

        Returns:
            numpy.ndarray -- Weights with movement applied
        """

        weights_neighbor = np.copy(weights)

        if movement == Movement.CONST_INCREASE:
            column_modified = self.rng.integers(weights_neighbor.shape[0] - 1, size=1)[
                0
            ]

            # if self.rng.random() <= 0.3:
            #    value_increase = (1.01 - -1) * self.rng.random() - 1
            # else:
            #    value_increase = (0.51 - -0.5) * self.rng.random() - 0.5

            # value_increase = (0.51 - -0.5) * self.rng.random() - 0.5

            value_increase = (1.01 - -1) * self.rng.random() - 1

            weights_neighbor[column_modified] += value_increase

        elif movement == Movement.MULTIPLE_CONST_INCREASE:
            # k = 2 if weights_neighbor.shape[0] - 1 <= 4 else 4
            k = weights_neighbor.shape[0] // 2

            number_columns_modified = self.rng.integers(2, k + 1, size=1)[0]

            columns_modified = []

            while len(columns_modified) != number_columns_modified:
                column_modified = self.rng.integers(
                    weights_neighbor.shape[0] - 1, size=1
                )[0]

                if not column_modified in columns_modified:
                    columns_modified.append(column_modified)

            for column_modified in columns_modified:
                # value_increase = (0.51 - -0.5) * self.rng.random() - 0.5

                # if self.rng.random() <= 0.3:
                #    value_increase = (1.01 - -1) * self.rng.random() - 1
                # else:
                #    value_increase = (0.51 - -0.5) * self.rng.random() - 0.5

                value_increase = (1.01 - -1) * self.rng.random() - 1

                weights_neighbor[column_modified] += value_increase

        elif movement == Movement.SWAP:
            column_modified = self.rng.integers(weights_neighbor.shape[0] - 1, size=1)[
                0
            ]

            column_swap = self.rng.integers(weights_neighbor.shape[0] - 1, size=1)[0]

            while column_swap == column_modified:
                column_swap = self.rng.integers(weights_neighbor.shape[0] - 1, size=1)[
                    0
                ]

            (weights_neighbor[column_modified], weights_neighbor[column_swap],) = (
                np.copy(weights_neighbor[column_swap]),
                np.copy(weights_neighbor[column_modified]),
            )

        elif movement == Movement.RESET:
            column_modified = self.rng.integers(weights_neighbor.shape[0] - 1, size=1)[
                0
            ]

            weights_neighbor[column_modified] = 0

        elif movement == Movement.MATHEMATICAL:
            column_modified = self.rng.integers(weights_neighbor.shape[0] - 1, size=1)[
                0
            ]

            # **2, sqrt, ln(>0), exp, *2, /2

            func = self.rng.integers(6, size=1)[0]

            if func == 0:
                weights_neighbor[column_modified] = (
                    weights_neighbor[column_modified] * 2
                )
            elif func == 1:
                weights_neighbor[column_modified] = (
                    weights_neighbor[column_modified] / 2
                )
            elif func == 2:
                weights_neighbor[column_modified] = np.square(
                    weights_neighbor[column_modified]
                )
            elif func == 3:
                weights_neighbor[column_modified] = np.sqrt(
                    weights_neighbor[column_modified]
                )
            elif func == 4:
                weights_neighbor[column_modified] = np.exp(
                    weights_neighbor[column_modified]
                )
            else:
                if weights_neighbor[column_modified] > 0:
                    weights_neighbor[column_modified] = np.log(
                        weights_neighbor[column_modified]
                    )

        return weights_neighbor

    def __lahc(self, X, y, frequencies_y):
        if X.shape[0] > self.max_samples:
            random_indexes = np.random.choice(
                X.shape[0], size=self.max_samples, replace=False,
            )

            X = np.copy(X[random_indexes])
            y = np.copy(y[random_indexes])

        weights = make_initial_weights(X, y, self.criterion, frequencies_y)

        weights_final = np.copy(weights)

        cost = calc_impurity(
            X, y, weights_final, self.criterion, frequencies_y, self.min_samples_leaf,
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
                # if cost_neighbor > cost:
                #    iteration = 0

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
        """criterion if the tree growth should be stopped

        Args:
            n_classes (int): number of classes in the node
            depth (int): max size of the tree

        Returns:
            bool: return if growth should be stopped or not
        """
        return (
            self.max_depth == depth
            or n_classes == 1
            or samples_node <= self.min_samples_split
        )

    def __make_tree(self, X, y, depth=1):
        print(depth)
        if X.shape[0] == 0 or y.shape[0] == 0:
            return Node()

        classes, count_classes = np.unique(y, return_counts=True)

        error = np.sum(np.delete(count_classes, np.argmax(count_classes)))

        frequencies_y = np.zeros(self.n_features_)
        frequencies_y[classes] = count_classes

        n_classes = classes.shape[0]

        if not self.__stopping_criterion(n_classes, depth, X.shape[0]):
            weights, _ = self.__lahc(X, y, frequencies_y)
            print(weights)

            split = np.array([apply_weights(record, weights) > 0 for record in X])

            if (
                np.sum(split) <= self.min_samples_leaf
                or np.sum(~split) <= self.min_samples_leaf
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

    def predict(self, X):
        """predict classes of the data

        Args:
            X (numpy.ndarray): data to be predict

        Returns:
            numpy.ndarray: classes predicted of the data
        """
        return self.__classify(self.tree_, X)

    def __classify(self, node, X):
        """aux function for the predict

        Args:
            node (DecisionNode): current node in prediction
            X (numpy.ndarray): current data to be predict

        Returns:
            numpy.ndarray: current classes predicted of the data
        """
        if node.is_leaf:
            return np.zeros(X.shape[0]) + np.argmax(node.results)

        split = np.array([apply_weights(record, node.weights) > 0 for record in X])

        y_pred = np.zeros(X.shape[0])

        if split.sum() > 0:
            y_pred[split] = self.__classify(node.children_left, X[split])

        if (~split).sum() > 0:
            y_pred[~split] = self.__classify(node.children_right, X[~split])

        return y_pred

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
