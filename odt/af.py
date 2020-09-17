import math
from enum import Enum

import numpy as np

from utils import best_in_column, check_X_y, add_virtual_feature


class Movement(Enum):
    """Represent a movement to generete a neighbors in a solution of the
    simulated annealing metaheuristic
    """

    FIX_INCREASE = 1
    UNIQUE_INCREASE = 2
    MULTIPLE_INCREASE = 3
    FIX_DECREASE = 4
    UNIQUE_DECREASE = 5
    MULTIPLE_DECREASE = 6
    SWAP = 7
    OPPOSITE = 8
    COMPLEMENT = 9
    MINIMUM = 10
    MAXIMUM = 11
    RESET = 12


class ArtificialFeature:
    """Generate a virtual feature in database utilizing impurity criterions
    how information gain and gini index, to increase information in
    database to be used in algorithms of machine learning
    """

    def __init__(
        self,
        criterion="entropy",
        max_samples=10000,
        initial_temperature=0.3,
        alpha=0.99999,
        k=5,
        fix_increase=0.25,
        unique_increase=0.1,
        multiple_increase=0.05,
        fix_decrease=0.25,
        unique_decrease=0.1,
        multiple_decrease=0.05,
        swap=0.05,
        opposite=0.05,
        complement=0.05,
        minimum=0.01,
        maximum=0.01,
        reset=0.01,
        seed=42,
    ):
        self.criterion = criterion
        self.max_samples = max_samples
        self.initial_temperature = initial_temperature
        self.alpha = alpha
        self.k = k
        self.fix_increase = fix_increase
        self.unique_increase = unique_increase
        self.multiple_increase = multiple_increase
        self.fix_decrease = fix_decrease
        self.unique_decrease = unique_decrease
        self.multiple_decrease = multiple_decrease
        self.swap = swap
        self.opposite = opposite
        self.complement = complement
        self.minimum = minimum
        self.maximum = maximum
        self.reset = reset
        self.rng = np.random.default_rng(seed)

    def build_weights(self, X):
        """build a random array of the weights to represent of the initial
        solution

        Arguments:
            X {numpy.ndarray} -- Dataset whit calculates length of the weights

        Returns:
            numpy.ndarray -- Array of the weights
        """
        n_features = X.shape[1]
        weights = (1.01 - -1.01) * self.rng.random((n_features,)) - 1.01
        while True:
            if np.sum(weights) != 0:
                break
            weights = (1.01 - -1.01) * self.rng.random((n_features,)) - 1.01

        return weights

    def calc_percentage_movements(self):
        """calculates the percentage of each movement"""
        self.percentage_movements = [
            (Movement.FIX_INCREASE, self.fix_increase),
            (Movement.UNIQUE_INCREASE, self.unique_increase),
            (Movement.MULTIPLE_INCREASE, self.multiple_increase),
            (Movement.FIX_DECREASE, self.fix_decrease),
            (Movement.UNIQUE_DECREASE, self.unique_decrease),
            (Movement.MULTIPLE_DECREASE, self.multiple_decrease),
            (Movement.SWAP, self.swap),
            (Movement.OPPOSITE, self.opposite),
            (Movement.COMPLEMENT, self.complement),
            (Movement.MINIMUM, self.minimum),
            (Movement.MAXIMUM, self.maximum),
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

    def build_movement(self):
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
        elif x <= self.percentages_transform[5]:
            return self.percentage_movements[5][0]
        elif x <= self.percentages_transform[6]:
            return self.percentage_movements[6][0]
        elif x <= self.percentages_transform[7]:
            return self.percentage_movements[7][0]
        elif x <= self.percentages_transform[8]:
            return self.percentage_movements[8][0]
        elif x <= self.percentages_transform[9]:
            return self.percentage_movements[9][0]
        elif x <= self.percentages_transform[10]:
            return self.percentage_movements[10][0]
        else:
            return self.percentage_movements[11][0]

    def make_movement(self, weights, movement):
        """makes a movement in weights, generating a neighbor

        Arguments:
            weights {numpy.ndarray} -- array of the weights
            movement {Movement} -- The movement to be applied in the weights array

        Returns:
            numpy.ndarray -- Weights with movement applied
        """
        weights = np.copy(weights)

        if movement == Movement.FIX_INCREASE:
            col_modified = self.rng.integers(weights.shape[0], size=1)[0]
            percentage_increase = self.rng.integers(1, 11, size=1)[0] / 100
            weights[col_modified] += max(
                (percentage_increase * abs(weights[col_modified])), 0.1
            )

            if weights[col_modified] > 1:
                weights[col_modified] = 1

        elif movement == Movement.UNIQUE_INCREASE:
            col_modified = self.rng.integers(weights.shape[0], size=1)[0]
            percentage_increase = self.rng.integers(11, 51, size=1)[0] / 100
            weights[col_modified] += max(
                (percentage_increase * abs(weights[col_modified])), 0.1
            )

            if weights[col_modified] > 1:
                weights[col_modified] = 1

        elif movement == Movement.MULTIPLE_INCREASE:
            k = 2 if weights.shape[0] < self.k else self.k
            number_cols_modified = self.rng.integers(2, k + 1, size=1)[0]
            cols_modified = []

            while len(cols_modified) != number_cols_modified:
                col_modified = self.rng.integers(weights.shape[0], size=1)[0]
                if not col_modified in cols_modified:
                    cols_modified.append(col_modified)

            percentage_increase = self.rng.integers(1, 51, size=1)[0] / 100

            for col_modified in cols_modified:
                weights[col_modified] += max(
                    (percentage_increase * abs(weights[col_modified])), 0.1
                )

                if weights[col_modified] > 1:
                    weights[col_modified] = 1

        elif movement == Movement.FIX_DECREASE:
            col_modified = self.rng.integers(weights.shape[0], size=1)[0]
            percentage_decrease = self.rng.integers(1, 11, size=1)[0] / 100
            weights[col_modified] -= max(
                (percentage_decrease * abs(weights[col_modified])), 0.1
            )

            if weights[col_modified] < -1:
                weights[col_modified] = -1

        elif movement == Movement.UNIQUE_DECREASE:
            col_modified = self.rng.integers(weights.shape[0], size=1)[0]
            percentage_decrease = self.rng.integers(11, 51, size=1)[0] / 100
            weights[col_modified] -= max(
                (percentage_decrease * abs(weights[col_modified])), 0.1
            )

            if weights[col_modified] < -1:
                weights[col_modified] = -1

        elif movement == Movement.MULTIPLE_DECREASE:
            k = 2 if weights.shape[0] < self.k else self.k
            number_cols_modified = self.rng.integers(2, k + 1, size=1)[0]
            cols_modified = []

            while len(cols_modified) != number_cols_modified:
                col_modified = self.rng.integers(weights.shape[0], size=1)[0]
                if not col_modified in cols_modified:
                    cols_modified.append(col_modified)

            percentage_decrease = self.rng.integers(1, 51, size=1)[0] / 100

            for col_modified in cols_modified:
                weights[col_modified] -= max(
                    (percentage_decrease * abs(weights[col_modified])), 0.1
                )
                if weights[col_modified] < -1:
                    weights[col_modified] = -1

        elif movement == Movement.SWAP:
            col_modified = self.rng.integers(weights.shape[0], size=1)[0]

            col_swap = self.rng.integers(weights.shape[0], size=1)[0]

            while col_swap == col_modified:
                col_swap = self.rng.integers(weights.shape[0], size=1)[0]

            weights[col_modified], weights[col_swap] = (
                np.copy(weights[col_swap]),
                np.copy(weights[col_modified]),
            )

        elif movement == Movement.OPPOSITE:
            col_modified = self.rng.integers(weights.shape[0], size=1)[0]
            weights[col_modified] = -1 * weights[col_modified]

        elif movement == Movement.COMPLEMENT:
            col_modified = self.rng.integers(weights.shape[0], size=1)[0]
            complement = 1 - abs(weights[col_modified])

            if weights[col_modified] < 0:
                complement *= -1

            weights[col_modified] = complement

        elif movement == Movement.MINIMUM:
            col_modified = self.rng.integers(weights.shape[0], size=1)[0]
            weights[col_modified] = -1

        elif movement == Movement.MAXIMUM:
            col_modified = self.rng.integers(weights.shape[0], size=1)[0]
            weights[col_modified] = 1

        elif movement == Movement.RESET:
            col_modified = self.rng.integers(weights.shape[0], size=1)[0]
            weights[col_modified] = 0

        return weights

    def fit(self, X, y):
        """loads the datas

        Arguments:
            X {numpy.ndarray} -- Original dataset for fit
            y {numpy.ndarray} -- Original classes for fit

        Returns:
            SimulatedAnnealingObliqueDecisionTree -- return the class
        """
        X, y = check_X_y(X, y)

        self.X = X
        self.y = y
        self.n_features_ = X.shape[1]
        self.calc_percentage_movements()

        return self

    def sa(self, unique_classes):
        """simulated annealing metaheuristic for find best weights
        for generate virtual feature in database utilizing the
        impurity criterions

        Returns:
            Tuple -- A tuple containing final X with virtual feature and the
            best solution weights
        """

        n_features = self.n_features_
        criterion = self.criterion
        X = self.X
        y = self.y
        t = self.initial_temperature

        X_with_virtual_feature = np.hstack((X, np.zeros((X.shape[0], 1))))

        if X_with_virtual_feature.shape[0] > self.max_samples:
            random_indexes = np.random.choice(
                X_with_virtual_feature.shape[0],
                size=self.max_samples,
                replace=False,
            )
            X_with_virtual_feature = X_with_virtual_feature[random_indexes]
            y = y[random_indexes]

        weights = self.build_weights(X)
        weights_final = np.copy(weights)

        add_virtual_feature(X_with_virtual_feature, weights_final)

        _, best_cost = best_in_column(
            X_with_virtual_feature,
            y,
            n_features,
            criterion,
            unique_classes,
        )
        current_cost = best_cost

        while t > 0.001:
            weights_neighbors = self.make_movement(weights, self.build_movement())

            while True:
                if np.sum(weights_neighbors) != 0:
                    break
                weights_neighbors = self.make_movement(weights, self.build_movement())

            add_virtual_feature(X_with_virtual_feature, weights_neighbors)

            _, neighbors_cost = best_in_column(
                X_with_virtual_feature,
                y,
                n_features,
                criterion,
                unique_classes,
            )

            delta = neighbors_cost - current_cost

            if delta > 0:
                weights = np.copy(weights_neighbors)
                current_cost = neighbors_cost
                if current_cost > best_cost:
                    weights_final = np.copy(weights)
                    best_cost = current_cost
            elif delta == 0:
                zeros_weights, zeros_weights_neighbors = 0, 0

                for i in range(len(weights)):
                    if weights[i] == 0:
                        zeros_weights += 1

                    if weights_neighbors[i] == 0:
                        zeros_weights_neighbors += 1

                if zeros_weights_neighbors > zeros_weights:
                    weights = np.copy(weights_neighbors)
                    current_cost = neighbors_cost
            else:
                x = self.rng.random()
                if x < math.e ** (delta / t):
                    weights = np.copy(weights_neighbors)
                    current_cost = neighbors_cost

            t *= self.alpha

        add_virtual_feature(X_with_virtual_feature, weights)

        return X_with_virtual_feature, weights_final, best_cost
