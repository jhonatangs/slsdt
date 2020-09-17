import numpy as np

from af import ArtificialFeature
from utils import check_X_y, add_virtual_feature, del_last_column, find_cut_point


class DecisionNode:
    """Represents a node in a decision tree"""

    def __init__(
        self,
        column=-1,
        threshold=None,
        weights=np.array([]),
        is_leaf=None,
        results=np.array([]),
        children_left=None,
        children_right=None,
    ):

        self.column = column
        self.threshold = threshold
        self.weights = weights
        self.is_leaf = is_leaf
        self.results = results
        self.children_left = children_left
        self.children_right = children_right


class SAODT:
    """A oblique decision tree using simulated annealing metaheuristic for
    find a array of weights for generate virtual feature using mutiples
    features of database
    """

    def __init__(
        self,
        criterion="entropy",
        max_depth=None,
        max_samples=10000,
        initial_temperature=0.4,
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
        self.max_depth = max_depth
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
        self.seed = seed

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
            "initial_temperature": self.initial_temperature,
            "alpha": self.alpha,
            "k": self.k,
            "fix_increase": self.fix_increase,
            "unique_increase": self.unique_increase,
            "multiple_increase": self.multiple_increase,
            "fix_decrease": self.fix_decrease,
            "unique_decrease": self.unique_decrease,
            "multiple_decrease": self.multiple_decrease,
            "swap": self.swap,
            "opposite": self.opposite,
            "complement": self.complement,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "reset": self.reset,
        }

    def set_params(self, **parametes):
        """Set params of the class

        Returns:
            SAODT: return the class
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
        X, y = check_X_y(X, y)

        self.X = X
        self.y = y
        self.unique_classes = np.unique(y)
        self.n_classes_ = self.unique_classes.shape[0]
        self.n_features_ = X.shape[1]
        self.tree_ = self.make_tree(X, y)

        return self

    def stopping_criterion(self, n_classes, depth):
        """criterion if the tree growth should be stopped

        Args:
            n_classes (int): number of classes in the node
            depth (int): max size of the tree

        Returns:
            bool: return if growth should be stopped or not
        """
        return self.max_depth == depth or n_classes == 1

    def make_tree(self, X, y, depth=1):
        """build the tree

        Args:
            X (numpy.ndarray): data in the node
            y (numpy.ndarray): classes in the node
            depth (int, optional): current tree size. Defaults to 1.

        Returns:
            DecisionNode: return recursively the nodes of the tree
        """
        if len(X) == 0 or len(y) == 0:
            return DecisionNode()
        classes, count_classes = np.unique(y, return_counts=True)
        n_classes = classes.shape[0]
        if not self.stopping_criterion(n_classes, depth):
            af = ArtificialFeature(
                criterion=self.criterion,
                max_samples=self.max_samples,
                initial_temperature=self.initial_temperature,
                alpha=self.alpha,
                k=self.k,
                fix_increase=self.fix_increase,
                unique_increase=self.unique_increase,
                multiple_increase=self.multiple_increase,
                fix_decrease=self.fix_decrease,
                unique_decrease=self.unique_decrease,
                multiple_decrease=self.multiple_decrease,
                swap=self.swap,
                opposite=self.opposite,
                complement=self.complement,
                minimum=self.minimum,
                maximum=self.maximum,
                reset=self.reset,
                seed=self.seed,
            )
            af.fit(X, y)
            X_with_virtual_feature, weights, _ = af.sa(self.unique_classes)
            index, threshold = find_cut_point(
                X_with_virtual_feature, y, self.criterion, self.unique_classes
            )
            div = X_with_virtual_feature[:, index] <= threshold
            X_left = del_last_column(X_with_virtual_feature[div])
            X_right = del_last_column(X_with_virtual_feature[~div])
            left = self.make_tree(X_left, y[div], depth + 1)
            right = self.make_tree(X_right, y[~div], depth + 1)

            return DecisionNode(
                column=index,
                threshold=threshold,
                weights=weights,
                children_left=left,
                children_right=right,
            )

        values = np.zeros(self.n_classes_)
        values[classes] = count_classes
        return DecisionNode(is_leaf=True, results=values)

    def predict(self, X):
        """predict classes of the data

        Args:
            X (numpy.ndarray): data to be predict

        Returns:
            numpy.ndarray: classes predicted of the data
        """
        return self.classify(self.tree_, X)

    def classify(self, node, X):
        """aux function for the predict

        Args:
            node (DecisionNode): current node in prediction
            X (numpy.ndarray): current data to be predict

        Returns:
            numpy.ndarray: current classes predicted of the data
        """
        if node.is_leaf:
            return np.zeros(X.shape[0]) + np.argmax(node.results)

        X = np.hstack((X, np.zeros((X.shape[0], 1))))
        add_virtual_feature(X, node.weights)
        div = X[:, node.column] <= node.threshold
        y_pred = np.zeros(X.shape[0])

        X_left = del_last_column(X[div])
        X_right = del_last_column(X[~div])

        if div.sum() > 0:
            y_pred[div] = self.classify(node.children_left, X_left)

        if (~div).sum() > 0:
            y_pred[~div] = self.classify(node.children_right, X_right)

        return y_pred