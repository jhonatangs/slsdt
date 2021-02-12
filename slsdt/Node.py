import numpy as np


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