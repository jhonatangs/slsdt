import glob

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

from slsdt import SLSDT


import os

print(os.listdir())
print(os.getcwd())
print(glob.glob("./*.py"))
print(glob.glob("./slsdt/*.py"))
