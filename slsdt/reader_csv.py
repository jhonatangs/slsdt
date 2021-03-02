from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def read_csv(file: str, target: str, return_names: bool = False):
    """
    Reads a database in csv file format and does the basic pre-processing
    required for the SLSDT.

    Args:
        file (str): csv file path
        target (str): name of the column of the classification target in csv file.

    Returns:
        tuple: A tuple containing:\n
        ndarray: The data array,\n
        ndarray: The classification target,\n
        If return_names is True:\n
        ndarray: Features names,\n
        ndarray: Unique class labels names,

    """
    df = pd.read_csv(file)

    features_names = df.drop([target], axis=1, inplace=False).columns.values
    target_names = np.unique(df[target].values)

    for column in df.columns:
        if df[column].dtype == object:
            encoder = LabelEncoder()
            df[column] = encoder.fit_transform(df[column])

    data = np.ascontiguousarray(df.drop([target], axis=1, inplace=False).values)

    encoder = LabelEncoder()
    classes = np.ascontiguousarray(encoder.fit_transform(df[target]))

    if return_names:
        return (
            np.array(data, np.float64),
            np.array(classes, np.int64),
            features_names,
            target_names,
        )
    else:
        return np.array(data, np.float64), np.array(classes, np.int64)
