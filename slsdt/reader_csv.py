from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def read_csv(file: str, target: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads a database in csv file format and does the basic pre-processing
    required for the SLSDT.

    Args:
        file (str): csv file path
        class_index (str): name of the column of the classification target
        in csv file

    Returns:
        Tuple object, containing:

        data (ndarray): The data array.
        Target (ndarray): The classification target.

    """
    df = pd.read_csv(file)

    for column in df.columns:
        if df[column].dtype == object:
            encoder = LabelEncoder()
            df[column] = encoder.fit_transform(df[column])

    data = np.ascontiguousarray(df.drop([target], axis=1, inplace=False).values)

    encoder = LabelEncoder()
    classes = np.ascontiguousarray(encoder.fit_transform(df[target]))

    return np.array(data, np.float64), np.array(classes, np.int64)
