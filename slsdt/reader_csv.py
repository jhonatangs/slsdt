from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def read_csv(file: str, class_index: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads a database in csv file format performing pre-processing. Returning
    datas and classes separately.

    Args:
        file (str): File path
        class_index (str): Name index of class in csv file

    Returns:
        (numpy.ndarray, numpy.ndarray): Returns a tuple containg a
        two-dimensional array representing datas and a one-dimensional array
        representing classes
    """
    df = pd.read_csv(file)

    for column in df.columns:
        if df[column].dtype == object:
            encoder = LabelEncoder()
            df[column] = encoder.fit_transform(df[column])

    data = np.ascontiguousarray(df.drop([class_index], axis=1, inplace=False).values)

    encoder = LabelEncoder()
    classes = np.ascontiguousarray(encoder.fit_transform(df[class_index]))

    return np.array(data, np.float64), np.array(classes, np.int64)
