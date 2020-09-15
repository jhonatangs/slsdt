from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer


def read_csv(file: str, class_index: str) -> Tuple[np.ndarray, np.ndarray]:
    """Reads a database in csv file format performing pre-processing. Returning
    datas and classes separately.

    Args:
        file (str): File path
        class_index (str): Name index of class in csv file

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: Returns a tuple containg a 
        two-dimensional array representing datas and a one-dimensional array
        representing classes
    """
    df = pd.read_csv(file)

    data = np.ascontiguousarray(df.drop([class_index], axis=1, inplace=False).values)
    classes = np.ascontiguousarray(df[class_index].values)

    data_num_columns = data.shape[1]
    data_indexs_str = [
        i for i in range(data_num_columns) if isinstance(data[0, i], str)
    ]
    column_transformer_data = ColumnTransformer(
        [("encoder", OneHotEncoder(), data_indexs_str)], remainder="passthrough"
    )
    try:
        data = column_transformer_data.fit_transform(data).toarray()
    except AttributeError:
        data = column_transformer_data.fit_transform(data)

    label_encoder_classes = LabelEncoder()
    classes = label_encoder_classes.fit_transform(classes)

    return np.array(data, np.float64), np.array(classes, np.int)
