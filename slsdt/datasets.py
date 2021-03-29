from slsdt.reader_csv import read_csv


def load_dataset(name: str, return_names: bool = False):
    """Loads a public dataset from UCI repository for testing.
    Availables: iris

    Args:
        name (str): name of the dataset for load
        return_names (bool, optional): if want return the names of features and
        classes. Defaults to False.

    Returns:
        tuple: A tuple containing:\n
        ndarray: The data array,\n
        ndarray: The classification target,\n
        If return_names is True:\n
        ndarray: Features names,\n
        ndarray: Unique class labels names,
    """
    if return_names:
        return read_csv(f"data/{name}.csv", "class", True)
    else:
        return read_csv(f"data/{name}.csv", "class")
