import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filename: str):
    """
    Load dataset
    Parameters
    ----------
    filename: str
        Path to the data set

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    df = pd.read_csv(filename).drop_duplicates()  # init data
    msk = np.random.binomial(1, .8, df.shape[0]).astype(bool)  # make array of boolean array
    train_data = df[msk]
    test_data = df[~msk]
    return train_data, test_data