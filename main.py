import numpy as np
import pandas as pd
import sys


def diluted_features(df: pd.DataFrame, diluted_proportion: float = .9) -> list:
    features = []
    n_samples = df.shape[0]
    for feature in df:
        num_empty_cell = df[feature].isnull().sum()
        if num_empty_cell / n_samples >= diluted_proportion:
            features.append(feature)
    return features


def preprocess(data: pd.DataFrame) -> (pd.DataFrame, list):
    # Remove duplicates by id and then removes the id  feature
    data.drop_duplicates(subset=['OBJECTID'], inplace=True)

    # Removes 'OBJECTID' 'nComments' and diluted (above 90% empty) features
    remove_features = diluted_features(data)
    remove_features.append(['OBJECTID', 'nComments'])
    data.drop(remove_features, axis=1, inplace=True)

    return data, remove_features


if __name__ == '__main__':
    args = sys.argv
    waze_data = pd.read_csv("waze_data.csv")
    process_data = preprocess(waze_data)
    print("GOOD LUCK")
