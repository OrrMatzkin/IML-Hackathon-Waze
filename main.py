import sys
import numpy as np
import pandas as pd
import plotly.express as px

from load_data import load_data
from preprocess_data import preprocess_data


def diluted_features(df: pd.DataFrame, diluted_proportion: float = .9) -> list:
    features = []
    n_samples = df.shape[0]
    for feature in df:
        num_empty_cell = df[feature].isnull().sum()
        if num_empty_cell / n_samples >= diluted_proportion:
            features.append(feature)
    return features


def preprocess(df: pd.DataFrame) -> (pd.DataFrame, list):
    # Remove duplicates by id and then removes the id  feature
    df.drop_duplicates(subset=['OBJECTID'], inplace=True)

    # Removes 'OBJECTID' 'nComments' and diluted (above 90% empty) features
    remove_features = diluted_features(df)
    remove_features.append(['OBJECTID', 'nComments'])
    df.drop(remove_features, axis=1, inplace=True)
    
    # most major accidents happened outside of city, so if 'linqmap_subtype' is null and is outside of the city, put 'ACCIDENT_MAJOR', and 'ACCIDENT_MINOR' otherwise
    accident_type = df[df["linqmap_type"] == "ACCIDENT"]
    accident_type['linqmap_subtype'].mask(accident_type['linqmap_subtype'].isna() & accident_type["linqmap_city"].isna(), 'ACCIDENT_MAJOR', inplace=True)
    accident_type['linqmap_subtype'].mask(accident_type['linqmap_subtype'].isna() & ~accident_type["linqmap_city"].isna(), 'ACCIDENT_MINOR', inplace=True)

    ROAD_CLOSED_type = df[df["linqmap_type"] == "ROAD_CLOSED"]
    fig = px.scatter(ROAD_CLOSED_type, x="linqmap_street", y="linqmap_subtype")
    fig.show()

    return df, remove_features


if __name__ == '__main__':
    preprocess(pd.read_csv("waze_data.csv"))
    np.random.seed(0)
    X_train, X_test = load_data(f"waze_data.csv")
    preprocess_data(X_train)
