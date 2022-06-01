import sys

import numpy as np
import pandas as pd
import plotly.express as px



def preprocess(df: pd.DataFrame):

    # most major accidents happened outside of city, so if 'linqmap_subtype' is null and is outside of the city, put 'ACCIDENT_MAJOR', and 'ACCIDENT_MINOR' otherwise
    accident_type = df[df["linqmap_type"] == "ACCIDENT"]
    accident_type['linqmap_subtype'].mask(accident_type['linqmap_subtype'].isna() & accident_type["linqmap_city"].isna(), 'ACCIDENT_MAJOR', inplace=True)
    accident_type['linqmap_subtype'].mask(accident_type['linqmap_subtype'].isna() & ~accident_type["linqmap_city"].isna(), 'ACCIDENT_MINOR', inplace=True)

    ROAD_CLOSED_type = df[df["linqmap_type"] == "ROAD_CLOSED"]
    fig = px.scatter(ROAD_CLOSED_type, x="linqmap_street", y="linqmap_subtype")
    fig.show()





from load_data import load_data
from preprocess_data import preprocess_data

if __name__ == '__main__':
    print("GOOD LUCK")
    preprocess(pd.read_csv("waze_data.csv"))
    np.random.seed(0)
    X_train, X_test = load_data(f"waze_data.csv")
    preprocess_data(X_train)
