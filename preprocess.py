from datetime import datetime
import pandas as pd
import numpy as np
from pyproj import CRS
from pyproj import Transformer
import plotly.express as px

EMPTY = ['linqmap_reportDescription', 'linqmap_nearby', 'linqmap_expectedBeginDate', 'linqmap_expectedEndDate', 'OBJECTID', 'nComments', 'linqmap_reportMood']
def convert_dates(data: pd.DataFrame) -> None:
    dates = data['update_date']
    data['update_date'] = pd.Series([datetime.fromtimestamp(time / 1000) for time in dates])


def convert_coordinates(data) -> None:
    X = np.array(data['x'])
    Y = np.array(data['y'])

    crs = CRS.from_epsg(6991)
    crs.to_epsg()
    crs = CRS.from_proj4(
        "+proj=tmerc +lat_0=31.7343936111111 +lon_0=35.2045169444445 "
        "+k=1.0000067 +x_0=219529.584 +y_0=626907.39 +ellps=GRS80 "
        "+towgs84=-24.002400,-17.103200,-17.844400,-0.33007,-1.852690,"
        "1.669690,5.424800 +units=m +no_defs")
    transformer = Transformer.from_crs("EPSG:6991", "EPSG:4326")
    wgs84_coords = [transformer.transform(X[i], Y[i]) for i in range(len(X))]
    data['y'] = pd.Series([tup[0] for tup in wgs84_coords])
    data['x'] = pd.Series([tup[1] for tup in wgs84_coords])


def categorize_linqmap_city(df: pd.DataFrame):
    pd.get_dummies(df, columns=['linqmap_city'])



def process_pubDate(df: pd.DataFrame):
    # "15/5/2022" datetime
    df['pubDate_day'] = [datetime.strptime(date[0:10], "%m/%d/%Y").date() for date in df['pubDate']]
    df['pubDate'] = pd.to_datetime(df['pubDate'])  # full date "15/5/2022 20:30:55" as datetime
    df['pubDate_hour'] = df['pubDate'].dt.hour  # hour as int from 0 to 24
    return df


def remove_diluted_features(df: pd.DataFrame, diluted_proportion: float = .9) -> list:
    df.drop_duplicates(subset=['OBJECTID'], inplace=True)
    features = []
    n_samples = df.shape[0]
    for feature in df:
        num_empty_cell = df[feature].isnull().sum()
        if num_empty_cell / n_samples >= diluted_proportion:
            features.append(feature)
    features += ['OBJECTID', 'nComments']
    df.drop(EMPTY, axis=1, inplace=True)
    return features


def add_accident_type(df: pd.DataFrame):
    # most major accidents happened outside of city, so if 'linqmap_subtype' is null and is outside of the city, put 'ACCIDENT_MAJOR', and 'ACCIDENT_MINOR' otherwise
    accident_type = df[df["linqmap_type"] == "ACCIDENT"]
    accident_type['linqmap_subtype'].mask(
        accident_type['linqmap_subtype'].isna() & accident_type["linqmap_city"].isna(), 'ACCIDENT_MAJOR', inplace=True)
    accident_type['linqmap_subtype'].mask(
        accident_type['linqmap_subtype'].isna() & ~accident_type["linqmap_city"].isna(), 'ACCIDENT_MINOR', inplace=True)

    # ROAD_CLOSED_type = df[df["linqmap_type"] == "ROAD_CLOSED"]
    # fig = px.scatter(ROAD_CLOSED_type, x="linqmap_street", y="linqmap_subtype")
    # fig.show()


def preprocess(df: pd.DataFrame) -> None:
    add_accident_type(df)
    convert_dates(df)
    convert_coordinates(df)
    categorize_linqmap_city(df)
    remove_diluted_features(df)


