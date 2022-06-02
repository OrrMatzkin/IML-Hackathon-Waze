from datetime import datetime
import pandas as pd
import numpy as np
from pyproj import CRS
from pyproj import Transformer
import plotly.express as px

EMPTY = ['linqmap_reportDescription', 'linqmap_nearby',
         'linqmap_expectedBeginDate', 'linqmap_expectedEndDate', 'OBJECTID',
         'nComments', 'linqmap_reportMood']


def convert_dates(data: pd.DataFrame) -> None:
    # dates = np.array(data['update_date'])
    # data['update_date2'] = pd.Series(
    #     [datetime.fromtimestamp(time / 1000) for time in dates])
    data['update_date2'] = pd.to_datetime(data['update_date'], unit='ms')
    print("test")


def convert_coordinates(data) -> None:
    X = pd.Series.to_numpy(data['x'])
    Y = pd.Series.to_numpy(data['y'])

    crs = CRS.from_epsg(6991)
    crs.to_epsg()
    crs = CRS.from_proj4(
        "+proj=tmerc +lat_0=31.7343936111111 +lon_0=35.2045169444445 "
        "+k=1.0000067 +x_0=219529.584 +y_0=626907.39 +ellps=GRS80 "
        "+towgs84=-24.002400,-17.103200,-17.844400,-0.33007,-1.852690,"
        "1.669690,5.424800 +units=m +no_defs")
    transformer = Transformer.from_crs("EPSG:6991", "EPSG:4326")
    wgs84_coords = [transformer.transform(X[i], Y[i]) for i in range(len(X))]
    data['y'] = [tup[0] for tup in wgs84_coords]
    data['x'] = [tup[1] for tup in wgs84_coords]


def categorize_linqmap_city(df: pd.DataFrame):
    pd.get_dummies(df, columns=['linqmap_city'])


def process_pubDate(df: pd.DataFrame):
    df['pubDate'] = pd.to_datetime(df['pubDate'])
    df['pubDate'] = df['pubDate'].dt.year
    df['pubDate'] = df['pubDate'].dt.month
    df['pubDate'] = df['pubDate'].dt.week
    df['pubDate'] = df['pubDate'].dt.day
    df['pubDate'] = df['pubDate'].dt.hour
    df['pubDate'] = df['pubDate'].dt.minute
    df['pubDate'] = df['pubDate'].dt.dayofweek
    df = df.drop(["pubDate"], axis=1)
    return df


def remove_diluted_features(df: pd.DataFrame,
                            diluted_proportion: float = .9) -> list:
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
        accident_type['linqmap_subtype'].isna() & accident_type[
            "linqmap_city"].isna(), 'ACCIDENT_MAJOR', inplace=True)
    accident_type['linqmap_subtype'].mask(
        accident_type['linqmap_subtype'].isna() & ~accident_type[
            "linqmap_city"].isna(), 'ACCIDENT_MINOR', inplace=True)

    # ROAD_CLOSED_type = df[df["linqmap_type"] == "ROAD_CLOSED"]
    # fig = px.scatter(ROAD_CLOSED_type, x="linqmap_street", y="linqmap_subtype")
    # fig.show()


def preprocess(df: pd.DataFrame) -> None:
    add_accident_type(df)
    categorize_linqmap_city(df)
    remove_diluted_features(df)
    convert_dates(df)
    convert_coordinates(df)
