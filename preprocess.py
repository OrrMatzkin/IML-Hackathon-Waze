import typing
from datetime import datetime
import pandas as pd
import numpy as np
from geopy.distance import distance
from pyproj import CRS
from pyproj import Transformer
import plotly.express as px
from geopy.geocoders import Nominatim

EMPTY = ['linqmap_reportDescription', 'linqmap_nearby',
         'linqmap_expectedBeginDate', 'linqmap_expectedEndDate', 'OBJECTID',
         'nComments', 'linqmap_reportMood']
LOCATION_TIMEOUT = 6


def convert_dates(data: pd.DataFrame) -> None:
    dts = pd.to_datetime(data['update_date'], unit='ms')
    data['update_date'] = [dt.date() for dt in dts]
    data['update_time'] = [dt.time() for dt in dts]


def convert_coordinates(data) -> None:
    X = pd.Series.to_numpy(data['x'])
    Y = pd.Series.to_numpy(data['y'])

    # crs = CRS.from_epsg(6991)
    # crs.to_epsg()
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


# def geolocator(coordinates: str) -> str:
#     """Return location coordinates and accurate address of the specified location."""
#     geolocator = Nominatim(user_agent="tutorial", timeout=LOCATION_TIMEOUT)
#     try:
#         location = geolocator.reverse(coordinates)
#         if location is not None:
#             return location.address
#     except GeocoderTimedOut as e:
#         print(str(e))
#     return ""

def get_nearest_location(x: float, y: float, df: pd.DataFrame) -> typing.Tuple[str, str]:
    x, y = float(x), float(y)
    min_dist = float('inf')
    city = ""
    street = ""
    for index, row in df.iterrows():
        x1, y1 = float(row['x']), float(row['y'])
        if x == x1 and y == y1:
            continue
        c1 = (y, x)
        c2 = (y1, x1)
        dist = distance(c1, c2)
        curr_city, curr_street = row['linqmap_city'], row['linqmap_street']
        if dist < min_dist and (curr_city or curr_street) and (
                curr_city is not np.nan or curr_street is not np.nan):
            city, street = curr_city, curr_street
            min_dist = dist
    return city, street


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
