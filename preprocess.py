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
    df['pubDate'] = pd.to_datetime(df['pubDate'])
    df['pubDate'] = pd.to_datetime(df['pubDate'])  # full date "15/5/2022 20:30:55" as datetime
    df['date'] = pd.to_datetime(df['pubDate']).dt.date
    df['time'] = pd.to_datetime(df['pubDate']).dt.time
    df['pubDate_hour'] = df['pubDate'].dt.hour  # hour as int from 0 to 24
    df['pubDate_day_of_week'] = df['pubDate'].dt.dayofweek  # hour as int from 0 to 24


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



def proccess_accident(df: pd.DataFrame):
    # most major accidents happened outside of city, so if 'linqmap_subtype' is null and is outside of the city, put 'ACCIDENT_MAJOR', and 'ACCIDENT_MINOR' otherwise
    accident_type = df[df["linqmap_type"] == "ACCIDENT"]

    accident_type['linqmap_subtype'].mask(
        accident_type['linqmap_subtype'].isna() & accident_type["linqmap_city"].isna(), 'ACCIDENT_MAJOR', inplace=True)
    accident_type['linqmap_subtype'].mask(
        accident_type['linqmap_subtype'].isna() & ~accident_type["linqmap_city"].isna(), 'ACCIDENT_MINOR', inplace=True)

    df['linqmap_subtype'].fillna(accident_type['linqmap_subtype'], inplace=True)

def proccess_road_closed(df):
    road_closed_type = df[df["linqmap_type"] == "ROAD_CLOSED"]
    null_road_closed_type = road_closed_type[road_closed_type['linqmap_subtype'].isna()]
    for idx, row in null_road_closed_type.iterrows():
        same_date = road_closed_type[road_closed_type["date"] == row['date']]
        if (not(row['linqmap_street'] is None)) and (same_date['linqmap_street'].str.contains(row['linqmap_street']).any()):
            df.at[idx,'linqmap_subtype'] = 'ROAD_CLOSED_CONSTRUCTION'
            continue
        row['linqmap_subtype'] = 'ROAD_CLOSED_EVENT'
    # print("hi")

def proccess_jam(df):
    jam_type = df[df["linqmap_type"] == "JAM"]
    null_jam_type = jam_type[jam_type['linqmap_subtype'].isna()]
    for idx, row in null_jam_type.iterrows():
        same_date_and_place = jam_type[(jam_type["date"] == row['date']) &
                                       (jam_type["linqmap_street"] == row['linqmap_street']) &
                                       (~(jam_type["linqmap_subtype"].isna()))]
        if same_date_and_place.empty:
            df.at[idx, 'linqmap_subtype'] = 'JAM_HEAVY_TRAFFIC'
            continue
        delta_time = same_date_and_place['pubDate'].apply(lambda x: np.abs((x - row['pubDate']).total_seconds()))
        closest = same_date_and_place.loc[delta_time.idxmin()]
        df.at[idx, 'linqmap_subtype'] = closest['linqmap_subtype']
        # print("h")

def proccess_weatherhazard(df):
    weatherhazard_type = df[df["linqmap_type"] == "WEATHERHAZARD"]
    dist = weatherhazard_type.linqmap_subtype.value_counts(normalize=True)
    missing = weatherhazard_type['linqmap_subtype'].isnull()
    weatherhazard_type.loc[missing, 'linqmap_subtype'] = np.random.choice(dist.index,
                                                 size=len(weatherhazard_type[missing]),
                                                 p=dist.values)
    df['linqmap_subtype'].fillna(weatherhazard_type['linqmap_subtype'], inplace=True)
    print('h')





def preprocess(df: pd.DataFrame) -> None:
    process_pubDate(df)
    convert_coordinates(df)
    convert_dates(df)
    proccess_accident(df)
    proccess_road_closed(df)
    proccess_jam(df)
    proccess_weatherhazard(df)
    categorize_linqmap_city(df)
    remove_diluted_features(df)


