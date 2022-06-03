import copy

import pandas as pd

import event_distribution_prediction
from preprocess import process_accident, process_road_closed, process_jam, \
    process_weatherhazard, process_city_street, remove_diluted_features

FEATURES_TO_DUMMIES_T2 = ['linqmap_city', 'linqmap_street', 'linqmap_roadType',
                          'linqmap_roadType']

def convert_dates_task2(df: pd.DataFrame) -> None:
    dts = pd.to_datetime(df['pubDate']).dt.tz_localize('UTC').dt.tz_convert('Israel')
    df['update_date_new'] = [dt.date() for dt in dts]
    df['update_time'] = [dt.time() for dt in dts]
    df['hour_in_day'] = dts.dt.hour  # hour as int from 0 to 24
    df['day_of_week'] = dts.dt.dayofweek


def make_dummies_task2(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(data=df, columns=FEATURES_TO_DUMMIES_T2)


def preprocess_task2(df: pd.DataFrame, geo: bool):
    data = copy.deepcopy(df)
    convert_dates_task2(data)
    process_accident(data)
    process_road_closed(data)
    process_jam(data)
    process_weatherhazard(data)
    process_city_street(data, geo)
    pubDates = data['pubDate']
    remove_diluted_features(data)
    data["pubDate"] = pubDates
    hours = data['hour_in_day']
    days = data['day_of_week']
    data = data.drop(['update_time', 'hour_in_day', 'day_of_week'], axis=1)
    data = make_dummies_task2(data)
    data['hour_in_day'], data['day_of_week'] = hours, days

    # Prediction
    event_distribution_prediction.graph_jams_by_hour(data)