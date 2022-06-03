import copy
import numpy as np
import pandas as pd
import csv

from sklearn.preprocessing import PolynomialFeatures

import event_distribution_prediction
from preprocess import process_accident, process_road_closed, process_jam, \
    process_weatherhazard, process_city_street, remove_diluted_features

FEATURES_TO_DUMMIES_T2 = ['linqmap_city', 'linqmap_roadType']
FEATURES_TO_DROP = ['linqmap_reportDescription', 'linqmap_nearby',
                    'update_date', 'linqmap_street',
                    'linqmap_expectedBeginDate', 'linqmap_expectedEndDate',
                    'OBJECTID', 'nComments',
                    'linqmap_reportMood', 'linqmap_magvar', 'pub_date',
                    'pub_time']

types = {'ACCIDENT': 0, 'JAM': 1, 'ROAD_CLOSED': 2, "WEATHERHAZARD": 3}


def convert_dates_task2(df: pd.DataFrame) -> None:
    dts = pd.to_datetime(df['pubDate']).dt.tz_localize('UTC').dt.tz_convert(
        'Israel')
    df['pub_date'] = [dt.date() for dt in dts]
    df['pub_time'] = [dt.time() for dt in dts]
    df['hour_in_day'] = dts.dt.hour  # hour as int from 0 to 24
    df['day_of_week'] = dts.dt.dayofweek

    dts = pd.to_datetime(df['update_date']).dt.tz_localize(
        'UTC').dt.tz_convert(
        'Israel')
    df['update_date_new'] = [dt.date() for dt in dts]


def remove_diluted_features_task2(df: pd.DataFrame,
                                  diluted_proportion: float = .9) -> list:
    df.drop_duplicates(subset=['OBJECTID'], inplace=True)
    features = []
    n_samples = df.shape[0]
    for feature in df:
        num_empty_cell = df[feature].isnull().sum()
        if num_empty_cell / n_samples >= diluted_proportion:
            features.append(feature)
    features += ['OBJECTID', 'nComments']
    df.drop(FEATURES_TO_DROP, axis=1, inplace=True)
    return features


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
    remove_diluted_features_task2(data)
    hours = data['hour_in_day']
    days = data['day_of_week']
    data = data.drop(['hour_in_day', 'day_of_week'], axis=1)
    data = make_dummies_task2(data)
    data['hour_in_day'], data['day_of_week'] = hours, days
    return data


def run_task_2(train_data):
    days = np.zeros((3, 3, 4), dtype=int)
    data = preprocess_task2(train_data, False)
    for j, d in enumerate([6, 1, 3]):
        day_data = data.loc[data["day_of_week"] == d]
        dts = pd.to_datetime(day_data["pubDate"]).dt.tz_localize(
            'UTC').dt.tz_convert('Israel')
        total = len(np.unique([dt.date() for dt in dts]))
        group = day_data.groupby(
            ["hour_in_day", "linqmap_type"]).size().reset_index(name='count')
        for i, t in enumerate(
                ["ACCIDENT", "JAM", "ROAD_CLOSED", "WEATHERHAZARD"]):
            y = group.loc[group["linqmap_type"] == t]
            y["count"] = y["count"] / total
            y_ = np.zeros(24)
            y_[y["hour_in_day"]] = y["count"]
            days[j][0][i] = np.mean(y_[[9, 10, 11]])  # 10 - 12 mean of type
            days[j][1][i] = np.mean(y_[[13, 14, 15]])  # 14 - 16 mean of type
            days[j][2][i] = np.mean(y_[[17, 18, 19]])  # 18 - 20 mean of type
    print(days)
    return days


if __name__ == '__main__':
    days = run_task_2(pd.read_csv("waze_data.csv"))
    with open("task2_predictions.csv", "w+") as my_csv:
        for i in range(len(days)):
            df = pd.DataFrame.from_records(days[i], columns=["ACCIDENT", "JAM",
                                                             "ROAD_CLOSED",
                                                             "WEATHERHAZARD"])
            df.to_csv(my_csv, index=False)
