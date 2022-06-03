import copy
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

import event_distribution_prediction
from preprocess import process_accident, process_road_closed, process_jam, \
    process_weatherhazard, process_city_street, remove_diluted_features

FEATURES_TO_DUMMIES_T2 = ['linqmap_city', 'linqmap_roadType']

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
    data = data.drop(['hour_in_day', 'day_of_week'], axis=1)
    data = make_dummies_task2(data)
    data['hour_in_day'], data['day_of_week'] = hours, days
    return data


def run_task_2(train_data):
    y_ = None
    x = [i for i in range(24)]
    data = preprocess_task2(train_data, False)
    for d in range(7):
        day_data = data.loc[data["day_of_week"] == d]
        dts = pd.to_datetime(day_data["pubDate"]).dt.tz_localize(
                    'UTC').dt.tz_convert('Israel')
        # dts = day_data["pub"].apply(pd.to_datetime)
        # dts.dt.tz_localize('UTC').dt.tz_convert('Israel')
        total = len(np.unique([dt.date() for dt in dts]))
        group = day_data.groupby(["hour_in_day", "linqmap_type"]).size().reset_index(name='count')
        for t in ["ACCIDENT", "JAM", "ROAD_CLOSED", "WEATHERHAZARD"]:
            y = group.loc[group["linqmap_type"] == t]
            y["count"] = y["count"] / total
            y_ = np.zeros(24)
            y_[y["hour_in_day"]] = y["count"]


if __name__ == '__main__':
    run_task_2(pd.read_csv("waze_data.csv"))