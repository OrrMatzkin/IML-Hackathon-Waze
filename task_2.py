from preprocess import preprocess_task2
import numpy as np
import pandas as pd


def run_task_2(train_data) -> np.ndarray:
    """
    Gives predictions for task No. 2.
    :param train_data: traind data
    :return: days predictions.
    """
    print("Starts task 2...\n")
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
    print("Task 2 output:")
    print(days)
    print("\nDone task 2!\n")
    return days

