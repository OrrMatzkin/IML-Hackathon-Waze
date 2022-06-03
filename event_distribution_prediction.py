from copy import deepcopy

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def graph_jams_by_hour(df):
    data = deepcopy(df)
    data = data.drop(["x", "y"], axis=1)
    hours_range = [i for i in range(24)]
    for d in range(7):
        day_data = data.loc[data["day_of_week"] == d]
        dts = pd.to_datetime(day_data).dt.tz_localize(
            'UTC').dt.tz_convert('Israel')
        total = len(np.unique([dt.date() for dt in dts]))
        # total = len(np.unique(np.array(day_data["pubDate"].dt.date())))
        group = day_data.groupby(["hour_in_day", "linqmap_type"]).size().reset_index(name='count')
        fig = go.Figure()
        for t in ["ACCIDENT", "JAM", "ROAD_CLOSED", "WEATHERHAZARD"]:
            y = group.loc[group["linqmap_type"] == t]
            y["count"] = y["count"] / total
            y_ = np.zeros(24)
            y_[y["hour_in_day"]] = y["count"]
            fig.add_trace(go.Scatter(x=hours_range, y=y_, name=t, mode='markers+lines'))
        fig.show(renderer="browser")