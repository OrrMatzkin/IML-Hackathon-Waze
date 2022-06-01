import sys
from dates_coords import convert_dates, convert_coordinates
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px



import numpy as np

from load_data import load_data

if __name__ == '__main__':
    print("GOOD LUCK")
    np.random.seed(0)
    X_train, X_test = load_data(f"waze_data.csv")
    convert_dates(X_train)
    convert_coordinates(X_train)
    fig = px.scatter(X_train, x="x", y="y", color="linqmap_type")
    fig.show(renderer='browser')

