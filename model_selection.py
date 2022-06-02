import typing
from datetime import datetime
import pandas as pd
import numpy as np
import re
from geopy.distance import distance
from pyproj import Transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from preprocess import *

def model_selection(train_data):

    train_data = train_data.replace(np.nan, 0, regex=True)
    train_data = train_data.drop(["pubDate"], axis=1)
    train_data = train_data.drop(["update_date"], axis=1)
    train_data = train_data.drop(["pubDate_day"], axis=1)
    train_data = train_data.drop(["update_time"], axis=1)

    train_data, valid_data = train_test_split(train_data, test_size=0.8, shuffle=True)

    train_X, train_y = compress_4_rows_into_one(train_data)
    valid_X, valid_y = compress_4_rows_into_one(valid_data)
    try_simply_evaluation(train_X, train_y, valid_X, valid_y)
    major_vote(train_data)

def try_simply_evaluation(train_X, train_y, test_X, test_y):
    model = RandomForestClassifier().fit(train_X, train_y)
    y_pred = model.predict(train_X)
    score = f1_score(train_y, y_pred, average='macro')
    print(f"value in KNN on the train set: {score}")
    y_pred = model.predict(test_X)
    score = f1_score(test_y, y_pred, average='macro')
    print(f"value in KNN on the test set: {score}")

def major_vote(df):
    list_of_4_rows = [[df.iloc[i+j]['linqmap_subtype'] for j in range(4)] for i in range(1000)]
    y_pred = [max(set(cur_list), key=cur_list.count) for cur_list in list_of_4_rows]
    y = df.loc[:, ["linqmap_subtype"]]
    y = y[4:1004]
    score = f1_score(y, y_pred, average='macro')
    print(f"value of major vote: {score}")