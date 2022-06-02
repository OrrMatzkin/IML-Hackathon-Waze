import typing
from datetime import datetime
import pandas as pd
import numpy as np
import re
from geopy.distance import distance
from pyproj import Transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import plotly.graph_objects as go

from preprocess import *

def model_selection(train_data, train_y):
    train_X, valid_X, train_y, valid_y = train_test_split(train_data, train_y, test_size=0.2, shuffle=True)
    try_simply_evaluation(train_X, train_y, valid_X, valid_y)
    # major_vote(train_data)
    print(f"found k: {kfold_cv(train_X, train_y, valid_X, valid_y)}")

def try_simply_evaluation(train_X, train_y, test_X, test_y):
    model = KNeighborsClassifier(n_neighbors=9).fit(train_X, train_y)
    y_pred = model.predict(train_X)
    score = f1_score(train_y, y_pred, average='micro')
    print(f"value in KNN on the train set with f1: {score}")
    score = mean_squared_error(train_y, y_pred)
    print(f"value in KNN on the train set with MSE: {score}")
    y_pred = model.predict(test_X)
    score = f1_score(test_y, y_pred, average='micro')
    print(f"value in KNN on the test set: {score}")
    score = mean_squared_error(test_y, y_pred)
    print(f"value in KNN on the train set with MSE: {score}")

# def major_vote(df):
#     list_of_4_rows = [[df.iloc[i+j]['linqmap_subtype'] for j in range(4)] for i in range(1000)]
#     y_pred = [max(set(cur_list), key=cur_list.count) for cur_list in list_of_4_rows]
#     y = df.loc[:, ["linqmap_subtype"]]
#     y = y[4:1004]
#     score = f1_score(y, y_pred, average='macro')
#     print(f"value of major vote: {score}")

def kfold_cv(X_train, y_train, X_test, y_test):
    k_range = np.linspace(1, 20, 20).astype(int)

    train_errors, test_errors = [], []
    for k in k_range:
        model = KNeighborsClassifier(k).fit(X_train, y_train)
        train_errors.append(1 - model.score(X_train, y_train))
        test_errors.append(1 - model.score(X_test, y_test))

    param_grid = {'n_neighbors': k_range}
    knn_cv = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='f1_micro').fit(X_train, y_train)
    cv_errors = 1 - knn_cv.cv_results_["mean_test_score"]
    std = knn_cv.cv_results_["std_test_score"]

    min_ind = np.argmin(np.array(cv_errors))
    selected_k = np.array(k_range)[min_ind]
    selected_error = cv_errors[min_ind]

    go.Figure([
        go.Scatter(name='Lower CV Error CI', x=k_range, y=cv_errors - 2 * std, mode='lines',
                   line=dict(color="lightgrey"), showlegend=False, fill=None),
        go.Scatter(name='Upper CV Error CI', x=k_range, y=cv_errors + 2 * std, mode='lines',
                   line=dict(color="lightgrey"), showlegend=False, fill="tonexty"),

        go.Scatter(name="Train Error", x=k_range, y=train_errors, mode='markers + lines',
                   marker_color='rgb(152,171,150)'),
        go.Scatter(name="CV Error", x=k_range, y=cv_errors, mode='markers + lines', marker_color='rgb(220,179,144)'),
        go.Scatter(name="Test Error", x=k_range, y=test_errors, mode='markers + lines', marker_color='rgb(25,115,132)'),
        go.Scatter(name='Selected Model', x=[selected_k], y=[selected_error], mode='markers',
                   marker=dict(color='darkred', symbol="x", size=10))]) \
        .update_layout(title=r"$\text{(4) }k\text{-NN Errors - Selection By Cross-Validation}$",
                       xaxis_title=r"$k\text{ - Number of Neighbors}$",
                       yaxis_title=r"$\text{Error Value}$").show()
    a = 5


    return selected_k, selected_error