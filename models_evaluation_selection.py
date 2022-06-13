from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, mean_squared_error
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from copy import deepcopy
import numpy as np
import pandas as pd

"""
This entire file is for our own use only!
We used those function to find our chosen model, create graphs for evaluation and give our models scores.
"""


def cross_validation(estimator, X_train, y_train, k_range):
    param_grid = {'n_neighbors': k_range}
    knn_cv = GridSearchCV(estimator(), param_grid, cv=10, scoring='f1_macro').fit(X_train, y_train)
    cv_errors = 1 - knn_cv.cv_results_["mean_test_score"]
    # std = knn_cv.cv_results_["std_test_score"]
    min_ind = np.argmin(np.array(cv_errors))
    selected_k = np.array(k_range)[min_ind]
    selected_error = cv_errors[min_ind]

    return selected_k, selected_error


def get_knn_model(X_train: np.ndarray, y_train: np.ndarray, k: int) -> KNeighborsClassifier:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model


def generate_pearson_correlation_heatmap(df):
    corr_df = df.corr()
    go.Figure([go.Heatmap(x=df.columns, y=df.columns, z=corr_df, type='heatmap', colorscale='Viridis')]).show(
        renderer="browser")


def evaluate_location(y_x_predict: pd.DataFrame, y_x_true: pd.DataFrame,
                      y_y_predict: pd.DataFrame, y_y_true: pd.DataFrame) -> float:
    return (mean_squared_error(y_x_predict, y_x_true) + mean_squared_error(y_y_predict, y_y_true)) / 2


def evaluate_type(y_type_predict: pd.DataFrame, y_type_true: pd.DataFrame) -> float:
    return f1_score(y_type_true, y_type_predict, average="macro")


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

    return selected_k, selected_error