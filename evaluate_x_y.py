import numpy as np
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
import plotly.graph_objects as go

def check_evaluate_x_y(df, y_for_x_result, y_for_y_result):
    k_range = np.linspace(1, 20, 20).astype(int)
    train_errors_x, test_errors_x = [], []
    train_errors_y, test_errors_y = [], []

    for k in k_range:
        print(f"values of: {k}")
        modelX = KNeighborsRegressor(n_neighbors=k)
        modelY = KNeighborsRegressor(n_neighbors=k)

        train_X, valid_X, train_y, valid_y = train_test_split(df, y_for_x_result, test_size=0.2, shuffle=True)
        modelX.fit(train_X, train_y)
        pred_for_x = modelX.predict(train_X)
        score = mean_squared_error(train_y, pred_for_x)
        train_errors_x.append(score)
        print(f"for x: value in KNeighborsRegressor on the train set with MSE: {score}")
        pred_for_x = modelX.predict(valid_X)
        score = mean_squared_error(valid_y, pred_for_x)
        test_errors_x.append(score)
        print(f"for x: value in KNeighborsRegressor on the test set with MSE: {score}")

        train_X, valid_X, train_y, valid_y = train_test_split(df, y_for_y_result, test_size=0.2, shuffle=True)
        modelY.fit(train_X, train_y)

        pred_for_y = modelY.predict(train_X)
        score = mean_squared_error(train_y, pred_for_y)
        train_errors_y.append(score)
        print(f"for x: value in KNeighborsRegressor on the train set with MSE: {score}")
        pred_for_y = modelY.predict(valid_X)
        score = mean_squared_error(valid_y, pred_for_y)
        test_errors_y.append(score)
        print(f"for x: value in KNeighborsRegressor on the test set with MSE: {score}")


    go.Figure([
        go.Scatter(name="Train Error", x=k_range, y=train_errors_x, mode='markers + lines',
                   marker_color='rgb(152,171,150)'),
        go.Scatter(name="Test Error", x=k_range, y=test_errors_x, mode='markers + lines', marker_color='rgb(25,115,132)')])\
    .update_layout(title=r"$\text{(4) }k\text{-NN Errors - Selection By Cross-Validation}$",
                   xaxis_title=r"$k\text{ - Number of Neighbors}$",
                   yaxis_title=r"$\text{Error Value}$").show()


def final_evaluate(df, y_for_x_result, y_for_y_result, test):
    modelX = KNeighborsRegressor(n_neighbors=9)
    modelY = KNeighborsRegressor(n_neighbors=9)

    modelX.fit(df, y_for_x_result)
    pred_for_test_x = modelX.predict(test)

    modelX.fit(df, y_for_y_result)
    pred_for_test_y = modelY.predict(test)

    return pred_for_test_x, pred_for_test_y