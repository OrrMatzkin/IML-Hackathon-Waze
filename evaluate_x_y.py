from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score, mean_squared_error


def evaluate_x_y(X_train, x_train_labels, y_train_labels, X_test):
    x_y_model = RandomForestRegressor()
    x_y_model.fit(X_train, x_train_labels)
    pred_x = x_y_model.predict(X_test)

    x_y_model.fit(X_train, y_train_labels)
    pred_y = x_y_model.predict(X_test)

    return pred_x, pred_y

