from os.path import exists
from task_1 import run_task_1
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
import sys
import warnings
from parse_data import *
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import plotly.graph_objects as go
from preprocess import *
from fit_kit import *


# def cross_validation(estimator, X_train, y_train, k_range):
    # param_grid = {'n_neighbors': k_range}
    # knn_cv = GridSearchCV(estimator(), param_grid, cv=5, scoring='f1_micro').fit(X_train, y_train)
    # cv_errors = 1 - knn_cv.cv_results_["mean_test_score"]
    # # std = knn_cv.cv_results_["std_test_score"]
    #
    # min_ind = np.argmin(np.array(cv_errors))
    # selected_k = np.array(k_range)[min_ind]
    # selected_error = cv_errors[min_ind]
    #
    # return selected_k, selected_error


# def get_knn_model(X_train: np.ndarray, y_train: np.ndarray, k: int) -> KNeighborsClassifier:
#     model = KNeighborsClassifier(n_neighbors=k)
#     model.fit(X_train, y_train)
#     return model


def generate_pearson_correlation_heatmap(df):
    corr_df = df.corr()
    go.Figure([go.Heatmap(x=df.columns, y=df.columns, z=corr_df, type='heatmap', colorscale='Viridis')]).show(renderer="browser")


def evalute_location(y_x_predict: pd.DataFrame, y_x_true: pd.DataFrame,
                     y_y_predict: pd.DataFrame, y_y_true: pd.DataFrame) -> float:
    return (mean_squared_error(y_x_predict, y_x_true) + mean_squared_error(y_y_predict, y_y_true))/2


def evaluate_type(y_type_predict: pd.DataFrame, y_type_true: pd.DataFrame) -> float:
    return f1_score(y_type_true, y_type_predict, average="macro")


if __name__ == '__main__':
    np.random.seed(0)
    warnings.filterwarnings('ignore')
    try:
        train_data = sys.argv[1]
        test_data_task_1 = sys.argv[2]
        test_data_task_2 = sys.argv[3]
    except IndexError:
        sys.stderr.write("Error: missing variables, Usage: <train_data> <test_data_task_1> <test_data_task_1>")
    else:
        if not exists(train_data):
            sys.stderr.write("Error: train data file does not exist!")
        elif not exists(test_data_task_1):
            sys.stderr.write("Error: test data for task 1 file does not exist!")
        elif not exists(test_data_task_2):
            sys.stderr.write("Error: test data for task 2 file does not exist!")
        else:
            # runs both tasks
            raw_train_data = pd.read_csv(train_data)
            run_task_1(raw_train_data, pd.read_csv(test_data_task_1))
    # raw_data = pd.read_csv("waze_data.csv")
    # train_data, test_data = train_test_split(raw_data, test_size=.2, random_state=42)
    #
    # preprocess_data = preprocess(train_data, True if geolocator_use == "-g" else False)  # all dummies, x,y and update_date
    # tel_aviv_data = preprocess_data[preprocess_data["linqmap_city_Tel Aviv District"] == 1]
    # tel_aviv_data.sort_values(by=['update_date'], inplace=True)
    # tel_aviv_data.drop(['update_date'], axis=1, inplace=True)
    # tel_aviv_data_for_samples = tel_aviv_data.drop(['x', 'y', 'linqmap_type', 'linqmap_subtype'], axis=1)
    #
    # X, y_type, y_subtype, y_x, y_y = make_train_data(tel_aviv_data_for_samples, tel_aviv_data)
    #
    # # train_X, valid_X, train_y_type, valid_y_type, train_y_subtype, valid_y_subtype, train_y_x, valid_y_x, train_y_y, valid_y_y = train_test_split(X, y_type, y_subtype, y_x, y_y, test_size=0.2, random_state=42,  shuffle=True)
    # type_model, y_type_predict, subtypes_models = fit_types_and_subtypes(X, y_type, y_subtype)
    # # #
    # # # k = cross_validation(KNeighborsClassifier, train_X, train_y_subtype, np.linspace(1, 20, 20).astype(int))[0]
    # # # model_types = get_knn_model(train_X, train_y_subtype, k)
    # # #
    # # # print(f"score: {evaluate_type(model_types.predict(train_X), train_y_subtype)}")
    # #
    # # ev = evaluate_type(y_type_predict, train_y_type)
    # # print(f"model type score {ev}")
    # # for type, model, predict_y, true_y in subtypes_models:
    # #     print(f"model subtype {type} score {evaluate_type(predict_y, true_y)}")
    # # print("####")
    # # y_predict = type_model.predict(valid_X)
    # # print(f"model type score valid data {evaluate_type(y_predict, valid_y_type)}")
    # # for type_name, model, predict_y, true_y in subtypes_models:
    # #     msk = np.where(y_predict == type_name, True, False)
    # #     new_X_valid = valid_X[msk]
    # #     new_y_valid = valid_y_subtype[msk]
    # #     if new_X_valid.empty:
    # #         continue
    # #     print(f"model subtype {type_name} score {evaluate_type(model.predict(new_X_valid), new_y_valid)}")
    # #
    # # print("#### all train ####")
    # # type_model, y_type_predict, subtypes_models = fit_types_and_subtypes(X, y_type, y_subtype)
    #
    # preprocess_data = preprocess(test_data, True if geolocator_use == "-g" else False)  # all dummies, x,y and update_date
    # tel_aviv_test_data = preprocess_data[preprocess_data["linqmap_city_Tel Aviv District"] == 1]
    # tel_aviv_test_data.sort_values(by=['update_date'], inplace=True)
    # tel_aviv_test_data.drop(['update_date'], axis=1, inplace=True)
    # tel_aviv_test_data_for_samples = tel_aviv_test_data.drop(['x', 'y', 'linqmap_type', 'linqmap_subtype'], axis=1)
    # X_test, y_type_test, y_subtype_test, y_x_test, y_y_test = make_train_data(tel_aviv_test_data_for_samples, tel_aviv_test_data)
    #
    #
    # ev = evaluate_type(y_type_predict, y_type)
    # print(f"model type score {ev}")
    # for type, model, predict_y, true_y in subtypes_models:
    #     print(f"model subtype {type} score {evaluate_type(predict_y, true_y)}")
    # print("####")
    # y_predict = type_model.predict(X_test)
    # print(f"model type score valid data {evaluate_type(y_predict, y_type_test)}")
    # for type_name, model, predict_y, true_y in subtypes_models:
    #     msk = np.where(y_predict == type_name, True, False)
    #     new_X_valid = X_test[msk]
    #     new_y_valid = y_subtype_test[msk]
    #     if new_X_valid.empty:
    #         continue
    #     print(f"model subtype {type_name} score {evaluate_type(model.predict(new_X_valid), new_y_valid)}")
    #
    # # selected_k, selected_error = cross_validation(KNeighborsClassifier, train_X_flatten, y_flatten ,valid_X_flatten, valid_y_flatten, np.linspace(1, 20, 10).astype(int))
    # # type_model = get_knn_model(train_X_flatten, y_flatten,valid_X_flatten, valid_y_flatten, selected_k)
    # print("done")
