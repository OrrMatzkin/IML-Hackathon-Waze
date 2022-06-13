from preprocess import preprocess_task1, print_progress_bar
from fit_predict import fit_types_and_subtypes
from sklearn.ensemble import RandomForestRegressor
import pandas as pd


def flat_samples(pre_process_df: pd.DataFrame, ) -> pd.DataFrame:
    """
    Returns a flat data frame, Every 4 samples are combined to a single sample.
    :param pre_process_df: raw data frame.
    :return: flat data frame.
    """
    no_shift = pre_process_df
    shift1 = pre_process_df.shift(periods=-1)
    shift2 = pre_process_df.shift(periods=-2)
    shift3 = pre_process_df.shift(periods=-3)

    no_shift = no_shift.add_suffix("_0")
    shift1 = shift1.add_suffix("_1")
    shift2 = shift2.add_suffix("_2")
    shift3 = shift3.add_suffix("_3")

    flatted_4_samples = pd.concat([no_shift, shift1, shift2, shift3], axis=1)
    flatted_4_samples = flatted_4_samples.iloc[:-3]
    flatted_4_samples = flatted_4_samples[flatted_4_samples.index % 4 == 0]
    return flatted_4_samples


def make_train_data(pre_process_df: pd.DataFrame, not_processes_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame,
                                                                                      pd.DataFrame, pd.DataFrame):
    """
    Turns the data into train data by adding labels for series of 4 events sorted by time.
    :param pre_process_df: preprocessed data.
    :param not_processes_df: not preprocessed data.
    :return:  X train data , type labels, subtype labels, x (coordinate) labels, y (coordinate) labels
    """
    no_shift = pre_process_df
    shift1 = pre_process_df.shift(periods=-1)
    shift2 = pre_process_df.shift(periods=-2)
    shift3 = pre_process_df.shift(periods=-3)
    shift4_for_labels = not_processes_df.shift(periods=-4)

    no_shift = no_shift.add_suffix("_0")
    shift1 = shift1.add_suffix("_1")
    shift2 = shift2.add_suffix("_2")
    shift3 = shift3.add_suffix("_3")

    flatted_4_samples = pd.concat([no_shift, shift1, shift2, shift3], axis=1)

    type_labels = shift4_for_labels['linqmap_type']
    subtype_labels = shift4_for_labels['linqmap_subtype']
    x_labels = shift4_for_labels['x']
    y_labels = shift4_for_labels['y']

    return flatted_4_samples.iloc[:-4], type_labels.iloc[:-4], subtype_labels.iloc[:-4], \
           x_labels.iloc[:-4], y_labels.iloc[:-4]


def evaluate_x_y(X_train, x_train_labels, y_train_labels, X_test) -> (float, float):
    """
    Predicts with RandomForestRegressor event coordinates (x,y).
    :param X_train: train data.
    :param x_train_labels: train labels (of x coordinates).
    :param y_train_labels: train data (of y coordinates).
    :param X_test: test data.
    :return: predicted (x,y).
    """
    x_y_model = RandomForestRegressor()
    x_y_model.fit(X_train, x_train_labels)
    pred_x = x_y_model.predict(X_test)

    x_y_model.fit(X_train, y_train_labels)
    pred_y = x_y_model.predict(X_test)

    return pred_x, pred_y


def run_task_1(train_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
    """
    Gives predictions for task No. 1.
    :param train_data: train data.
    :param test_data: test data
    :return: predictions.
    """
    print("Starts task 1...\n")
    # Pre Process
    # print("Preprocessing the Training Data")
    test_data.drop(['test_set'], axis=1, inplace=True)

    preprocess_train = preprocess_task1(train_data, "train")  # all dummies, x,y and update_date
    preprocess_test = preprocess_task1(test_data, "test")

    preprocess_train = preprocess_train[preprocess_train["linqmap_city_Tel Aviv District"] == 1]  # filter only tel aviv
    preprocess_test = preprocess_test[preprocess_test["linqmap_city_Tel Aviv District"] == 1]

    preprocess_train.sort_values(by=['update_date'], inplace=True)  # sorts by time
    preprocess_test.sort_values(by=['update_date'], inplace=True)

    preprocess_train.drop(['update_date'], axis=1, inplace=True)  # drops the time feature
    preprocess_test.drop(['update_date'], axis=1, inplace=True)

    preprocess_train_for_predict = preprocess_train.drop(['x', 'y', 'linqmap_type', 'linqmap_subtype'], axis=1)
    preprocess_test_2 = preprocess_test.drop(['x', 'y', 'linqmap_type', 'linqmap_subtype'], axis=1)

    train_X, train_y_type, train_y_subtype, train_y_x, train_y_y = make_train_data(preprocess_train_for_predict,
                                                                                   preprocess_train)

    preprocess_test_flat = flat_samples(preprocess_test_2)

    type_model, y_type_predict, subtypes_models = fit_types_and_subtypes(train_X, train_y_type, train_y_subtype)

    # Predictions
    print_progress_bar(0, 13, prefix=f'Predicting data:', suffix='Complete', length=50)

    predict_events = type_model.predict(preprocess_test_flat)

    subtypes_predict = []
    for i, (index, sample) in enumerate(preprocess_test_flat.iterrows()):
        a = sample.values.reshape(1, -1)
        type = predict_events[i]
        model = subtypes_models[type]
        sub_type = model.predict(a)
        subtypes_predict.append(sub_type[0])
        print_progress_bar(2 + i, 13, prefix=f'Predicting data:', suffix='Complete', length=50)

    x_pred, y_pred = evaluate_x_y(train_X, train_y_x, train_y_y, preprocess_test_flat)

    d = {'linqmap_type': predict_events, 'linqmap_subtype': subtypes_predict, 'x': x_pred, 'y': y_pred}
    final_df = pd.DataFrame(data=d)
    print_progress_bar(13, 13, prefix=f'Predicting data:', suffix='Complete', length=50)
    print("Task 1 output:")
    print(final_df)
    print("\nDone task 1!\n")

    return final_df
