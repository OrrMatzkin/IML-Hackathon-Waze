import pandas as pd
import numpy as np
from preprocess import preprocess, printProgressBar
from parse_data import make_train_data
from fit_kit import fit_types_and_subtypes
from sklearn.ensemble import RandomForestRegressor




def flat_samples(pre_process_df: pd.DataFrame,):
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


def evaluate_x_y(X_train, x_train_labels, y_train_labels, X_test):
    x_y_model = RandomForestRegressor()
    x_y_model.fit(X_train, x_train_labels)
    pred_x = x_y_model.predict(X_test)

    x_y_model.fit(X_train, y_train_labels)
    pred_y = x_y_model.predict(X_test)

    return pred_x, pred_y

def run_task_1(train_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
    print("Starts task 1...\n")
    # Pre Process
    # print("Preprocessing the Training Data")
    test_data.drop(['test_set'], axis=1, inplace=True)

    preprocess_train = preprocess(train_data, "train")  # all dummies, x,y and update_date
    preprocess_test = preprocess(test_data, "test")

    preprocess_train = preprocess_train[preprocess_train["linqmap_city_Tel Aviv District"] == 1]  # filter only tel aviv
    preprocess_test = preprocess_test[preprocess_test["linqmap_city_Tel Aviv District"] == 1]

    preprocess_train.sort_values(by=['update_date'], inplace=True)  # sorts by time
    preprocess_test.sort_values(by=['update_date'], inplace=True)

    preprocess_train.drop(['update_date'], axis=1, inplace=True)  # drops the time feature
    preprocess_test.drop(['update_date'], axis=1, inplace=True)

    preprocess_train_for_predict = preprocess_train.drop(['x', 'y', 'linqmap_type', 'linqmap_subtype'], axis=1)
    preprocess_test_2 = preprocess_test.drop(['x', 'y', 'linqmap_type', 'linqmap_subtype'], axis=1)

    train_X, train_y_type, train_y_subtype, train_y_x, train_y_y = make_train_data(preprocess_train_for_predict, preprocess_train)

    preprocess_test_flat = flat_samples(preprocess_test_2)

    type_model, y_type_predict, subtypes_models = fit_types_and_subtypes(train_X, train_y_type, train_y_subtype)

    # Predictions
    printProgressBar(0, 13, prefix=f'Predicting data:', suffix='Complete', length=50)

    predict_events = type_model.predict(preprocess_test_flat)

    subtypes_predict = []
    for i, (index, sample) in enumerate(preprocess_test_flat.iterrows()):
        a = sample.values.reshape(1, -1)
        type = predict_events[i]
        model = subtypes_models[type]
        sub_type = model.predict(a)
        subtypes_predict.append(sub_type[0])
        printProgressBar(2+i, 13, prefix=f'Predicting data:', suffix='Complete', length=50)

    x_pred, y_pred = evaluate_x_y(train_X, train_y_x, train_y_y, preprocess_test_flat)


    d = {'linqmap_type': predict_events, 'linqmap_subtype': subtypes_predict, 'x':x_pred, 'y': y_pred}
    final_df = pd.DataFrame(data=d)
    printProgressBar(13, 13, prefix=f'Predicting data:', suffix='Complete', length=50)
    print("Task 1 output:")
    print(final_df)
    print("\nDone task 1!\n")
    return final_df
