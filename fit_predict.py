import numpy as np
from typing import Dict, Tuple
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from preprocess import print_progress_bar
from preprocess import EVENT_TYPES


def fit_types_and_subtypes(X_train: pd.DataFrame, y_train_type: pd.DataFrame, y_train_subtype: pd.DataFrame)\
        -> Tuple[ExtraTreesClassifier, pd.DataFrame, Dict[str, ExtraTreesClassifier]]:
    """
    Fits the 5 ExtraTreesClassifier models, one for the type and 4 models for the subtypes.
    :param X_train: train data.
    :param y_train_type: train type label.
    :param y_train_subtype: train subtype labels.
    :return: the type model, type label predict, Dict with key = subtype name, value = the subtype model.
    """
    print_progress_bar(0, 4, prefix='Training:', suffix='Complete', length=50)
    # k = cross_validation(KNeighborsClassifier, X_train, y_train_type, np.linspace(1, 20, 20).astype(int))[0]
    # model_types = get_knn_model(X_train, y_train_type, k)
    model_types = ExtraTreesClassifier()
    model_types.fit(X_train, y_train_type)
    y_predict = model_types.predict(X_train)

    model_sub_types = {}
    i = 0
    for type_name in EVENT_TYPES.keys():
        msk = np.where(y_predict == type_name, True, False)
        new_X_train = X_train[msk]
        new_y_train = y_train_subtype[msk]
        if new_X_train.empty:
            model_sub_types.append(None)
            continue
        # k = cross_validation(KNeighborsClassifier, new_X_train, new_y_train, np.linspace(1, 20, 20).astype(int))[0]
        # model = get_knn_model(new_X_train, new_y_train, k)
        model = ExtraTreesClassifier()
        model.fit(new_X_train, new_y_train)
        model_sub_types[type_name] = model
        print_progress_bar(i + 1, 4, prefix='Training:', suffix='Complete', length=50)
        i += 1
    return model_types, y_predict, model_sub_types


def predict_type_and_subtype(test_data: pd.DataFrame, model_types: pd.DataFrame, model_sub_types: pd.DataFrame)\
        -> (pd.DataFrame, pd.DataFrame):
    """
    Return the ExtraTreesClassifier model prediction for the type and subtype
    :param test_data: test data (4 events).
    :param model_types: the type model
    :param model_sub_types: the relevant subtype model
    :return: type prediction, subtype prediction
    """
    types_pred = model_types.predict(test_data)
    m = test_data.shape[0]
    subtype_pred = np.zeros(m).astype(int)
    for i in range(4):
        msk = test_data.loc[:, ['linqmap_type']] == i
        filtered_X_test = test_data[msk]
        filtered_y_test = model_sub_types[type].predict(filtered_X_test)
        subtype_pred[msk] = filtered_y_test
    return types_pred, subtype_pred