import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from main import cross_validation


def fit_types_and_subtypes(X_train, y_train):
    k = cross_validation(KNeighborsClassifier, X_train, y_train, np.linspace(1, 20, 20))[0]
    model_types = get_knn_model(X_train, y_train, k)

    model_sub_types = []
    for i in range(4):
        msk = X_train.loc[:, ['linqmap_type']] == i
        new_X_train = X_train[msk]
        new_y_train = y_train[msk]

        k = cross_validation(KNeighborsClassifier, new_X_train, new_y_train, np.linspace(1, 20, 20))[0]
        model_sub_types.append(get_knn_model(new_X_train, new_y_train, k))

    return model_types, model_sub_types

def predict_type_and_subtype(real_test, model_types, model_sub_types):
    types_pred = model_types.predict(real_test)

    m = real_test.shape[0]
    subtype_pred = np.zeros(m).astype(int)
    for i in range(4):
        msk = real_test.loc[:, ['linqmap_type']] == i
        filtered_X_test = real_test[msk]
        filtered_y_test = model_sub_types[type].predict(filtered_X_test)
        subtype_pred[msk] = filtered_y_test
    return types_pred, subtype_pred