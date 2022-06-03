import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_score
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from preprocess import printProgressBar

types = {'ACCIDENT': 0, 'JAM': 1, 'ROAD_CLOSED': 2, "WEATHERHAZARD": 3}


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


def fit_types_and_subtypes(X_train, y_train_type, y_train_subtype):
    printProgressBar(0, 4, prefix='Training:', suffix='Complete', length=50)
    # k = cross_validation(KNeighborsClassifier, X_train, y_train_type, np.linspace(1, 20, 20).astype(int))[0]
    # model_types = get_knn_model(X_train, y_train_type, k)
    model_types = ExtraTreesClassifier()
    # model_types = ExtraTreesClassifier()
    model_types.fit(X_train, y_train_type)
    y_predict = model_types.predict(X_train)

    model_sub_types = {}
    i = 0
    for type_name in types.keys():
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
        printProgressBar(i + 1, 4, prefix='Training:', suffix='Complete', length=50)
        i += 1
    return model_types, y_predict, model_sub_types


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