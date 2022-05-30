import numpy as np


def train_RF_and_apply(train_features: np.array, train_labels: np.array, test_features: np.array) -> np.array:
    from sklearn.ensemble import RandomForestRegressor
    forest = RandomForestRegressor().fit(X=train_features, y=train_labels.ravel())
    result = forest.predict(X=test_features)
    return result


def train_SVC_and_apply(train_features: np.array, train_labels: np.array, test_features: np.array) -> np.array:
    from sklearn.svm import SVC
    model = SVC(kernel='linear').fit(X=train_features, y=train_labels.ravel())
    result = model.predict(X=test_features)
    return result
