import numpy as np
from sklearn.ensemble import RandomForestRegressor


def train_RF_and_apply(train_features: np.array, train_labels: np.array, test_features: np.array) -> np.array:
    forest = RandomForestRegressor().fit(X=train_features, y=train_labels.ravel())
    result = forest.predict(X=test_features)
    return result
