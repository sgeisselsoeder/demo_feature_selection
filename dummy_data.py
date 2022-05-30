import numpy as np


def get_features_and_labels_regression():
    # from sklearn.datasets import make_regression
    # X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)

    # 100 timesteps, 5 features, 1 label
    dummy_data = np.random.randint(0, 100, size=(100, 4))
    dummy_labels = np.reshape(np.arange(0.0, 10.0, 0.1), (-1, 1))
    labels_np = dummy_labels * 10.0 + 12.5
    features_np = np.concatenate([dummy_data, dummy_labels], axis=1)
    return features_np, labels_np


def get_features_and_labels_classification():
    features_np, labels_np = get_features_and_labels_regression()
    labels_np[labels_np <= 58.0] = 0
    labels_np[labels_np > 58.0] = 1
    return features_np, labels_np.astype(int)
