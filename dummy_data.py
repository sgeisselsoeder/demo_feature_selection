import numpy as np


def get_features_and_labels():
    # from sklearn.datasets import make_regression
    # X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)

    # 100 timesteps, 5 features, 1 label
    dummy_data = np.random.randint(0, 100, size=(100, 4))
    dummy_labels = np.reshape(np.arange(0.0, 10.0, 0.1), (-1, 1))
    labels_np = dummy_labels * 10.0 + 12.5
    features_np = np.concatenate([dummy_data, dummy_labels], axis=1)

    # import pandas as pd
    # dummy_df = pd.DataFrame(features_np, columns=list('ABCDE'))
    # split into features and labels
    # labels_np = dummy_df["E"].values
    # feature_np = dummy_df[["A", "B", "C", "D"]].values
    # features_np = dummy_df.values
    return features_np, labels_np
