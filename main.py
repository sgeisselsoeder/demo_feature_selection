from dummy_data import get_features_and_labels
from feature_selection import greedy_feature_selection
from model import train_RF_and_apply

features, labels = get_features_and_labels()
selection = greedy_feature_selection(features=features, labels=labels, model_function=train_RF_and_apply)
