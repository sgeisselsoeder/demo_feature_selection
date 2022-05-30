from dummy_data import get_features_and_labels_regression, get_features_and_labels_classification
from feature_selection import greedy_feature_selection
from model import train_RF_and_apply, train_SVC_and_apply
from sklearn.metrics import mean_absolute_error, accuracy_score

# # optimize the feature selection for a regression task with a random forest model and metric mean absolute error
features, labels = get_features_and_labels_regression()
selection = greedy_feature_selection(features=features, labels=labels, model_function=train_RF_and_apply,
                                     metric_function=mean_absolute_error)

# optimize the feature selection for a classification task with a SVM as model and classification accuracy metric
features, labels = get_features_and_labels_classification()
selection2 = greedy_feature_selection(features=features, labels=labels, model_function=train_SVC_and_apply,
                                      metric_function=accuracy_score, minimize_metric=False)
