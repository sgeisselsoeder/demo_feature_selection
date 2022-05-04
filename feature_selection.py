import numpy as np
import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def _evaluate_one_selection(features: np.array, labels: np.array, model_function, feature_selection_to_test: list,
                            metric=mean_absolute_error, shuffle: bool = False) -> float:
    # TODO: prefer crossvalidation to single train/test split
    current_features = features[:, feature_selection_to_test]
    train_feat, test_feat, train_labels, test_labels = train_test_split(current_features, labels, shuffle=shuffle)

    result = model_function(train_features=train_feat, train_labels=train_labels, test_features=test_feat)
    error = metric(result, test_labels)
    return error


def greedy_feature_selection(features: np.array, labels: np.array, model_function, metric=mean_absolute_error,
                             max_solution_length: int = 2, shuffle: bool = False) -> list:
    current_selection = []
    best_overall_selection = []
    best_overall_error = None

    for i in range(max_solution_length):
        print("Searching next feature beyond current selection: ", current_selection)
        best_next_feature = None
        current_performance = None

        # TODO: optional remaining_features without already selected features, prevents double selection of the same feature
        remaining_features = features
        for next_feature_index in range(remaining_features.shape[1]):
            print("Testing with next feature", next_feature_index)

            feature_selection_to_test = current_selection + [next_feature_index]
            error = _evaluate_one_selection(features=features, labels=labels, model_function=model_function, metric=metric,
                                            feature_selection_to_test=feature_selection_to_test, shuffle=shuffle)
            print("Obtained error ", error)

            # compare the performance of the current selection to the so far best selection of this iteration
            if current_performance is None or error < current_performance:
                print("Found new best candidate ", next_feature_index)
                print(next_feature_index, " with ", error, " was better than ", best_next_feature, " with ", current_performance)
                best_next_feature = next_feature_index
                current_performance = error

        current_selection.append(best_next_feature)
        print("Found current best selection with error", current_performance, " to be", current_selection)

        # compare the best selection of this iteration to the overall best
        if best_overall_error is None or current_performance < best_overall_error:
            best_overall_selection = copy.copy(current_selection)
            best_overall_error = current_performance

    print("Best selection with error ", best_overall_error, " was ", best_overall_selection)
    return best_overall_selection
