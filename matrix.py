import numpy as np
import pandas as pd
from itertools import product

def create_confusion_matrix(actual, predicted):
    # Flatten the list of labels to get all unique labels
    all_labels = set(label for sublist in actual + predicted for label in sublist)
    label_list = sorted(all_labels)
    label_index = {label: idx for idx, label in enumerate(label_list)}

    # Initialize the confusion matrix
    matrix_size = len(label_list)
    confusion_matrix = np.zeros((matrix_size, matrix_size), dtype=int)

    # Populate the confusion matrix
    for actual_labels, predicted_labels in zip(actual, predicted):
        for a_label, p_label in product(actual_labels, predicted_labels):
            a_idx = label_index[a_label]
            p_idx = label_index[p_label]
            confusion_matrix[a_idx, p_idx] += 1

    # Convert to DataFrame for better readability
    df_confusion_matrix = pd.DataFrame(confusion_matrix, index=label_list, columns=label_list)
    return df_confusion_matrix

# Example usage:
actual = [(1, 8), (6, 7, 9)]
predicted = [(1, 5, 8), (6, 7, 9)]

conf_matrix = create_confusion_matrix(actual, predicted)
print(conf_matrix)
