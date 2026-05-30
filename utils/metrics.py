import numpy as np

def accuracy(y_true, y_pred):

    y_pred_labels = (y_pred >= 0.5).astype(int)

    accuracy = np.mean(y_true == y_pred_labels)

    return accuracy