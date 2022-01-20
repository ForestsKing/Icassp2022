import numpy as np


def get_score(pred, label):
    pred, label = np.array(pred), np.array(label)
    plus = np.sum(pred * label, axis=1)
    minus = np.sum(pred * (1 - label), axis=1)
    return np.mean((plus - minus) / np.sum(label, axis=1))