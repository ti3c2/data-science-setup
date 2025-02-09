from sklearn.metrics import mean_squared_error
import numpy as np


def rmse(y_true, y_predict):
    return np.sqrt(mean_squared_error(y_true, y_predict))