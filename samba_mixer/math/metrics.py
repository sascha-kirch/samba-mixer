import numpy as np
import sklearn.metrics


def root_mean_squared_error(prediction: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Calculates the RMSE aka. root mean squared error between two tensors.

    Args:
        prediction (np.ndarray): Predictions from a neuronal network
        label (np.ndarray): Ground truth labels.

    Returns:
        np.ndarray: RMSE
    """
    return np.sqrt(sklearn.metrics.mean_squared_error(prediction, label))


def mape(prediction: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Calculates the MAPE aka. mean absolute percentage error between two tensors.

    Since the sklearn implementation of MAPE outputs the error in a range of [0,1],
    the result must be multiplied by 100 to obtain actual percentage values.

    Args:
        prediction (np.ndarray): Predictions from a neuronal network
        label (np.ndarray): Ground truth labels.

    Returns:
        np.ndarray: MAPE
    """
    return sklearn.metrics.mean_absolute_percentage_error(y_true=label, y_pred=prediction) * 100
