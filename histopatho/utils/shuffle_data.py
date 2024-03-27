import numpy as np
from typing import Tuple

def shuffle_data(values_array: np.ndarray, labels_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shuffle the values and corresponding labels arrays in unison.

    Args:
        values_array (np.ndarray): The array of input values.
        labels_array (np.ndarray): The array of corresponding labels.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the shuffled values array
        and the corresponding shuffled labels array.

    Example:
        values = np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
        labels = np.array([0., 1., 0.])
        shuffled_values, shuffled_labels = shuffle_data(values, labels)
    """
    indices = np.arange(values_array.shape[0])
    np.random.shuffle(indices)

    values_array = values_array[indices]
    labels_array = labels_array[indices]

    return values_array, labels_array