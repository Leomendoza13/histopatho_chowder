"""Define shuffle function"""

from typing import Tuple
import numpy as np


def shuffle_data(
    values_array: np.ndarray, labels_array: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shuffle the values and corresponding labels arrays in unison.

    Args:
        values_array (np.ndarray): The array of input values.
        labels_array (np.ndarray): The array of corresponding labels.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the shuffled values array
        and the corresponding shuffled labels array.
    """
    indices = np.arange(values_array.shape[0])
    np.random.shuffle(indices)

    values_array = values_array[indices]
    labels_array = labels_array[indices]

    return values_array, labels_array
