"""Define split function"""

from typing import Tuple
import numpy as np
from torch.utils.data import Subset
from torch.utils.data import Dataset


def split_dataset_in_subset(train_dataset: Dataset, ratio: int) -> Tuple[Subset, Subset]:
    """
    Splits a given training dataset into two subsets: a training subset and a validation subset.

    Args:
        train_dataset (Dataset): The original training dataset to be split.

    Returns:
        Tuple[Subset, Subset]: A tuple containing two subsets - the train subset and the val subset.
    """

    split_value = len(train_dataset) * ratio // 100
    train_indices = np.arange(split_value)
    val_indices = np.arange(split_value, len(train_dataset))

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    return train_subset, val_subset
