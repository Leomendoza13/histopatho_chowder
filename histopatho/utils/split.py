from typing import Tuple
import numpy as np
from torch.utils.data import Subset
from torch.utils.data import Dataset

def split_train_val_dataset_to_subset(train_dataset: Dataset) -> Tuple[Subset, Subset]:
    """
    Splits a given training dataset into two subsets: a training subset and a validation subset.

    Parameters:
    - train_dataset (Dataset): The original training dataset to be split.

    Returns:
    - Tuple[Subset, Subset]: A tuple containing two subsets - the training subset and the validation subset.

    Example:
    ```
    import pandas as pd
    from torch.utils.data import Subset
    from rl_benchmarks.datasets import SlideFeaturesDataset

    # Load train values and labels
    train_values = pd.read_csv("train_values.csv")
    train_labels = pd.read_csv("train_labels.csv")
    
    # Load CIFAR-10 dataset
       train_dataset = SlideFeaturesDataset(
        features = train_values,
        labels = train_labels,
    )

    # Split the training dataset into training and validation subsets
    train_subset, val_subset = split_train_val_to_subset(train_dataset)
    ```
    """
    
    split_value = len(train_dataset) * 80 // 100
    train_indices = np.arange(split_value)
    val_indices = np.arange(split_value, len(train_dataset))

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    return train_subset, val_subset