"""Define output csv generation function"""

import pandas as pd
import torch


def generate_output_csv(path: str, prediction: torch.tensor, path_generate: str) -> None:
    """
    Generate a CSV file with predicted targets and save it to the specified path.

    Args:
        path (str): Path to the original CSV file containing data to be modified.
        prediction (torch.tensor): Tensor containing predicted values.
        path_generate (str): Path where the generated CSV file will be saved.

    Returns:
        None

    Raises:
        None
    """
    numpy_array = prediction.cpu().data.numpy()
    numpy_array = numpy_array.flatten()
    df_results = pd.read_csv(path)
    df_results = df_results.drop(columns=['Patient ID', 'Center ID'])
    df_results['Target'] = numpy_array
    df_results['Target'] = df_results['Target'].round(4)
    df_results.to_csv(path_generate, index=False, sep=',')
