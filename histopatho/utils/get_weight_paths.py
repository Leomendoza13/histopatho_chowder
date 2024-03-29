"""Define function to get paths of pth files"""

import os


def get_weight_paths(directory_path: str) -> list:
    """
    Retrieve paths of weight files (ending with '.pth') within the specified directory.

    Args:
        directory_path (str): The path to the directory containing weight files.

    Returns:
        list: A list of file paths of weight files within the directory.
    """
    list_paths = []

    for file in os.listdir(directory_path):
        if file.endswith('pth'):
            list_paths.append(file)
    return list_paths
