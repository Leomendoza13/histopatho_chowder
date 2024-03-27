import os
import numpy as np
import pandas as pd

def load_data_from_npy(path: str) -> np.ndarray:
    """
    Loads data from .npy files located in the specified directory.

    Args:
        path (str): The path to the directory containing .npy files.

    Returns:
        numpy.ndarray: A 3D numpy array containing the loaded data.
        
    Raises:
        FileNotFoundError: If the specified directory does not exist or does not contain any .npy files.
        ValueError: If the loaded data arrays have inconsistent shapes.
        IOError: If there is an error loading the data from .npy files.
    """
    #List files in the directory and sort them
    files = os.listdir(path)
    files.sort() 

    #Initialize empty list to store loaded data
    my_list = []

    # Load data from each .npy file
    for file in files:
        npy_path = os.path.join(path, file)
        data = np.load(npy_path)

        # Append the loaded data to the list
        my_list.append(data)

    # Stack the arrays along the first dimension and convert to float32
    data_array = np.stack(my_list).astype(np.float32)

    return data_array

def load_data_from_csv(path: str) -> np.ndarray:
    """
    Loads data from a CSV file located at the specified path.

    Parameters:
        path (str): The file path to the CSV file.

    Returns:
        np.ndarray: A NumPy array containing the data extracted from the CSV file. 
                    The data is extracted from the 'Target' column of the CSV file
                    and converted to a NumPy array of dtype np.float32.
    """
    df_data = pd.read_csv(path)
    data_array = df_data['Target'].to_numpy().astype(np.float32)
    
    return data_array