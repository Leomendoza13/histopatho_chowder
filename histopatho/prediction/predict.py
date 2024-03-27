"""Define Predict"""

import torch


def predict(model: torch.nn.Module, test_tensor: torch.Tensor) -> torch.Tensor:
    """
    Makes predictions using the given PyTorch model on the input data.

    Args:
        model (torch.nn.Module): The PyTorch model to use for making predictions.
        input (torch.Tensor): The input data for which predictions are to be made.

    Returns:
        torch.Tensor: Predicted output tensor obtained from the model.
    """
    output = model(test_tensor)
    prediction = torch.sigmoid(output)
    return prediction
