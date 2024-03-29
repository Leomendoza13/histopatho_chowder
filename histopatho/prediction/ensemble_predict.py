"""Define ensemble prediction function"""

from typing import List
import torch
from .predict import predict


def ensemble_predict(models: List[torch.nn.Module], test_values: torch.tensor) -> torch.tensor:
    """
    Make predictions using an ensemble of models.

    Args:
        models (List[torch.nn.Module]): A list of PyTorch models for prediction.
        test_values (torch.Tensor): Input tensor for making predictions.

    Returns:
        torch.Tensor: Ensemble prediction tensor obtained by averaging predictions from all models.
    """
    predictions = []
    for model in models:
        prediction = predict(model, test_values)
        predictions.append(prediction)

    ensemble_prediction = torch.stack(predictions).mean(dim=0)
    return ensemble_prediction
