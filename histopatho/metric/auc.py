from sklearn.metrics import roc_auc_score
import numpy as np

def auc(labels: np.array, logits: np.array) -> float:
    """ROC AUC score for binary classification.
    Parameters
    ----------
    labels: np.array
        Labels of the outcome.
    logits: np.array
        Probabilities.
    """
    preds = 1.0 / (1.0 + np.exp(-logits))
    return roc_auc_score(labels, preds)