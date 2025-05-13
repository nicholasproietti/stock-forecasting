"""
ensemble.py: Module to create ensemble forecasts.
"""
import numpy as np
from sklearn.linear_model import LinearRegression

def weighted_ensemble(predictions, actual):
    """
    Weighted average ensemble using inverse RMSE weighting.
    
    Parameters:
    predictions (dict): Model name -> forecast array.
    actual (array): True values for error computation.
    
    Returns:
    np.ndarray: Weighted ensemble forecasts.
    """
    # Compute RMSE per model
    rmses = {}
    for name, preds in predictions.items():
        rmses[name] = np.sqrt(np.mean((actual - preds) ** 2))
    # Compute weights proportional to 1/RMSE
    inv = {name: 1.0/r for name,r in rmses.items()}
    total = sum(inv.values())
    weights = {name: inv_val/total for name,inv_val in inv.items()}
    # Weighted sum
    ensemble = np.zeros_like(actual)
    for name, preds in predictions.items():
        ensemble += weights[name] * preds
    return ensemble

def stacking_ensemble(train_preds, train_actual, test_preds):
    """
    Train a linear meta-model on base forecasts (stacking).
    """
    meta = LinearRegression()
    meta.fit(train_preds, train_actual)
    return meta.predict(test_preds)