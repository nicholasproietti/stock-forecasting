"""
evaluation.py: Module to compute performance metrics and generate visualizations.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

def compute_metrics(actual, predicted):
    """
    Compute RMSE and MAPE between actual and predicted values.
    """
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = mean_absolute_percentage_error(actual, predicted)
    return {'RMSE': rmse, 'MAPE': mape}

def plot_forecasts(dates, actual, forecasts_dict, title, filename):
    """
    Plot actual vs forecasted returns over time for multiple models.
    """
    plt.figure(figsize=(10,6))
    # Actual returns
    plt.plot(dates, actual, label='Actual', color='black', linewidth=1.5)
    # Forecasts
    for name, preds in forecasts_dict.items():
        plt.plot(dates, preds, label=name, linewidth=1)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Daily Return (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()