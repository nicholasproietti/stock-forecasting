"""
main.py: Main script to run the forecasting pipeline.
"""
import pandas as pd
from data_loader import download_stock_data
from features import add_technical_indicators
from models import forecast_arima, forecast_garch, forecast_prophet, forecast_lstm, forecast_xgboost
from evaluation import compute_metrics, plot_forecasts
from ensemble import weighted_ensemble

def main():
    # Parameters
    ticker = 'AMZN'
    start_date = '2013-01-01'
    end_date = '2023-01-01'
    test_years = 2

    # 1. Load data
    df = download_stock_data(ticker, start_date, end_date)
    df = add_technical_indicators(df)

    # 2. Train/test split (last 2 years as test)
    test_size = test_years * 252  # ~252 trading days/year
    train_df = df.iloc[:-test_size].copy()
    test_df = df.iloc[-test_size:].copy()
    train_df.index.name = 'Date'
    test_df.index.name = 'Date'

    # 3. Individual forecasts
    arima_preds = forecast_arima(train_df['Return'], len(test_df))
    garch_preds = forecast_garch(train_df['Return'], len(test_df))
    prophet_input = train_df.reset_index().rename(columns={'Date':'ds', 'Return':'y'})
    prophet_preds = forecast_prophet(prophet_input, len(test_df))
    lstm_preds = forecast_lstm(train_df, test_df, lookback=5)
    xgb_preds = forecast_xgboost(train_df, test_df)

    # 4. Evaluate each model
    actual = test_df['Return'].values
    models = {
        'ARIMA': arima_preds,
        'GARCH': garch_preds,
        'Prophet': prophet_preds,
        'LSTM': lstm_preds,
        'XGBoost': xgb_preds
    }
    for name, preds in models.items():
        m = compute_metrics(actual, preds)
        print(f"{name}: RMSE={m['RMSE']:.4f}, MAPE={m['MAPE']:.4f}")

    # 5. Weighted ensemble
    ensemble_w = weighted_ensemble(models, actual)
    me = compute_metrics(actual, ensemble_w)
    print(f"Weighted Ensemble: RMSE={me['RMSE']:.4f}, MAPE={me['MAPE']:.4f}")

    # 6. Plot forecasts vs actual
    forecasts = {
        'ARIMA': arima_preds,
        'Prophet': prophet_preds,
        'LSTM': lstm_preds,
        'XGBoost': xgb_preds,
        'Ensemble': ensemble_w
    }
    plot_forecasts(test_df.index, actual, forecasts,
                   title=f"Forecasts vs Actual for {ticker}", 
                   filename='figures/forecasts_vs_actual.png')

if __name__ == "__main__":
    main()