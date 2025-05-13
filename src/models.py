"""
models.py: Module for training and forecasting using various models.
"""
import numpy as np
from pmdarima import auto_arima
from arch import arch_model
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from xgboost import XGBRegressor

def forecast_arima(train_series, periods):
    """
    Fit an ARIMA model and forecast future returns.
    """
    model = auto_arima(train_series, seasonal=False, stepwise=True, suppress_warnings=True)
    forecast = model.predict(n_periods=periods)
    return forecast

def forecast_garch(train_series, periods):
    """
    Fit a GARCH(1,1) model (zero mean). 
    Returns zeros as the mean forecast (captures volatility only).
    """
    model = arch_model(train_series, mean='Zero', vol='GARCH', p=1, q=1)
    res = model.fit(disp='off')
    # Forecasted mean is zero; return zeros
    return np.zeros(periods)

def forecast_prophet(df_prophet, periods):
    """
    Fit a Prophet model on historical returns (df with 'ds' and 'y').
    """
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=periods, freq='B')
    forecast = m.predict(future)
    # Return the last 'periods' forecasts
    return forecast['yhat'].tail(periods).values

def forecast_lstm(train_df, test_df, lookback=5):
    """
    Train an LSTM on past returns and forecast future returns (using teacher forcing).
    """
    # Helper to create sequences
    def create_sequences(data, lookback):
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i:(i+lookback)])
            y.append(data[i+lookback])
        return np.array(X), np.array(y)
    # Combine train and test returns for sequence generation
    series = np.concatenate([train_df['Return'].values, test_df['Return'].values])
    X_all, y_all = create_sequences(series, lookback)
    # Split back into train/test
    X_train = X_all[:len(train_df)-lookback]
    y_train = y_all[:len(train_df)-lookback]
    X_test = X_all[len(train_df)-lookback:]
    # Reshape for LSTM [samples, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], lookback, 1))
    X_test = X_test.reshape((X_test.shape[0], lookback, 1))
    # Build LSTM model
    model = Sequential([
        LSTM(50, activation='tanh', input_shape=(lookback, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)
    preds = model.predict(X_test)
    return preds.flatten()

def forecast_xgboost(train_df, test_df):
    """
    Train an XGBoost regressor on engineered features and predict returns.
    """
    features = ['Volatility_21d', 'MA_7', 'MA_21', 'Return_lag1', 'Return_lag2', 'Return_lag3']
    X_train = train_df[features]
    y_train = train_df['Return']
    X_test = test_df[features]
    model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds