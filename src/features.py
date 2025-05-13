"""
features.py: Module for creating technical features from stock price data.
"""
import pandas as pd

def add_technical_indicators(df):
    """
    Adds technical indicators like volatility, moving averages, and lagged returns.
    
    Parameters:
    df (pd.DataFrame): DataFrame with 'Close' and 'Return' columns.
    
    Returns:
    pd.DataFrame: DataFrame augmented with new feature columns.
    """
    df = df.copy()
    # 21-day rolling volatility of returns
    df['Volatility_21d'] = df['Return'].rolling(window=21).std()
    # 7-day and 21-day moving averages of price
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_21'] = df['Close'].rolling(window=21).mean()
    # Lagged returns for previous days
    for lag in [1, 2, 3]:
        df[f'Return_lag{lag}'] = df['Return'].shift(lag)
    # Drop rows with NaN values (from rolling/lagging)
    df = df.dropna()
    return df