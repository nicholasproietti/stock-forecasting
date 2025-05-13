"""
data_loader.py: Module to download stock data and compute base returns.
"""
import yfinance as yf
import pandas as pd

def download_stock_data(ticker, start_date, end_date):
    """
    Downloads historical stock price data using yfinance.
    
    Parameters:
    ticker (str): Stock ticker symbol (e.g., 'AAPL').
    start_date (str): Start date in 'YYYY-MM-DD' format.
    end_date (str): End date in 'YYYY-MM-DD' format.
    
    Returns:
    pd.DataFrame: DataFrame with daily close prices and returns (%).
    """
    # Download data
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    # Keep only the 'Close' price
    df = df[['Close']]
    # Ensure business-day frequency and forward-fill missing days
    df = df.asfreq('B').ffill()
    # Compute daily returns as percentage
    df['Return'] = df['Close'].pct_change() * 100
    # Drop the first NaN return
    df = df.dropna(subset=['Return'])
    return df