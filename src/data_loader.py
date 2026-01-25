import yfinance as yf
import pandas as pd
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_data(tickers, start_date, end_date):
    """
    Fetches historical data for the given tickers from yfinance.
    
    Args:
        tickers (list): List of ticker symbols (e.g., ['TSLA', 'BND', 'SPY']).
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        
    Returns:
        pd.DataFrame: Combined DataFrame with all tickers.
    """
    logging.info(f"Fetching data for {tickers} from {start_date} to {end_date}...")
    
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)
    
    # Check if data is empty
    if data.empty:
        logging.warning("No data fetched. Please check your tickers or date range.")
        return None
    
    # Flatten MultiIndex columns if necessary
    # yfinance returns MultiIndex (Ticker, Price Type) if group_by='ticker'
    # Or (Price Type, Ticker) if group_by='column' (default)
    # With group_by='ticker', columns are like ('TSLA', 'Open'), ('TSLA', 'Close')...
    
    logging.info("Data fetched successfully.")
    return data

def clean_data(data):
    """
    Cleans the fetched data.
    
    Args:
        data (pd.DataFrame): Raw data from yfinance.
        
    Returns:
        pd.DataFrame: Cleaned data.
    """
    # Check for missing values
    if data.isnull().sum().sum() > 0:
        logging.info("Missing values found. Fulfilling with forward fill...")
        data = data.ffill().bfill()
        
    return data

def save_data(data, filepath):
    """
    Saves the data to a CSV file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data.to_csv(filepath)
    logging.info(f"Data saved to {filepath}")

if __name__ == "__main__":
    TICKERS = ['TSLA', 'BND', 'SPY']
    START_DATE = '2015-01-01'
    END_DATE = '2026-01-15'
    
    raw_data = fetch_data(TICKERS, START_DATE, END_DATE)
    if raw_data is not None:
        cleaned_data = clean_data(raw_data)
        save_data(cleaned_data, "data/processed/historical_data.csv")
        
        # Display head
        print(cleaned_data.head())
