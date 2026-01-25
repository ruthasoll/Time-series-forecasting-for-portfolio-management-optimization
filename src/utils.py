import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

def check_stationarity(series, name="Series"):
    """
    Performs Augmented Dickey-Fuller test on a time series.
    """
    result = adfuller(series.dropna())
    print(f"ADF Test for {name}:")
    print(f"  ADF Statistic: {result[0]:.6f}")
    print(f"  p-value: {result[1]:.6f}")
    print(f"  Critical Values:")
    for key, value in result[4].items():
        print(f"    {key}: {value:.6f}")
    
    if result[1] < 0.05:
        print(f"Result: {name} is STATIONARY (p < 0.05)")
    else:
        print(f"Result: {name} is NON-STATIONARY (p >= 0.05)")
    print("-" * 30)
    return result[1]

def calculate_daily_returns(data):
    """
    Calculates daily percentage returns.
    """
    return data.pct_change().dropna()

def plot_price_series(data, title="Price Series"):
    """
    Plots the price history of the assets.
    """
    plt.figure(figsize=(14, 7))
    for column in data.columns:
        plt.plot(data.index, data[column], label=column)
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_risk_metrics(returns, risk_free_rate=0.0):
    """
    Calculates VaR and Sharpe Ratio.
    """
    metrics = {}
    
    for col in returns.columns:
        # Value at Risk (VaR) at 95% confidence level
        var_95 = np.percentile(returns[col], 5)
        
        # Sharpe Ratio (assuming daily returns, annualized by sqrt(252))
        avg_return = returns[col].mean()
        std_dev = returns[col].std()
        sharpe_ratio = (avg_return - risk_free_rate) / std_dev * np.sqrt(252)
        
        metrics[col] = {
            "VaR_95": var_95,
            "Sharpe_Ratio": sharpe_ratio,
            "Mean_Daily_Return": avg_return,
            "Daily_Volatility": std_dev
        }
    
    return pd.DataFrame(metrics).T
