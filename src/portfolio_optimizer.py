import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import logging

class PortfolioOptimizer:
    def __init__(self, data):
        """
        Initializes the optimizer with historical price data.
        
        Args:
            data (pd.DataFrame): Time series of asset closing prices.
        """
        self.data = data
        self.mu = None
        self.S = None
        self.weights = None

    def calculate_metrics(self, mu_override=None):
        """
        Calculates expected returns and the covariance matrix.
        
        Args:
            mu_override (pd.Series, optional): Custom expected returns to override historical means.
        """
        logging.info("Calculating expected returns and covariance matrix...")
        
        if mu_override is not None:
            self.mu = mu_override
        else:
            # Calculate expected returns using mean historical return
            self.mu = expected_returns.mean_historical_return(self.data)
        
        # Calculate sample covariance matrix
        # You can use different risk models like Ledoit-Wolf shrinkage
        self.S = risk_models.CovarianceShrinkage(self.data).ledoit_wolf()
        
    def optimize_portfolio(self):
        """
        Optimizes for the maximal Sharpe Ratio.
        
        Returns:
            dict: Optimized weights for each asset.
        """
        if self.mu is None or self.S is None:
            self.calculate_metrics()
            
        logging.info("Optimizing portfolio for Maximal Sharpe Ratio...")
        ef = EfficientFrontier(self.mu, self.S)
        
        # Maximize Sharpe Ratio
        raw_weights = ef.max_sharpe()
        self.weights = ef.clean_weights()
        
        # Log performance
        performance = ef.portfolio_performance(verbose=True)
        logging.info(f"Max Sharpe Ratio: Returns {performance[0]:.2%}, Volatility {performance[1]:.2%}, Sharpe {performance[2]:.2f}")
        
        return self.weights

    def optimize_min_volatility(self):
        """
        Optimizes for the minimum volatility.
        
        Returns:
            dict: Optimized weights for each asset.
        """
        if self.mu is None or self.S is None:
            self.calculate_metrics()
            
        logging.info("Optimizing portfolio for Minimum Volatility...")
        ef = EfficientFrontier(self.mu, self.S)
        
        # Minimize Volatility
        raw_weights = ef.min_volatility()
        self.weights = ef.clean_weights()
        
        # Log performance
        performance = ef.portfolio_performance(verbose=True)
        logging.info(f"Min Volatility: Returns {performance[0]:.2%}, Volatility {performance[1]:.2%}, Sharpe {performance[2]:.2f}")
        
        return self.weights

if __name__ == "__main__":
    # Test run
    # Mock data generation for testing
    dates = pd.date_range(start='2020-01-01', periods=100)
    data = pd.DataFrame(np.random.normal(100, 5, size=(100, 3)), index=dates, columns=['TSLA', 'BND', 'SPY'])
    
    optimizer = PortfolioOptimizer(data)
    weights = optimizer.optimize_portfolio()
    print("Optimized Weights:", weights)
