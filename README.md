# Time Series Forecasting for Portfolio Management Optimization

This project focuses on building a data-driven investment strategy for GOMY Wealth Management. By leveraging advanced time series forecasting (vividly demonstrated using ARIMA and LSTM models), the project aims to predict stock market trends and optimize asset allocation across a diversified portfolio.

## 🚀 Project Overview

The portfolio consists of three diverse assets:
*   **TSLA (Tesla):** High-growth, high-volatility stock.
*   **BND (Vanguard Total Bond Market ETF):** Stable income and low risk.
*   **SPY (S&P 500 ETF):** Market-wide diversification representing the US economy.

---

## 🛠 Project Workflow (Tasks 1-5)

### Task 1: Preprocessing and Data Cleaning
*   **Data Acquisition:** Extracted historical data for TSLA, BND, and SPY from YFinance (2015 - present).
*   **Cleaning:** Addressed missing values and ensured correct data types for time-series analysis.
*   **Exploration:** Initial analysis of price trends and normalization of data for model compatibility.

### Task 2: Quantitative Analysis
*   **Exploratory Data Analysis (EDA):** Visualized rolling averages and standard deviations to understand volatility.
*   **Decomposition:** Analyzed Trend, Seasonality, and Residuals using seasonal decomposition.
*   **Stationarity:** Performed Augmented Dickey-Fuller (ADF) tests to confirm the requirements for ARIMA.

### Task 3: Time Series Forecasting
*   **ARIMA Model:** Built a statistical baseline model. Optimized (p, d, q) parameters using `auto_arima` to minimize AIC.
*   **LSTM Model:** Implemented a Deep Learning (Long Short-Term Memory) network to capture non-linear market dependencies and long-term patterns.
*   **Evaluation:** Validated models using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE).

### Task 4: Portfolio Optimization
*   **Return Forecasting:** Used the model forecasts to estimate future returns for the next trading year.
*   **Modern Portfolio Theory (MPT):** Constructed the Efficient Frontier to calculate optimal weights.
*   **Optimization Results:** 
    *   **Max Sharpe Ratio:** Optimized for the best risk-adjusted returns.
    *   **Min Volatility:** Optimized for the lowest possible risk.

### Task 5: Interactive Dashboard Development
*   **Streamlit App:** Built a high-fidelity interactive dashboard for real-time portfolio management.
*   **Features:**
    *   Interactive price and return charts (Plotly).
    *   Customizable forecast horizons.
    *   Live portfolio weight calculation and allocation pie charts.

---

## 💻 Installation and Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/ruthasoll/Time-series-forecasting-for-portfolio-management-optimization.git
    cd Time-series-forecasting-for-portfolio-management-optimization
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Dashboard:**
    ```bash
    streamlit run scripts/dashboard.py
    ```

---

## 📂 Folder Structure

```text
├── data/
│   ├── raw/                # Original data from yfinance
│   └── processed/          # Cleaned and engineered data
├── notebooks/
│   ├── EDA.ipynb           # Exploratory analysis
│   ├── Forecast_Analysis.ipynb # Model training & evaluation (ARIMA/LSTM)
│   └── Portfolio_Optimization.ipynb # Risk management and MVO
├── scripts/
│   └── dashboard.py        # Streamlit application
├── src/
│   ├── data_loader.py      # Data fetching utilities
│   ├── models.py           # ML/Stat model implementations
│   ├── utils.py            # Helper functions for plotting & stats
│   └── portfolio_optimizer.py # Optimization logic
└── requirements.txt        # Project dependencies
```

## 🛡 License
This project is licensed under the MIT License - see the LICENSE file for details.