import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models import ARIMAModel
from portfolio_optimizer import PortfolioOptimizer

# Set page config
st.set_page_config(page_title="Portfolio Optimization Dashboard", layout="wide")

st.title("📈 Time Series Forecasting & Portfolio Optimization")
st.markdown("""
This dashboard provides insights into historical asset performance, future price forecasts, and optimized portfolio allocations.
""")

# 1. Load Data
@st.cache_data
def load_data():
    data_path = os.path.join('data', 'processed', 'historical_data.csv')
    if not os.path.exists(data_path):
        st.error(f"Data not found at {data_path}. Please run data_loader.py first.")
        return None
    df = pd.read_csv(data_path, index_col=0, parse_dates=True, header=[0, 1])
    return df

data = load_data()

if data is not None:
    tickers = ['TSLA', 'BND', 'SPY']
    close_prices = pd.DataFrame()
    for ticker in tickers:
        close_prices[ticker] = data[ticker]['Close']

    # Sidebar
    st.sidebar.header("Settings")
    selected_ticker = st.sidebar.selectbox("Select Ticker for Forecasting", tickers)
    forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 30, 365, 180)

    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Market Overview", "Price Forecasting", "Portfolio Optimization"])

    with tab1:
        st.subheader("Historical Price Trends")
        fig_hist = px.line(close_prices, labels={"value": "Price (USD)", "Date": "Date"})
        st.plotly_chart(fig_hist, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.write("Daily Returns Distribution")
            returns = close_prices.pct_change().dropna()
            fig_dist = px.histogram(returns, barmode='overlay')
            st.plotly_chart(fig_dist, use_container_width=True)
        with col2:
            st.write("Correlation Matrix")
            corr = returns.corr()
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
            st.plotly_chart(fig_corr, use_container_width=True)

    with tab2:
        st.subheader(f"Forecasting {selected_ticker}")
        st.write(f"Generating ARIMA forecast for the next {forecast_days} days...")
        
        with st.spinner("Model is thinking..."):
            model = ARIMAModel()
            # Use last year of data for quick demo, or full data
            train_data = close_prices[selected_ticker].dropna()
            model.optimize_and_fit(train_data)
            forecast = model.predict(n_periods=forecast_days)
            
            # Generate dates
            last_date = train_data.index[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
            forecast_series = pd.Series(forecast.values if hasattr(forecast, 'values') else forecast, index=forecast_dates)

            # Plot
            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(x=train_data.index[-252:], y=train_data.values[-252:], name="Historical (Last 1Y)"))
            fig_forecast.add_trace(go.Scatter(x=forecast_dates, y=forecast_series, name="ARIMA Forecast", line=dict(dash='dash', color='red')))
            fig_forecast.update_layout(title=f"{selected_ticker} Price Forecast", xaxis_title="Date", yaxis_title="Price (USD)")
            st.plotly_chart(fig_forecast, use_container_width=True)

    with tab3:
        st.subheader("Portfolio Allocation Results")
        
        optimizer = PortfolioOptimizer(close_prices)
        
        # Max Sharpe Optimization
        st.write("### Strategy: Maximum Sharpe Ratio")
        try:
            weights_sharpe = optimizer.optimize_portfolio()
            perf_sharpe = optimizer.get_performance()
            
            w_col1, w_col2 = st.columns([1, 2])
            with w_col1:
                st.write("**Optimal Weights:**")
                st.json(dict(weights_sharpe))
                st.metric("Expected Annual Return", f"{perf_sharpe[0]:.2%}")
                st.metric("Annual Volatility", f"{perf_sharpe[1]:.2%}")
                st.metric("Sharpe Ratio", f"{perf_sharpe[2]:.2f}")
            
            with w_col2:
                labels = list(weights_sharpe.keys())
                values = list(weights_sharpe.values())
                fig_pie = px.pie(names=labels, values=values, title="Allocation Breakdown")
                st.plotly_chart(fig_pie, use_container_width=True)
        except Exception as e:
            st.error(f"Optimization error: {e}")

        st.divider()
        
        # Min Volatility Optimization
        st.write("### Strategy: Minimum Volatility")
        try:
            weights_vol = optimizer.optimize_min_volatility()
            perf_vol = optimizer.get_performance()
            
            w_col3, w_col4 = st.columns([1, 2])
            with w_col3:
                st.write("**Safe Weights:**")
                st.json(dict(weights_vol))
                st.metric("Expected Annual Return", f"{perf_vol[0]:.2%}")
                st.metric("Annual Volatility", f"{perf_vol[1]:.2%}")
                st.metric("Sharpe Ratio", f"{perf_vol[2]:.2f}")
            
            with w_col4:
                labels_vol = list(weights_vol.keys())
                values_vol = list(weights_vol.values())
                fig_pie_vol = px.pie(names=labels_vol, values=values_vol, title="Low-Risk Breakdown")
                st.plotly_chart(fig_pie_vol, use_container_width=True)
        except Exception as e:
            st.error(f"Optimization error: {e}")
