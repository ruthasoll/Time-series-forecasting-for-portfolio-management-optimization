# Time Series Forecasting for Portfolio Management Optimization

This project aims to apply time series forecasting to historical financial data (Tesla, Vanguard Total Bond Market ETF, S&P 500 ETF) to optimize portfolio management strategies.

## Structure
- `data/`: Data storage (raw/processed)
- `notebooks/`: Jupyter notebooks for analysis and modeling
- `src/`: Source code for data fetching, processing, and modeling
- `scripts/`: Utility scripts
- `tests/`: Unit tests

## Setup
1. Install dependencies: `pip install -r requirements.txt`

## Features

- **Time Series Forecasting**: Implements both LSTM and ARIMA models.
- **Data Processing**: Functions for loading, cleaning, and preparing data for analysis.
- **Evaluation Metrics**: Calculates metrics such as MAE, RMSE, and MAPE to evaluate model performance.

## Getting Started

### Prerequisites

Make sure you have [Python](https://www.python.org/) installed on your machine. It is also recommended to use a virtual environment for this project.

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/your-repository.git
   cd your-repository
2. Create a virtual environment (optional but recommended):

    bash
    python -m venv .venv
3. Activate the virtual environment:

    On Windows:

    bash
    .venv\Scripts\activate
4. Install the required packages:

    bash
    pip install -r requirements.txt
5. Running the Project
    To run the project and execute the forecasting models, use the following command:

    bash
    python src/models.py