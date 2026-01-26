import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
except Exception:
    # Fallback to standalone keras if tensorflow.keras is not available
    from keras.models import Sequential
    from keras.layers import LSTM, Dense
import logging

def split_data(data, test_size=0.2):
    """
    Splits data into training and testing sets chronologically.
    """
    split_idx = int(len(data) * (1 - test_size))
    train, test = data[:split_idx], data[split_idx:]
    return train, test

def evaluate_forecast(y_true, y_pred, model_name):
    """
    Calculates MAE, RMSE, MAPE.
    """
    # Convert inputs to pandas Series for safe alignment and NaN handling
    y_true_s = pd.Series(y_true).astype(float)
    y_pred_s = pd.Series(y_pred).astype(float)

    # Align on index if possible (keeps intersection); otherwise align by position
    try:
        y_true_s, y_pred_s = y_true_s.align(y_pred_s, join='inner')
    except Exception:
        # Fallback: truncate to same length by position
        min_len = min(len(y_true_s), len(y_pred_s))
        y_true_s = y_true_s.iloc[:min_len]
        y_pred_s = y_pred_s.iloc[:min_len]

    # Drop any remaining NaNs
    mask = y_true_s.notna() & y_pred_s.notna()
    if mask.sum() == 0:
        raise ValueError("No overlapping non-NaN values available to evaluate.")
    if mask.sum() < len(y_true_s):
        logging.warning(f"evaluate_forecast: Dropped {len(y_true_s) - mask.sum()} rows due to NaNs.")

    y_true_clean = y_true_s[mask].values
    y_pred_clean = y_pred_s[mask].values

    mae = float(mean_absolute_error(y_true_clean, y_pred_clean))
    rmse = float(np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)))

    # Compute MAPE safely (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = np.where(y_true_clean == 0, np.nan, y_true_clean)
        mape_vals = np.abs((y_true_clean - y_pred_clean) / denom)
        mape = float(np.nanmean(mape_vals) * 100)

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }

class ARIMAModel:
    def __init__(self, order=None, seasonal=False, m=1):
        self.order = order
        self.seasonal = seasonal
        self.m = m
        self.model = None
        self.model_fit = None

    def optimize_and_fit(self, train_data):
        """
        Uses auto_arima to find best parameters and fit.
        """
        logging.info("Optimizing ARIMA parameters...")
        self.model = pm.auto_arima(train_data, seasonal=self.seasonal, m=self.m, 
                                   stepwise=True, suppress_warnings=True, 
                                   error_action='ignore', trace=True)
        logging.info(f"Best ARIMA order: {self.model.order}")
        self.model_fit = self.model
        return self.model.order

    def fit(self, train_data, order):
        """
        Fits ARIMA with specific order.
        """
        self.model = ARIMA(train_data, order=order)
        self.model_fit = self.model.fit()
        return self.model_fit

    def predict(self, n_periods):
        if self.model_fit is None:
            raise ValueError("Model not fitted yet.")
        
        # pmdarima predict
        if hasattr(self.model_fit, 'predict'):
            return self.model_fit.predict(n_periods=n_periods)
        # statsmodels forecast
        else:
            return self.model_fit.forecast(steps=n_periods)

class LSTMModel:
    def __init__(self, look_back=60, epochs=20, batch_size=32, units=50):
        self.look_back = look_back
        self.epochs = epochs
        self.batch_size = batch_size
        self.units = units
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def create_dataset(self, dataset):
        X, Y = [], []
        for i in range(len(dataset) - self.look_back):
            a = dataset[i:(i + self.look_back), 0]
            X.append(a)
            Y.append(dataset[i + self.look_back, 0])
        return np.array(X), np.array(Y)

    def fit(self, train_data):
        # Scale data
        train_data = np.array(train_data).reshape(-1, 1)
        self.scaler.fit(train_data)
        scaled_train = self.scaler.transform(train_data)
        
        X_train, y_train = self.create_dataset(scaled_train)
        
        # Reshape input to be [samples, time steps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
        # Build LSTM
        self.model = Sequential()
        self.model.add(LSTM(self.units, return_sequences=True, input_shape=(self.look_back, 1)))
        self.model.add(LSTM(self.units, return_sequences=False))
        self.model.add(Dense(25))
        self.model.add(Dense(1))
        
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs)

    def predict(self, data):
        # Prepare input
        # Note: logic requires careful handling of train/test boundary
        # If 'data' is the test set, we need the last 'look_back' points from train
        # Caller needs to provide context
        
        data = np.array(data).reshape(-1, 1)
        scaled_data = self.scaler.transform(data)
        X_test, _ = self.create_dataset(scaled_data) # This might shrink size
        
        # Assuming we pass full sequence (last train + test) to get test predictions
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        predictions_scaled = self.model.predict(X_test)
        predictions = self.scaler.inverse_transform(predictions_scaled)
        
        return predictions.flatten()


