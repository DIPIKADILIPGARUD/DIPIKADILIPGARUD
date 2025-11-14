import pandas as pd
import yfinance as yf
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np

def get_data(ticker):
    """
    Fetch historical stock data from 2020-01-01 to today using yfinance.
    Returns DataFrame with OHLCV columns.
    """
    start = "2020-01-01"
    end = datetime.now().strftime("%Y-%m-%d")
    data = yf.download(ticker, start=start, end=end, progress=False)
    return data

def get_rolling_mean(close_prices):
    """
    Apply 12-period rolling mean to smooth the closing prices.
    Used as preprocessing for stationarity checks.
    """
    return close_prices.rolling(window=12).mean()

def stationary_check(close_price):
    """
    Perform Augmented Dickey-Fuller (ADF) test for stationarity.
    Returns p-value; <0.05 indicates stationary series.
    """
    result = adfuller(close_price.dropna())
    return result[1]  # p-value

def get_differencing_order(close_price):
    """
    Iteratively difference the series until it's stationary (p-value < 0.05).
    Returns the order 'd' for ARIMA.
    """
    d = 0
    while True:
        p_value = stationary_check(close_price)
        if p_value < 0.05:
            return d
        close_price = close_price.diff().dropna()
        d += 1
        if d > 10:  # Safety break to prevent infinite loop
            return d

def scale_data(data):
    """
    Scale the data using StandardScaler for ARIMA stability.
    Returns scaled Series and fitted scaler (for inverse transform later).
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    return pd.Series(scaled_data.flatten(), index=data.index), scaler

def fit_model(data, d_order, steps=30):
    """
    Fit ARIMA model (p=3, d=auto, q=3) and forecast given steps.
    Returns the forecast Series.
    """
    model = ARIMA(data, order=(3, d_order, 3))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

def evaluate_model(scaled_data, d_order):
    """
    Evaluate ARIMA on 80/20 train/test split.
    Returns RMSE score (lower is better).
    """
    split = int(0.8 * len(scaled_data))
    train, test = scaled_data[:split], scaled_data[split:]
    
    # Forecast same number of steps as test length
    forecast = fit_model(train, d_order, steps=len(test))
    forecast = forecast[:len(test)]
    
    rmse = sqrt(mean_squared_error(test, forecast))
    return rmse


import plotly.graph_objects as go

def moving_average_forecast(actual, forecast_df):
    # Drop NaN from rolling mean
    actual = actual.dropna()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actual.index,
        y=actual.values,
        mode='lines',
        name='Actual Rolling Close',
        line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Close'],
        mode='lines',
        name='Forecast',
        line=dict(color='red', dash='dash', width=2)
    ))

    fig.update_layout(
        title='Stock Price Forecast',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white',
        hovermode='x unified',
        height=600,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),  # ✅ enables horizontal scroll
            type="date"
        )
    )
    return fig

def get_forecast(scaled_data, d_order, scaler):
    """
    Generate 30-day forecast DataFrame with dates and inverse-scaled prices.
    Returns DF with 'Date' and 'Close' columns.
    """
    forecast = fit_model(scaled_data, d_order, steps=30)
    start_date = datetime.now()
    forecast_index = pd.date_range(start=start_date, periods=30, freq='D')
    forecast_values = scaler.inverse_transform(forecast.values.reshape(-1, 1)).flatten()
    return pd.DataFrame({'Date': forecast_index, 'Close': forecast_values})
