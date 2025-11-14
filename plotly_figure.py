import pandas as pd
import yfinance as yf
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import plotly.graph_objects as go

# -----------------------------
# Data fetching & preprocessing
# -----------------------------
def get_data(ticker):
    start = "2020-01-01"
    end = datetime.now().strftime("%Y-%m-%d")
    data = yf.download(ticker, start=start, end=end, progress=False)
    return data

def get_rolling_mean(close_prices):
    return close_prices.rolling(window=12).mean()

def stationary_check(close_price):
    result = adfuller(close_price.dropna())
    return result[1]

def get_differencing_order(close_price):
    d = 0
    while True:
        p_value = stationary_check(close_price)
        if p_value < 0.05:
            return d
        close_price = close_price.diff().dropna()
        d += 1
        if d > 10:
            return d

def scale_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    return pd.Series(scaled_data.flatten(), index=data.index), scaler

def fit_model(data, d_order, steps=30):
    model = ARIMA(data, order=(3, d_order, 3))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

def evaluate_model(scaled_data, d_order):
    split = int(0.8 * len(scaled_data))
    train, test = scaled_data[:split], scaled_data[split:]
    forecast = fit_model(train, d_order, steps=len(test))
    forecast = forecast[:len(test)]
    rmse = sqrt(mean_squared_error(test, forecast))
    return rmse

# -----------------------------
# Forecast generation
# -----------------------------
def get_forecast(scaled_data, d_order, scaler, steps=30):
    forecast = fit_model(scaled_data, d_order, steps=steps)
    start_date = datetime.now()
    forecast_index = pd.date_range(start=start_date, periods=steps, freq='D')
    forecast_values = scaler.inverse_transform(forecast.values.reshape(-1, 1)).flatten()
    return pd.DataFrame({'Date': forecast_index, 'Close': forecast_values})

# -----------------------------
# Plotting: Actual vs Forecast
# -----------------------------
def moving_average_forecast(actual, forecast_df):
    """
    Plot Actual vs Forecast with proper NaN handling and alignment.
    """
    # Agar DataFrame hai
    if isinstance(actual, pd.DataFrame):
        # Try 'Close' first, else 'Adj Close', else first numeric column
        if 'Close' in actual.columns:
            actual = actual['Close']
        elif 'Adj Close' in actual.columns:
            actual = actual['Adj Close']
        else:
            # Automatically pick first numeric column
            numeric_cols = actual.select_dtypes(include=np.number).columns
            if len(numeric_cols) == 0:
                raise ValueError("DataFrame passed to actual has no numeric column")
            actual = actual[numeric_cols[0]]

    # Ensure 1D numeric series
    actual = pd.Series(actual).astype(float).dropna()
    
    # Forecast close values
    forecast_df['Close'] = pd.to_numeric(forecast_df['Close'], errors='coerce')

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
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    return fig



# -----------------------------
# Plotly Table
# -----------------------------
def plotly_table(df):
    """
    Create a Plotly Table from a DataFrame.
    """
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[df[col] for col in df.columns],
                   fill_color='lavender',
                   align='left'))
    ])
    return fig
