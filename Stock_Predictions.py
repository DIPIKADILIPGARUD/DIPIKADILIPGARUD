import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime

# ✅ Corrected imports
from pages.utils.model_train import (
    get_data,
    get_rolling_mean,
    get_differencing_order,
    scale_data,
    evaluate_model,
    get_forecast,
)

# ✅ Changed to singular file name
from pages.utils.plotly_figure import plotly_table, moving_average_forecast

# Page config
st.set_page_config(page_title="Stock Prediction", layout="wide", page_icon="🔮")
st.title("Stock Prediction")

# Input
col1, col2 = st.columns(2)
ticker = col1.text_input("Stock Ticker", value="AAPL").upper()

if ticker:
    st.subheader(f"Predicting Next 30 Days Close Price for {ticker}")

    # Get and prep data
    data = get_data(ticker)

    if not data.empty:
        rolling_price = get_rolling_mean(data['Close'])
        d_order = get_differencing_order(rolling_price)
        scaled_data, scaler = scale_data(data['Close'])

        # Evaluate model
        rmse_score = evaluate_model(scaled_data, d_order)
        st.write(f"**Model RMSE Score:** {rmse_score:.4f}")

        # Forecast
        forecast_df = get_forecast(scaled_data, d_order, scaler)
        st.subheader("📊 30-Day Forecast Table")
        st.plotly_chart(plotly_table(forecast_df), use_container_width=True)

        # Actual vs Predicted
        st.subheader("📈 Actual vs Predicted Prices")
        fig = moving_average_forecast(rolling_price, forecast_df)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for the given ticker symbol.")
else:
    st.info("Please enter a valid stock ticker symbol to start.")
