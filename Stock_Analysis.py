import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, date
from pages.utils.plotly_figure import plotly_table  # Import custom table function
import pandas_ta as ta  # For technical indicators

# Page config
st.set_page_config(page_title="Stock Analysis", layout="wide", page_icon="📈")

# Page title and user inputs
st.title("Stock Analysis")

today = date.today()
col1, col2, col3 = st.columns(3)
ticker = col1.text_input("Stock Ticker", value="TSLA").upper()
start_date = col2.date_input("Choose Start Date", value=datetime(today.year - 1, today.month, today.day))
end_date = col3.date_input("Choose End Date", value=today)

# Fetch and display stock info if ticker provided
if ticker:
    stock = yf.Ticker(ticker)
    st.subheader(ticker)  # Display ticker as subheader
    
    # Basic company summary
    st.write(stock.info.get('longBusinessSummary', 'No summary available'))
    
    # Key details
    st.write("**Sector:**", stock.info.get('sector', 'N/A'))
    st.write("**Full Time Employees:**", stock.info.get('fullTimeEmployees', 'N/A'))
    st.write("**Website:**", stock.info.get('website', 'N/A'))
    
    # Metrics tables using plotly_table
    col1, col2 = st.columns(2)
    with col1:
        # First metrics table: Market Cap, Beta, etc.
        df1 = pd.DataFrame({
            'Metrics': ['Market Cap', 'Beta', 'Trailing EPS', 'Trailing PE Ratio'],
            'Value': [
                stock.info.get('marketCap'), 
                stock.info.get('beta'), 
                stock.info.get('trailingEps'), 
                stock.info.get('trailingPE')
            ]
        })
        fig1 = plotly_table(df1)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Second metrics table: Quick Ratio, etc.
        df2 = pd.DataFrame({
            'Metrics': ['Quick Ratio', 'Revenue Per Share', 'Profit Margin', 'Debt to Equity', 'Return on Equity'],
            'Value': [
                stock.info.get('quickRatio'), 
                stock.info.get('revenuePerShare'), 
                stock.info.get('profitMargins'), 
                stock.info.get('debtToEquity'), 
                stock.info.get('returnOnEquity')
            ]
        })
        fig2 = plotly_table(df2)
        st.plotly_chart(fig2, use_container_width=True)

# Download historical data
data = yf.download(ticker, start=start_date, end=end_date, progress=False)
if not data.empty:
    # ------------------- FIXED Daily Change Metric -------------------
    col1, col2, col3 = st.columns(3)

    # Safely extract last two closing prices
    close_series = data['Close']
    last_close = float(close_series.iloc[-1]) if not close_series.empty else 0.0
    prev_close = float(close_series.iloc[-2]) if len(close_series) > 1 else last_close

    # Calculate daily change
    daily_change = ((last_close - prev_close) / prev_close * 100) if prev_close != 0 else 0.0

    # Display metric
    col1.metric("Daily Change", f"{daily_change:.2f}%", delta=f"{last_close:.2f}")
    # -------------------------------------------------------------------

    # Last 10 days historical table
    last_10_df = data.tail(10).sort_index(ascending=False).round(2)
    st.subheader("Historical Data (Last 10 Days)")
    st.dataframe(last_10_df)

    # Time period selection buttons
    period = None
    cols = st.columns(4)
    periods = ["5D", "1M", "6M", "YTD", "1Y", "5Y", "MAX"]
    for i, p in enumerate(periods):
        if cols[i % 4].button(p):
            period = p

    # Chart options
    col1, col2, col3 = st.columns(3)
    chart_type = col1.selectbox("Chart Type", ["Candlestick", "Line"])
    if chart_type == "Candlestick":
        indicator = col2.selectbox("Indicator", ["RSI", "MACD"])
    else:
        indicator = col2.selectbox("Indicator", ["RSI", "Moving Average", "MACD"])

    # Function to filter data by selected period
    def filter_data(data, period):
        if period == "5D":
            start = data.index[-1] - pd.DateOffset(days=5)
        elif period == "1M":
            start = data.index[-1] - pd.DateOffset(months=1)
        elif period == "6M":
            start = data.index[-1] - pd.DateOffset(months=6)
        elif period == "YTD":
            start = data.index[0].replace(month=1, day=1)
        elif period == "1Y":
            start = data.index[-1] - pd.DateOffset(years=1)
        elif period == "5Y":
            start = data.index[-1] - pd.DateOffset(years=5)
        elif period == "MAX":
            return data
        else:
            return data
        return data.loc[start:]

    filtered_data = filter_data(data, period) if period else data

    # Chart rendering functions (for candlestick/line with indicators)
    def candlestick_rsi(data):
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Candlestick"))
        rsi = ta.rsi(data['Close'])
        fig.add_trace(go.Scatter(x=data.index, y=rsi, name="RSI", yaxis="y2"))
        fig.add_hline(y=70, line_dash="dash", annotation_text="Overbought", yref="y2")
        fig.add_hline(y=30, line_dash="dash", annotation_text="Oversold", yref="y2")
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 100]))
        return fig

    def candlestick_macd(data):
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Candlestick"))
        macd = ta.macd(data['Close'])
        fig.add_trace(go.Scatter(x=data.index, y=macd['MACD_12_26_9'], name="MACD", yaxis="y2"))
        fig.add_trace(go.Scatter(x=data.index, y=macd['MACDs_12_26_9'], name="Signal", yaxis="y2"))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right"))
        return fig

    def line_rsi(data):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Close Price"))
        rsi = ta.rsi(data['Close'])
        fig.add_trace(go.Scatter(x=data.index, y=rsi, name="RSI", yaxis="y2"))
        fig.add_hline(y=70, line_dash="dash", annotation_text="Overbought", yref="y2")
        fig.add_hline(y=30, line_dash="dash", annotation_text="Oversold", yref="y2")
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 100]))
        return fig

    def line_moving_average(data):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Close Price"))
        ma = data['Close'].rolling(window=50).mean()
        fig.add_trace(go.Scatter(x=data.index, y=ma, name="50-Day MA"))
        return fig

    def line_macd(data):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Close Price"))
        macd = ta.macd(data['Close'])
        fig.add_trace(go.Scatter(x=data.index, y=macd['MACD_12_26_9'], name="MACD", yaxis="y2"))
        fig.add_trace(go.Scatter(x=data.index, y=macd['MACDs_12_26_9'], name="Signal", yaxis="y2"))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right"))
        return fig

    # Render the selected chart
    if chart_type == "Candlestick":
        if indicator == "RSI":
            fig = candlestick_rsi(filtered_data)
        else:  # MACD
            fig = candlestick_macd(filtered_data)
    else:  # Line chart
        if indicator == "RSI":
            fig = line_rsi(filtered_data)
        elif indicator == "Moving Average":
            fig = line_moving_average(filtered_data)
        else:  # MACD
            fig = line_macd(filtered_data)

    st.plotly_chart(fig, use_container_width=True)
