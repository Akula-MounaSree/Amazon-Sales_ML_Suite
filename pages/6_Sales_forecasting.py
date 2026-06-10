import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📈 Sales Forecasting (Simple Version)")

# Load Data
df = pd.read_csv("Amazon Sale Report.csv")

# Convert columns
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

df = df.dropna(subset=['Date', 'Amount'])

# Monthly Sales
monthly_sales = (
    df.groupby(df['Date'].dt.to_period('M'))['Amount']
    .sum()
    .reset_index()
)

monthly_sales['Date'] = monthly_sales['Date'].astype(str)

st.subheader("Monthly Sales Trend")

fig = px.line(
    monthly_sales,
    x='Date',
    y='Amount',
    markers=True
)

st.plotly_chart(fig, use_container_width=True)

# Simple Forecast
last_month_sales = monthly_sales['Amount'].iloc[-1]
avg_sales = monthly_sales['Amount'].mean()

forecast_sales = (last_month_sales + avg_sales) / 2

st.subheader("Next Month Forecast")

st.metric(
    "Predicted Next Month Sales",
    f"₹ {forecast_sales:,.0f}"
)