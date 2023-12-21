#main.py

#Stock Predictor App

import streamlit as st
from datetime import date

import pandas as pd

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objects as go

START = " 2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("STOCK PREDICTON APP")

stocks= ("RELIANCE.NS", "ASIANPAINT.NS", "SBIN.NS","ITC.NS" )
selected_stock = st.selectbox("Select dataset for prediction",stocks)

n_years = st.slider("Years of Prediction", 1, 4)
period = n_years * 365

def load_data(ticker):
    #data = yf.download(ticker, START, TODAY)
    data_obj = yf.Ticker(ticker)
    data = data_obj.history()
    data = data.reset_index()  
    data.head()

    #data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load Data ... ")
data = load_data(selected_stock)
data_load_state.text("Loading Data ... DONE ")

data = pd.DataFrame(data)

st.subheader('Raw Data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    print(data['Open'])
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'],name ="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'],name ="stock_close"))
    fig.layout.update(title="Time Series data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

ds_column = data['Date']
ds_column = ds_column.dt.tz_localize(None)
data['Date'] = ds_column

#Forecasting

df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close":'y'})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader("Forecast Data")
st.write(data.tail())

st.write("Forecast Data")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast Components")
fig2 = m.plot_components(forecast)
st.write("fig2")
st.plotly_chart(fig2)  
