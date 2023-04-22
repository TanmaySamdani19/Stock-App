import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data

from keras.models import load_model
import streamlit as st

start = '2010-01-01'
end = '2022-12-31'

st.title('Stock Trend Predictions')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = data.DataReader(user_input, 'yahoo', start, end)

st.subheader('Data from 2010-2022')
st.write(df.describe())

#Visualising Data
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

df_train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
df_test = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

df_train_arr = scaler.fit_transform(df_train)

x_train=[]
y_train=[]

for i in range(100, df_train_arr.shape[0]):
    x_train.append(df_train_arr[i - 100:i])
    y_train.append(df_train_arr[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

model = load_model('keras_model.h5')

past_100_days = df_train.tail(100)
final_df = past_100_days.append(df_test, ignore_index=True)
input_data = scaler.fit_transform(final_df)

X_test = []
y_test = []

for i in range(100,input_data.shape[0]):
    X_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

X_test, y_test = np.array(X_test), np.array(y_test)
y_pred = model.predict(X_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_pred = y_pred*scale_factor
y_test = y_test*scale_factor

st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_pred, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)