# import yfinance as yf
# import pandas as pd
# import datetime as dt
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas_datareader as web
#
# from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, LSTM
#
# company = 'FB'
# start = dt.datetime(2012,1,1)
# end = dt.datetime(2020,1,1)
#
# data = web.DataReader(company, 'yahoo', start, end)
#
# # dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
# # dataName = 'jsw_arima.csv'
# # data = yf.Ticker("TSLA").history(period="max", interval="1d",)
# # data.to_csv(f"{dataName}",index = True, encoding="utf-8", index_label = "Date")
# # data = pd.read_csv(f'{dataName}', header=0, index_col='Date', parse_dates=True, date_parser=dateparse).fillna(0)
# # data.head()
# # print(f'Number of rows with missing values: {data.isnull().any(axis=1).mean()}')
#
# #Prepare Data
# scaler = MinMaxScaler(feature_range=(0,1))
# scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
#
# prediction_days = 120
# x_train =[]
# y_train = []

# for x in range(prediction_days, len(scaled_data)):
#     x_train.append(scaled_data[x-prediction_days:x, 0])
#     y_train.append(scaled_data[x, 0])
#
# x_train, y_train = np.array(x_train), np.array(y_train)
# x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
#
# # Build the Model
# model = Sequential()
#
# model.add(LSTM(units=50,  return_sequences=True, input_shape=(x_train.shape[1], 1)))
# model.add(Dropout(0.2))
#
# model.add(LSTM(units=50,  return_sequences=True))
# model.add(Dropout(0.2))
#
#
# model.add(LSTM(units=50))
# model.add(Dropout(0.2))
#
# model.add(Dense(units=1)) # Prediction of the next closing value
#
# model.compile(optimizer="adam", loss="mean_squared_error")
# model.fit(x_train, y_train, epochs=25, batch_size=32)
#
# ''' TEST THE MODEL Accuracy on Existing Data '''
#
# # Load Test Data
# test_start = dt.datetime(2020,1,1)
# test_end = dt.datetime.now()
# test_data = web.DataReader(company, 'yahoo', test_start, test_end)
# actual_prices = test_data['Close'].values
#
# total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
#
# model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
# model_inputs = model_inputs.reshape(-1, 1)
# model_inputs = scaler.transform(model_inputs)
#
# # Make Predictons on Test Data
#
# x_test = []
#
# for x in range(prediction_days, len(model_inputs)):
#     x_test.append(model_inputs[x-prediction_days:x, 0])
#
# x_test = np.array(x_test)
# x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
#
# predicted_prices = model.predict(x_test)
# predicted_prices = scaler.inverse_transform(predicted_prices)
#
# # Plot The Test Predictions
#
# plt.plot(actual_prices, color="black", label=f"Actual {company} prices")
# plt.plot(predicted_prices, color="green", label =f"Predicted {company} prices")
# plt.title(f"{company} Share Price")
# plt.xlabel("Time")
# plt.ylabel(f"{company} Share price")
# plt.legend()
# plt.show()
#
# # Predict Next Day
#
# real_data = [model_inputs[len(model_inputs) + 2 - prediction_days: len(model_inputs+2), 0]]
# real_data = np.array(real_data)
# real_data = np.reshape(real_data, (real_data.shape[0], 1, real_data.shape[1]))
#
# prediction = model.predict(real_data)
# prediction = scaler.inverse_transform(prediction)
# print(f"Prediction: {prediction}")
#

''' SECONE METHOD'''
import datetime as dt
from datetime import datetime

import numpy as np
import pandas as pd
import pandas_datareader as web
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler


# Transform values by scaling each feature to a given range

def stock_data(company='SYNA', start=dt.datetime(2012, 1, 1), end=datetime.now().date()):
    return web.DataReader(company, 'yahoo', start, end)


# Prepare Data
def prepare_data(data, train_percentage=0.80):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

    # Split Into Train and Test Sets
    train_size = int(len(scaled_data) * train_percentage)
    test_size = len(scaled_data) - train_size
    train, test = scaled_data[0:train_size, :], scaled_data[test_size:len(scaled_data), :]
    return train, test


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


def prediction_Data_Prepare(data, window_size):
    x, y = [], []
    z = data[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(z)
    z = scaler.transform(z)
    for i in range(window_size, len(z)):
        x.append(z[i - window_size: i])
        y.append(z[i])

    return np.array(x), np.array(y)


def forecast(x, y, future_day=100):
    # generate the multi-step forecasts
    y_future = []

    x_pred = x[-1:, :, :]  # last observed input sequence
    y_pred = y[-1]  # last observed target value

    for i in range(future_day):
        # feed the last forecast back to the model as an input
        x_pred = np.append(x_pred[:, 1:, :], y_pred.reshape(1, 1, 1), axis=1)

        # generate the next forecast
        y_pred = model.predict(x_pred)

        # save the forecast
        y_future.append(y_pred.flatten()[0])
    # transform the forecasts back to the original scale
    y_future = np.array(y_future).reshape(-1, 1)
    y_future = scaler.inverse_transform(y_future)
    return y_future


# ===================================== PREPARE DATA ====================================#
company = 'SYNA'
data = stock_data(company)
train, test = prepare_data(data['Close'])

trainX, trainY = create_dataset(train)
testX, testY = create_dataset(test)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainY, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# extract the input sequences and target values
window_size = 60

x, y = prediction_Data_Prepare(data, window_size)
##=============================== Build the Model =======================================##

model = Sequential()

# model.add(LSTM(units=100,  return_sequences=True, input_shape=(1, 1)))
model.add(LSTM(units=200, return_sequences=True, input_shape=x.shape[1:]))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1))  # Prediction of the next closing value

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(x, y, epochs=100, batch_size=128, verbose=2)
##=========================== Time To Forecasting! =========================================##

future_sample = 365
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
y_future = forecast(x, y, future_sample)
# transform the forecasts back to the original scale

##========================== Serialize and plotting ======================================##
# organize the results in a data frame
df_past = data[['Close']].reset_index()
df_past.rename(columns={'index': 'Date'}, inplace=True)
df_past['Date'] = pd.to_datetime(df_past['Date'])
df_past['Forecast'] = np.nan

df_future = pd.DataFrame(columns=['Date', 'Close', 'Forecast'])
df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=future_sample)
df_future['Forecast'] = y_future.flatten()
df_future['Close'] = np.nan

results = df_past.append(df_future).set_index('Date')

# plot the results
results.plot(title=company)
