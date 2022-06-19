import math
import yfinance as yf
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas_datareader as web
import datetime as dt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# Transform values by scaling each feature to a given range

class NeuralNetwork():

    def __init__(self, company: str, start: datetime, end: datetime):
        self._data = web.DataReader(company, 'yahoo', start, end)
        self._scaler = MinMaxScaler(feature_range=(0, 1))
        self._scaled_data = self._scaler.fit_transform(self.data.values.reshape(-1, 1))

    def prepare_data(self, train_percentage=0.80):

        def _create_dataset(dataset, look_back=1):
            dataX, dataY = [], []
            for i in range(len(dataset) - look_back - 1):
                a = dataset[i:(i + look_back), 0]
                dataX.append(a)
                dataY.append(dataset[i + look_back, 0])
            return np.array(dataX), np.array(dataY)

        # Split Into Train and Test Sets
        train_size = int(len(self._scaled_data) * train_percentage)
        test_size = len(self._scaled_data) - train_size
        self._train, self._test = self._scaled_data[0:train_size, :], self._scaled_data[test_size:len(self._scaled_data), :]
        self._trainX, self._trainY = _create_dataset(self._train)
        self._testX, self._testY = _create_dataset(self._test)

    def prediction_Data_Prepare(self, window_size: int):
        x, y = [], []
        z = self._data[['Close']]
        self._scaler = MinMaxScaler(feature_range=(0, 1)).fit(z)
        z = self._scaler.transform(z)
        for i in range(window_size, len(z)):
            x.append(z[i - window_size: i])
            y.append(z[i])

        self._x, self._y = np.array(x), np.array(y)

    def forecast(self, future_day: int, model):
        # generate the multi-step forecasts
        y_future = []
        x_pred = self.x[-1:, :, :]  # last observed input sequence
        y_pred = self.y[-1]  # last observed target value

        for i in range(future_day):
            # feed the last forecast back to the model as an input
            x_pred = np.append(x_pred[:, 1:, :], y_pred.reshape(1, 1, 1), axis=1)

            # generate the next forecast
            y_pred = model.predict(x_pred)

            # save the forecast
            y_future.append(y_pred.flatten()[0])
        # transform the forecasts back to the original scale
        y_future = np.array(y_future).reshape(-1, 1)
        y_future = self._scaler.inverse_transform(y_future)
        self._y_future = y_future

    @property
    def x(self) -> np.ndarray:
        return self._x

    @property
    def y(self) -> np.ndarray:
        return self._y
    
    @property
    def y_future(self):
        return self._y_future

    @property
    def data(self):
        return self._data

    @property
    def scale(self):
        return self._scaler

# start=dt.datetime(2012, 1, 1)
# end=datetime.now().date()
# company = "SYNA"
# p =NeuralNetwork(company, start, end)
# p.prepare_data(0.80)
# p.prediction_Data_Prepare(60)
#
# ##=============================== Build the Model =======================================##
#
# model = Sequential()
#
# # model.add(LSTM(units=100,  return_sequences=True, input_shape=(1, 1)))
# model.add(LSTM(units=200, return_sequences=True, input_shape=p.x.shape[1:]))
# model.add(Dropout(0.2))
#
# model.add(LSTM(units=100, return_sequences=True))
# model.add(Dropout(0.2))
#
# model.add(LSTM(units=50))
# model.add(Dropout(0.2))
#
# model.add(Dense(units=1))  # Prediction of the next closing value
#
# model.compile(optimizer="adam", loss="mean_squared_error")
# model.fit(p.x, p.y, epochs=5, batch_size=128, verbose=2)
#
# future_sample = 150
#
# p.forecast(future_sample, model)
# ##========================== Serialize and plotting ======================================##
# # organize the results in a data frame
# df_past = p._data[['Close']].reset_index()
# df_past.rename(columns={'index': 'Date'}, inplace=True)
# df_past['Date'] = pd.to_datetime(df_past['Date'])
# df_past['Forecast'] = np.nan
#
# df_future = pd.DataFrame(columns=['Date', 'Close', 'Forecast'])
# df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=future_sample)
# df_future['Forecast'] = p.y_future.flatten()
# df_future['Close'] = np.nan
#
# results = df_past.append(df_future).set_index('Date')
#
# # plot the results
# results.plot(title=company)
