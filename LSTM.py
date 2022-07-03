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
    def __init__(self, df):
        # ,company: str, start: datetime, end: datetime):
        # self._data = web.DataReader(company, 'yahoo', start, end)
        self._data = df
        self._scaler = MinMaxScaler(feature_range=(0, 1))
        self._scaled_data = self._scaler.fit_transform(self._data.values.reshape(-1, 1))

    def prepare_data(self, future_day):

        def _create_dataset(dataset, look_back=1):
            dataX, dataY = [], []
            for i in range(len(dataset) - look_back - 1):
                a = dataset[i:(i + look_back), 0]
                dataX.append(a)
                dataY.append(dataset[i + look_back, 0])
            return np.array(dataX), np.array(dataY)

        # Split Into Train and Test Sets
        train_size = int(len(self._scaled_data) - future_day)
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

    @property
    def train(self):
        return self._train

    @property
    def test(self):
        return self._test
