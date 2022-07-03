import datetime as dt
import math
from datetime import datetime
from pylab import rcParams
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as web
import yfinance as yf
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

class ARIMA_Model():

    # def __init__(self, company: str, start: datetime, end: datetime):
    def __init__(self, df_values):
        # self._data = web.DataReader(company, 'yahoo', start, end)
        # self._df_close = self._data['Close']
        self._df_close = df_values
    def test_stationarity(self):
        # Determing rolling statistics
        rolmean = self._df_close.rolling(12).mean()
        rolstd = self._df_close.rolling(12).std()

        # plot rolling statistics:
        plt.plot(self._df_close, color='blue', label='Original')
        plt.plot(rolmean, color='red', label='Rolling Mean')
        plt.plot(rolstd, color='black', label='Rolling STD')
        plt.legend(loc='best')
        plt.title('Rolling mean and standard deviation')
        plt.show(block=True)
        adft = adfuller(self._df_close, autolag='AIC')
        # output for dft will give us without defining what the values are.
        # hence we manually write what values does it explains using a for loop
        output = pd.Series(adft[0:4],
                           index=['Test Statistics', 'p-value', 'No. of lags used', 'Number of observations used'])
        for key, values in adft[4].items():
            output['critical value (%s)' % key] = values
        print(output)

    def eliminateTred(self):
        rcParams['figure.figsize'] = 10, 6
        self._log_series = np.log(self._df_close)
        self._log_series = self._df_close
        moving_avg = self._df_close.rolling(12).mean()
        std_dev = self._df_close.rolling(12).std()
        plt.legend(loc='best')
        plt.title('Moving Average')
        plt.plot(std_dev, color="black", label="Standard Deviation")
        plt.plot(moving_avg, color="red", label="Mean")
        plt.legend()
        plt.show()

    def split_data(self, forecast_days=365):
        self._train_data, self._test_data = self._log_series[0:int(len(self._log_series) - forecast_days)], self._log_series[int(len(self._log_series) - forecast_days):]
        print(self._test_data)
        plt.figure(figsize=(10, 6))
        plt.grid(True)
        plt.xlabel('Dates')
        plt.ylabel('Closing Prices')
        plt.plot(self._train_data, 'green', label='Train data')
        plt.plot(self._test_data, 'blue', label='Test data')
        plt.legend()
        plt.show()

    def buildModel(self, train_data, order: tuple):
        model = ARIMA(train_data, order=order)
        self._fitted = model.fit()
        self._fitted.summary()


    @property
    def fitted(self):
        return self._fitted

    @property
    def train_data(self):
        return self._train_data

    @property
    def test_data(self):
        return self._test_data


#Use auto_arima to find best p,d,q parameters for forecasting
def arimaModel(train_data):
    model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
                                 test='adf',  # use adftest to find optimal 'd'
                                 max_p=10, max_q=10,  # maximum p and q
                                 m=100,  # frequency of series
                                 d=None,  # let model determine 'd'
                                 seasonal=True,  # No Seasonality
                                 start_P=0,
                                 D=0,
                                 trace=True,
                                 error_action='ignore',
                                 suppress_warnings=True,
                                 stepwise=True)
    print(model_autoARIMA.summary())
    print(model_autoARIMA.aic())
    model_autoARIMA.plot_diagnostics(figsize=(15, 8))
    plt.show()
    return model_autoARIMA