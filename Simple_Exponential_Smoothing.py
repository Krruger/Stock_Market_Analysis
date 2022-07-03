import numpy as np
from statsmodels.tsa.api import SimpleExpSmoothing
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

class SimpleExponentialSmoothing():
    def __init__(self, data, forecast_day):
        self._data = data
        self.split_data(forecast_day)

    def split_data(self, forecast_day = 365):
        self._train_data, self._test_data = self._data[0:int(len(self._data)-forecast_day)], self._data[int(len(self._data)-forecast_day):]
        plt.figure(figsize=(10,6))
        plt.grid(True)
        plt.xlabel('Dates')
        plt.ylabel('Closing Prices')
        plt.plot(self._data, 'green', label='Train data')
        plt.plot(self._test_data, 'blue', label='Test data')
        plt.legend()
        plt.show()

    def prediction(self,forecast_day,smoothing_level):
        self._data.plot(color='black', legend=True, figsize=(14, 7))

        fit1 = SimpleExpSmoothing(self._train_data).fit(smoothing_level=smoothing_level, optimized=False,)
        fcast1 = fit1.forecast(forecast_day).rename(r'$\alpha={}$'.format(smoothing_level))
        # specific smoothing level
        fcast1 = pd.Series(fcast1, ).set_axis(self._test_data.index)
        fcast1.plot(color='red', legend=True)
        fit1.fittedvalues.plot( color='blue')
        plt.show()
        return fcast1

    @property
    def train_data(self):
        return self._train_data

    @property
    def test_data(self):
        return self._test_data