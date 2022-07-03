import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import Holt

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')

dataName = 'jsw_arima.csv'
# data = yf.Ticker("TSLA").history(period="max", interval="1d", )
# data.to_csv(f"{dataName}", index=True, encoding="utf-8", index_label="Date")

class Hold_Winters():
    
    def __init__(self, df,future_day):
        self._airline = pd.DataFrame({"Values":df})
        self._train_airline = self._airline[:len(self._airline) - future_day]
        self._test_airline = self._airline[len(self._airline) - future_day:]

    def prediction(self,seasonal_periods = 24, future_day = 365):
        # airline = pd.read_csv(f'{dataName}', header=0, index_col='Date', parse_dates=True, date_parser=dateparse).fillna(0)
        # airline.head()
        #
        # decompose_result = seasonal_decompose(self._airline["Values"], model='multiplicative', period=60)
        # decompose_result.plot()

        # Set the frequency of the date time index as Monthly start as indicated by the data
        forecast_data = pd.read_csv(f'{dataName}', header=0, index_col='Date', parse_dates=True, date_parser=dateparse).fillna(
            0)

        # Split into train and test set

        fitted_model = ExponentialSmoothing(self._train_airline["Values"], trend='add', seasonal='add', seasonal_periods=seasonal_periods).fit()
        self._test_predictions = fitted_model.forecast(future_day).set_axis(self._test_airline.index)
        self._train_airline["Values"].plot(legend=True, label='TRAIN')
        self._test_airline["Values"].plot(legend=True, label='TEST', figsize=(6, 4))
        self._test_predictions.plot(legend=True, label='PREDICTION')
        plt.title('Train, Test and Predicted Test using Holt Winters')
        plt.show()

    @property
    def traing_airline(self):
        return self._train_airline

    @property
    def test_airline(self):
        return self._test_airline

    @property
    def test_predictions(self):
        return self._test_predictions