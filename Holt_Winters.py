# dataframe opertations - pandas
import pandas as pd
# plotting data - matplotlib
from matplotlib import pyplot as plt
# time series - statsmodels
# Seasonality decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
# holt winters
# single exponential smoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
# double and triple exponential smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima
# from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima.model import ARIMA
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')

dataName = 'jsw_arima.csv'
data = yf.Ticker("JSW.WA").history(period="max", interval="1d",)
data.to_csv(f"{dataName}",index = True, encoding="utf-8", index_label = "Date")

airline = pd.read_csv(f'{dataName}', header=0, index_col='Date', parse_dates=True, date_parser=dateparse).fillna(0)
airline.head()

# finding shape of the dataframe
print(airline.shape)
# having a look at the data
print(airline.head())
# plotting the original data
# airline['Close'].plot(title='Close Data')

decompose_result = seasonal_decompose(airline["Close"],model='multiplicative',period = 30)
decompose_result.plot()


# Set the frequency of the date time index as Monthly start as indicated by the data
# airline.index.freq = 'MS'
# Set the value of Alpha and define m (Time Period)
m = 12
alpha = 1/(2*m)
airline['HWES1'] = SimpleExpSmoothing(airline['Close']).fit(smoothing_level=alpha,optimized=False,use_brute=True).fittedvalues
airline[['Close','HWES1']].plot(title='Holt Winters Single Exponential Smoothing');

airline['HWES2_ADD'] = ExponentialSmoothing(airline['Close'],trend='add').fit().fittedvalues
airline['HWES2_MUL'] = ExponentialSmoothing(airline['Close'],trend='mul').fit().fittedvalues
airline[['Close','HWES2_ADD','HWES2_MUL']].plot(title='Holt Winters Double Exponential Smoothing: Additive and Multiplicative Trend');


airline['HWES3_ADD'] = ExponentialSmoothing(airline['Close'],trend='add',seasonal='add',seasonal_periods=12).fit().fittedvalues
airline['HWES3_MUL'] = ExponentialSmoothing(airline['Close'],trend='mul',seasonal='mul',seasonal_periods=12).fit().fittedvalues
airline[['Close','HWES3_ADD','HWES3_MUL']].plot(title='Holt Winters Triple Exponential Smoothing: Additive and Multiplicative Seasonality');
plt.show()


forecast_data = pd.read_csv(f'{dataName}', header=0, index_col='Date', parse_dates=True, date_parser=dateparse).fillna(0)


# forecast_data.index.freq = 'MS'
length = 250
# Split into train and test set
train_airline = forecast_data[:len(forecast_data)-length]
test_airline = forecast_data[len(forecast_data)-length:]

fitted_model = ExponentialSmoothing(train_airline['Close'],trend='mul',seasonal='mul',seasonal_periods=24).fit()
test_predictions = fitted_model.forecast(length).set_axis(test_airline.index)
train_airline['Close'].plot(legend=True,label='TRAIN')
test_airline['Close'].plot(legend=True,label='TEST',figsize=(6,4))
test_predictions.plot(legend=True,label='PREDICTION')
plt.title('Train, Test and Predicted Test using Holt Winters')
plt.show()