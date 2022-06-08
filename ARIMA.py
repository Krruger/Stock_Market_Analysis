import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from pylab import rcParams
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima
# from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima.model import ARIMA
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')


dataName = 'jsw_arima.csv'
data = yf.Ticker("TSLA").history(period="max", interval="1d",)
data.to_csv(f"{dataName}",index = True, encoding="utf-8", index_label = "Date")

df = pd.read_csv(f'{dataName}', header=0, index_col='Date', parse_dates=True, date_parser=dateparse).fillna(0)
df.head()

# Updating the header
# df.drop(['Open','High','Low','Volume','Dividends','Stock Splits'],axis=1,inplace=True)
# test_result=adfuller(df['Close'])

#Plot close price
def _plotData(data):
    plot1 = plt.figure(1, figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Date')
    plt.ylabel('Close Prices')
    plt.plot(data['Close'])
    plt.show()
_plotData(df)

#Distribution of the dataset
def plot_distribution(data):
    plt.figure(2)
    df_close = data['Close']
    df_close.plot(kind='kde')
    plt.show()
    return df_close
df_close = plot_distribution(df)

#Test for stationarity
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()

    #plot rolling statistics:
    plt.plot(timeseries, color='blue', label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label='Rolling STD')
    plt.legend(loc='best')
    plt.title('Rolling mean and standard deviation')
    plt.show(block=True)
    adft = adfuller(timeseries,autolag='AIC')
    # output for dft will give us without defining what the values are.
    #hence we manually write what values does it explains using a for loop
    output = pd.Series(adft[0:4], index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
    for key,values in adft[4].items():
        output['critical value (%s)'%key] =  values
    print(output)

test_stationarity(df_close)

def decompose(timeseries):
    from statsmodels.tsa.seasonal import seasonal_decompose
    x = seasonal_decompose(timeseries, model="multiplicative", period = 30)
    fig = plt.figure()
    fig = x.plot()
    fig.set_size_inches(16,9)
    plt.show()

decompose(df_close)

#if not stationary then eliminate trend
#Eliminate trend
from pylab import rcParams
def eliminateTred(timeseries):
    rcParams['figure.figsize'] = 10, 6
    timeseries = np.log(timeseries)
    moving_avg = timeseries.rolling(12).mean()
    std_dev = timeseries.rolling(12).std()
    plt.legend(loc='best')
    plt.title('Moving Average')
    plt.plot(std_dev, color ="black", label = "Standard Deviation")
    plt.plot(moving_avg, color="red", label = "Mean")
    plt.legend()
    plt.show()
    return timeseries

log_series = eliminateTred(df_close)

#split data into train and training set

def split_data(df_log):
    # train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]
    length = 250
    train_data = df_log[:len(df_log)-length]
    test_data = df_log[len(df_log)-length:]
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Dates')
    plt.ylabel('Closing Prices')
    plt.plot(df_log, 'green', label='Train data')
    plt.plot(test_data, 'blue', label='Test data')
    plt.legend()
    plt.show()
    return train_data, test_data

train_data, test_data = split_data(log_series)

def _arimaModel(train_data):
    model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
                          test='adf',       # use adftest to find optimal 'd'
                          max_p=10, max_q=10, # maximum p and q
                          m=1,              # frequency of series
                          d=None,           # let model determine 'd'
                          seasonal=False,   # No Seasonality
                          start_P=0,
                          D=0,
                          trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)
    print(model_autoARIMA.summary())
    model_autoARIMA.plot_diagnostics(figsize=(15,8))
    plt.show()

_arimaModel(train_data)

#Modeling
# Build Model
def _buildModel(train_data):
    model = ARIMA(train_data, order=(3,0,8), trend=[0,1])
    fitted = model.fit()
    print(fitted.summary())
    return fitted

fitted = _buildModel(train_data)
fcast = fitted.get_forecast(len(test_data), alpha=0.2).summary_frame()

fc_series = pd.Series(fcast['mean'],).set_axis(test_data.index)
lower_series = pd.Series(fcast['mean_ci_lower'].set_axis(test_data.index))
upper_series = pd.Series(fcast['mean_ci_upper'].set_axis(test_data.index))
# Plot
plt.figure(figsize=(10,5), dpi=100)
plt.plot(train_data, label='training data')
plt.plot(test_data, color = 'blue', label='Actual Stock Price')
plt.plot(fc_series, color = 'orange',label='Predicted Stock Price')
plt.fill_between(lower_series.index, lower_series, upper_series,
                 color='k', alpha=.10)
plt.title('Prediction data')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend(loc='upper left', fontsize=8)
plt.show()

# report performance
mse = mean_squared_error(test_data, fc_series)
print('MSE: '+str(mse))
mae = mean_absolute_error(test_data, fc_series)
print('MAE: '+str(mae))
rmse = math.sqrt(mean_squared_error(test_data, fc_series))
print('RMSE: '+str(rmse))
mape = np.mean(np.abs(fc_series - test_data)/np.abs(test_data))
print('MAPE: '+str(mape))