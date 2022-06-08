import math

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.api import SimpleExpSmoothing
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import Holt, ExponentialSmoothing

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')


dataName = 'jsw_arima.csv'
data = yf.Ticker("TSLA").history(period="max", interval="1d",)
data.to_csv(f"{dataName}",index = True, encoding="utf-8", index_label = "Date")

df = pd.read_csv(f'{dataName}', header=0, index_col='Date', parse_dates=True, date_parser=dateparse).fillna(0)
df.head()


def split_data(df_log):
    # train_data, test_data = df_log[3:int(len(df_log)*0.8)], df_log[int(len(df_log)*0.8):
    length = 250
    train_data = df_log[:len(df_log)-length]
    test_data = df_log[len(df_log)-length:]
    # plt.figure(figsize=(10,6))
    # plt.grid(True)
    # plt.xlabel('Dates')
    # plt.ylabel('Closing Prices')
    # plt.plot(df_log, 'green', label='Train data')
    # plt.plot(test_data, 'blue', label='Test data')
    # plt.legend()
    # plt.show()
    return train_data, test_data

#Aggregating the dataset at daily level
train_data, test_data = split_data(df)

y_hat_avg = test_data.copy()
fit1 = ExponentialSmoothing(np.asarray(train_data['Close']) ,seasonal_periods=2, trend='add', seasonal='add',).fit()
y_hat_avg['Holt_Winter'] = fit1.forecast(len(test_data['Close']))
plt.figure(figsize=(16,8))
plt.plot(train_data['Close'], label='Train')
plt.plot(test_data['Close'], label='Test')
plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.legend(loc='best')
plt.show()

# report performance
mse = mean_squared_error(test_data['Close'], y_hat_avg['Holt_Winter'])
print('MSE: '+str(mse))
mae = mean_absolute_error(test_data['Close'], y_hat_avg['Holt_Winter'])
print('MAE: '+str(mae))
rmse = math.sqrt(mean_squared_error(test_data['Close'], y_hat_avg['Holt_Winter']))
print('RMSE: '+str(rmse))
mape = np.mean(np.abs(y_hat_avg['Holt_Winter'] - test_data['Close'])/np.abs(test_data['Close']))
print('MAPE: '+str(mape))