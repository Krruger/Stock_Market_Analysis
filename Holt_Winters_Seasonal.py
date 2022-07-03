import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')

dataName = 'jsw_arima.csv'
data = yf.Ticker("TSLA").history(period="max", interval="1d", )
data.to_csv(f"{dataName}", index=True, encoding="utf-8", index_label="Date")

df = pd.read_csv(f'{dataName}', header=0, index_col='Date', parse_dates=True, date_parser=dateparse).fillna(0)
df.head()


def split_data(df, forecast_days=365):
    train_data, test_data = df[:int(len(df) - forecast_days)], df[int(len(df) - forecast_days):]
    return train_data, test_data


# Aggregating the dataset at daily level
train_data, test_data = split_data(df)

y_hat_avg = test_data.copy()
fit1 = ExponentialSmoothing(np.asarray(train_data['Close']), seasonal_periods=50, trend='add', seasonal='mul',
                            initialization_method='estimated').fit()
y_hat_avg['Holt_Winter'] = fit1.forecast(len(test_data['Close']))
plt.figure(figsize=(16, 8))
plt.plot(train_data['Close'], label='Train')
plt.plot(test_data['Close'], label='Test')
plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.legend(loc='best')
plt.show()
