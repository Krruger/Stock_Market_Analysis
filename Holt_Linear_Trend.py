import numpy as np
from statsmodels.tsa.api import SimpleExpSmoothing
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import Holt

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')


dataName = 'jsw_arima.csv'
data = yf.Ticker("TSLA").history(period="max", interval="1d",)
data.to_csv(f"{dataName}",index = True, encoding="utf-8", index_label = "Date")

df = pd.read_csv(f'{dataName}', header=0, index_col='Date', parse_dates=True, date_parser=dateparse).fillna(0)
df.head()


def split_data(df_log):
    train_data, test_data = df_log[3:int(len(df_log)*0.98)], df_log[int(len(df_log)*0.98):]
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

x = sm.tsa.seasonal_decompose(train_data['Close'],model="multiplicative",period = 120).plot()
result = sm.tsa.stattools.adfuller(train_data['Close'])
# plt.show()

y_hat_avg = test_data.copy()

fit1 = Holt(np.asarray(train_data['Close'])).fit(smoothing_level = 0.3,smoothing_slope = 0.1)
y_hat_avg['Holt_linear'] = fit1.forecast(len(test_data))

plt.figure(figsize=(16,8))
plt.plot(train_data['Close'], label='Train')
plt.plot(test_data['Close'], label='Test')
plt.plot(y_hat_avg['Holt_linear'], label='Holt_linear')
plt.legend(loc='best')
plt.show()