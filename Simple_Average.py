import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')


dataName = 'jsw_arima.csv'
data = yf.Ticker("TSLA").history(period="max", interval="1d",)
data.to_csv(f"{dataName}",index = True, encoding="utf-8", index_label = "Date")

df = pd.read_csv(f'{dataName}', header=0, index_col='Date', parse_dates=True, date_parser=dateparse).fillna(0)
df.head()


def split_data(df_log):
    train_data, test_data = df_log[3:int(len(df_log)*0.95)], df_log[int(len(df_log)*0.95):]
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
y_hat_avg['avg_forecast'] = train_data['Close'].mean()
y_hat_avg = pd.Series(y_hat_avg['avg_forecast'],).set_axis(test_data.index)
plt.figure(figsize=(12,8))
# plt.plot(train_data['Close'], label='Train')
# plt.plot(test_data['Close'], label='Test')
plt.plot(y_hat_avg, label='Average Forecast')
plt.legend(loc='best')
plt.show()