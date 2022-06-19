import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from statsmodels.tsa.holtwinters import Holt

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')

company = "SYNA"
dataName = company + '.csv'
data = yf.Ticker(company).history(period="max", interval="1d",)
data.to_csv(f"{dataName}",index = True, encoding="utf-8", index_label = "Date")

df = pd.read_csv(f'{dataName}', header=0, index_col='Date', parse_dates=True, date_parser=dateparse).fillna(0)


def split_data(df_log, train_persentage = 0.98):
    train_data, test_data = df_log[3:int(len(df_log)*train_persentage)], df_log[int(len(df_log)*train_persentage):]
    return train_data, test_data

#Aggregating the dataset at daily level
train_data, test_data = split_data(df)

x = sm.tsa.seasonal_decompose(train_data['Close'],model="multiplicative",period = 120).plot()
result = sm.tsa.stattools.adfuller(train_data['Close'])

y_hat_avg = test_data.copy()

fit1 = Holt(np.asarray(train_data['Close'])).fit(smoothing_level = 0.3,smoothing_slope = 0.1)
y_hat_avg['Holt_linear'] = fit1.forecast(len(test_data))

plt.figure(figsize=(16,8))
plt.plot(train_data['Close'], label='Train')
plt.plot(test_data['Close'], label='Test')
plt.plot(y_hat_avg['Holt_linear'], label='Holt_linear')
plt.legend(loc='best')
plt.show()