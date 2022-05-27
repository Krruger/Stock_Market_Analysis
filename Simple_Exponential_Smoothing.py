import numpy as np
from statsmodels.tsa.api import SimpleExpSmoothing
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

def ses(y, y_to_train, y_to_test, smoothing_level, predict_date):
    y['Close'].plot(marker='o', color='black', legend=True, figsize=(14, 7))

    fit1 = SimpleExpSmoothing(y_to_train['Close']).fit(smoothing_level=smoothing_level, optimized=False,)
    fcast1 = fit1.forecast(predict_date).rename(r'$\alpha={}$'.format(smoothing_level))
    # specific smoothing level
    fcast1 = pd.Series(fcast1, ).set_axis(test_data.index)
    fcast1.plot(marker='o', color='blue', legend=True)
    fit1.fittedvalues.plot(marker='o', color='blue')
    mse1 = ((fcast1 - y_to_test['Close']) ** 2).mean()
    print('The Root Mean Squared Error of our forecasts with smoothing level of {} is {}'.format(smoothing_level,
                                                                                                 round(np.sqrt(mse1),
                                                                                                       2)))

    ## auto optimization
    fit2 = SimpleExpSmoothing(y_to_train['Close']).fit()
    fcast2 = fit2.forecast(predict_date).rename(r'$\alpha=%s$' % fit2.model.params['smoothing_level'])
    fcast2 = pd.Series(fcast2,).set_axis(test_data.index)
    # plot
    fcast2.plot(marker='o', color='green', legend=True)
    fit2.fittedvalues.plot(marker='o', color='green')

    mse2 = ((fcast2 - y_to_test['Close']) ** 2).mean()
    print('The Root Mean Squared Error of our forecasts with auto optimization is {}'.format(round(np.sqrt(mse2), 2)))

    plt.show()
    return fcast1

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

predict_date=150
x = ses(df, train_data,test_data,0.8,predict_date)