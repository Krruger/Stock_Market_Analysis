from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Modeling and Forecasting
# ==============================================================================
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoregMultiOutput import ForecasterAutoregMultiOutput
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster

from joblib import dump, load


dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')


dataName = 'jsw_arima.csv'
data = yf.Ticker("TSLA").history(period="max", interval="1d",)
data.to_csv(f"{dataName}",index = True, encoding="utf-8", index_label = "Date")

data = pd.read_csv(f'{dataName}', header=0, index_col='Date', parse_dates=True, date_parser=dateparse).fillna(0)
data.head()
print(f'Number of rows with missing values: {data.isnull().any(axis=1).mean()}')

length = 50
def split_data(df_log):
    # train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]
    train_data = df_log[:len(df_log)-length]
    test_data = df_log[len(df_log)-length:]
    return train_data, test_data

data_train, data_test = split_data(data)

print(f"Train dates : {data_train.index.min()} --- {data_train.index.max()}  (n={len(data_train)})")
print(f"Test dates  : {data_test.index.min()} --- {data_test.index.max()}  (n={len(data_test)})")

fig, ax=plt.subplots(figsize=(9, 4))
data_train['Close'].plot(ax=ax, label='train')
data_test['Close'].plot(ax=ax, label='test')
ax.legend();

forecaster = ForecasterAutoreg(
                regressor = RandomForestRegressor(random_state=12),
                lags = 2
                )

forecaster.fit(y=data_train['Close'])
forecaster

predictions = forecaster.predict(steps=length)
predictions.head()

fc_series = pd.Series(predictions,).set_axis(data_test.index)

fig, ax = plt.subplots(figsize=(9, 4))
data_train['Close'].plot(ax=ax, label='train')
data_test['Close'].plot(ax=ax, label='test')
fc_series.plot(ax=ax, label='predictions')
ax.legend()