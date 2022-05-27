import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import yfinance as yf
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')


dataName = 'jsw_arima.csv'
data = yf.Ticker("TSLA").history(period="max", interval="1d",)
data.to_csv(f"{dataName}",index = True, encoding="utf-8", index_label = "Date")

df = pd.read_csv(f'{dataName}', header=0, index_col='Date', parse_dates=True, date_parser=dateparse).fillna(0)
df.head()

