import investpy as inv
import pandas as pd
import pandas_datareader
from datetime import datetime
import requests
from urllib.request import urlretrieve
ticker = "JSW"
interval ="5"
csv_file = "jsw.csv"

import yfinance as yf

#
# url = urlretrieve(f'https://stooq.com/q/d/l/?s={ticker}&i={interval}',csv_file)
#
# data = pd.read_csv(csv_file, index_col='Date', parse_dates=['Date'],
#             date_parser=lambda x: datetime.strptime(x, '%Y-%m-%d'))
# data.tail()
# print(data)

print(yf.Ticker("JSW.WA").history(period="200d", interval="5m"))