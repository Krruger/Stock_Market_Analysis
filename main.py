import investpy as inv
import pandas as pd
# import pandas_datareader
from datetime import datetime, timezone
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import requests
from urllib.request import urlretrieve
ticker = "JSW"
interval ="5"
csv_file = "jsw.csv"

import yfinance as yf
import csv
def remove_Tz(df):
    df.apply(lambda x: datetime.replace(x, tzinfo=None))
    return df
data = yf.Ticker("JSW.WA").history(period="5d", interval="5m",)
data.to_csv("jsw.csv",index = True, encoding="utf-8", index_label = "Datetime")
data = pd.read_csv("jsw.csv")

data['Datetime'] = pd.to_datetime(data['Datetime'])
data['Datetime'] = data['Datetime'].apply(lambda x: datetime.replace(x, tzinfo=None))
print(data)



# fig = px.scatter(data, x="Open", y="Close")
# fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=data.index,
    y=data["Close"],
    mode="lines"
))


# fig = px.line(data, x='Datetime', y="Close", range_x=['2022-02-24', '2022-05-18'])
# fig.update_xaxes(
#     rangeslider_visible=True,
#     rangebreaks=[
#         dict(bounds=['sat', 'sun']),#hide weekends
#         dict(bounds=[17,9], pattern="hour"),
#     ]
# )
#
#
# fig.show()
# print(df)

# grab first and last observations from df.date and make a continuous date range from that
dt_all = pd.date_range(start=data['Datetime'].iloc[0],end=data['Datetime'].iloc[-1], freq = '1h')

# check which dates from your source that also accur in the continuous date range
dt_obs = [d.strftime("%Y-%m-%d %H:%M:%S") for d in data['Datetime']]

# isolate missing timestamps
dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d %H:%M:%S").tolist() if not d in dt_obs]
dt_breaks = pd.to_datetime(dt_breaks)

fig.show()
fig.update_xaxes(rangebreaks=[dict(dvalue = 5*60*1000, values=dt_breaks)] )
print(fig.layout.xaxis.rangebreaks)
fig.show()