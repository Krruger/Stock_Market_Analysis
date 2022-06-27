# from LSTM import NeuralNetwork
# from ARIMA_Model import ARIMA_Model, arimaModel
# import datetime as dt
# from datetime import datetime
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, LSTM
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# #======================================================= INPUTS =====================================================##
# start=dt.datetime(2012, 1, 1)
# end=datetime.now().date()
# company = "SYNA"
#
# #===================================================LSTM NEURAL NETWO================================================##
#
#
# p = NeuralNetwork(company, start, end)
# p.prepare_data(0.80)
# p.prediction_Data_Prepare(60)
#
# ##=============================== Build the Model =======================================##
#
# model = Sequential()
#
# # model.add(LSTM(units=100,  return_sequences=True, input_shape=(1, 1)))
# model.add(LSTM(units=200, return_sequences=True, input_shape=p.x.shape[1:]))
# model.add(Dropout(0.2))
#
# model.add(LSTM(units=100, return_sequences=True))
# model.add(Dropout(0.2))
#
# model.add(LSTM(units=50))
# model.add(Dropout(0.2))
#
# model.add(Dense(units=1))  # Prediction of the next closing value
#
# model.compile(optimizer="adam", loss="mean_squared_error")
# model.fit(p.x, p.y, epochs=2, batch_size=128, verbose=2)
#
# future_sample = 150
#
# p.forecast(future_sample, model)
# ##========================== Serialize and plotting ======================================##
# # organize the results in a data frame
# df_past = p._data[['Close']].reset_index()
# df_past.rename(columns={'index': 'Date'}, inplace=True)
# df_past['Date'] = pd.to_datetime(df_past['Date'])
# df_past['Forecast'] = np.nan
#
# df_future = pd.DataFrame(columns=['Date', 'Close', 'Forecast'])
# df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=future_sample)
# df_future['Forecast'] = p.y_future.flatten()
# df_future['Close'] = np.nan
# results = df_past.append(df_future).set_index('Date')
#
# # plot the results
# results.plot(title=company + "LSTM neural network prediction")
#
#
# #================================================ ARIMA =============================================================##
# from ARIMA_Model import ARIMA_Model, arimaModel
# p = ARIMA_Model(company, start, end)
# p.test_stationarity()
# p.eliminateTred()
# p.split_data(0.98)
# model_autoARIMA = arimaModel(p.train_data)
#
# #Prediction
#
# order = model_autoARIMA.order
# p.buildModel(p.train_data, order)
# fcast = p.fitted.get_forecast(len(p.test_data), alpha=0.2).summary_frame()
#
# fc_series = pd.Series(fcast['mean'], ).set_axis(p.test_data.index)
# lower_series = pd.Series(fcast['mean_ci_lower'].set_axis(p.test_data.index))
# upper_series = pd.Series(fcast['mean_ci_upper'].set_axis(p.test_data.index))
# # Plot
# plt.figure(figsize=(10, 5), dpi=100)
# plt.plot(p.train_data, label='training data')
# plt.plot(p.test_data, color='blue', label='Actual Stock Price')
# plt.plot(fc_series, color='orange', label='Predicted Stock Price')
# plt.fill_between(lower_series.index, lower_series, upper_series,
#                  color='k', alpha=.10)
# plt.title('ARIMA prediction model')
# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.legend(loc='upper left', fontsize=8)
# plt.show()

import sys
import pyodbc
import os
import glob

import config
from os.path import join
# import win32com.shell.shell as shell
# filename = "TFM3.bak"
# filepath = (os.getcwd() + "\\" + filename)
# print(filepath)
# conn_info = "DRIVER={SQL Server};SERVER=%s;DATABASE=master;UID=%s;PWD=%s" % ("TFM",
#                                                                              "USER",
#                                                                              "PASS")
# cnct_str = pyodbc.connect(conn_info, autocommit=True)
# cur = cnct_str.cursor()
# cur.execute(
#     """RESTORE DATABASE [%s] FROM  DISK = N'%s' WITH  FILE = 1, NOUNLOAD, REPLACE, STATS = 5""" % (db_name, filepath))
# while cur.nextset():
#     pass
# print("restore_backup completed successfully")

# import pandas as pd
# import pyodbc
#
# conn = pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};'
#                       'Server=LAPTOP-UDJ0KQGQ\SQLEXPRESS;'
#                       'Database=TFM3;'
#                       'Trusted_Connection=yes;')
import pymssql
print(pymssql.__version__)
import pandas as pd
# Database credentials and variables
server = r'LAPTOP-UDJ0KQGQ\SQLEXPRESS'
user = ''
password = ''
database = 'TFM'

# Connect to SQL database
database_connection = pymssql.connect(server, user, password, database)
cursor = database_connection.cursor()
dataframe = pd.read_sql_query('SELECT * FROM [dbo].[Node]', database_connection)
x = pd.read_sql_query("SELECT * FROM [dbo].[Activity] WHERE VariableName ='Temperature' AND NodeName = 'Górne łożysko silnika głównego'", database_connection)
ask = "SELECT * FROM [dbo].[Activity] WHERE VariableName ='Temperature'"
