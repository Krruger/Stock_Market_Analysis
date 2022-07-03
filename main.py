from LSTM import NeuralNetwork
from ARIMA_Model import ARIMA_Model, arimaModel, ARIMA
import datetime as dt
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
#
# import sys
# import pyodbc
# import os
# import glob
#
# import config
# from os.path import join
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
import pandas as pd
# Database credentials and variables
server = r'LAPTOP-UDJ0KQGQ\SQLEXPRESS'
user = ''
password = ''
database = 'TFM'

# Connect to SQL database
database_connection = pymssql.connect(server, user, password, database)
cursor = database_connection.cursor()
# dataframe = pd.read_sql_query('SELECT * FROM [dbo].[Node]', database_connection)
x = pd.read_sql_query("SELECT VariableValue FROM [dbo].[Activity] WHERE VariableName ='Temperature' AND "
                      "NodeName = 'Górne łożysko silnika głównego'", database_connection)

for current in range(len(x)):
    current = current+1
    previous = current - 1
    if x['VariableValue'][previous] > 250 and x['VariableValue'][previous] > x['VariableValue'][current]:
        index = current

x = x[index:len(x)]
x = x.set_axis(range(0,x.index.stop-x.index.start,x.index.step))
x = x.rename(columns={"VariableValue":'Close'})
forecast_day = 365
z = x[int(len(x))-730:len(x)]
z =  z.set_axis(range(0,z.index.stop-z.index.start,z.index.step))
error = pd.DataFrame([],columns=[])

## ================================= Simple_Exponential_Smoothing ================================================##
# from Simple_Exponential_Smoothing import SimpleExponentialSmoothing
# smoothing_list = [0,0.2,0.6,1]
# ses = SimpleExponentialSmoothing(x['Close'],forecast_day)
#
# for smoothing in smoothing_list:
#     fcast = ses.prediction(forecast_day, smoothing)
#     from Error_Calc import error_Calc
#     error = error.append(error_Calc(ses.test_data, fcast, fr"alpha={smoothing}"))
#
# error_key_list = ['MAE', 'MAPE', 'MSE', 'RMSE']
# for error_key in error_key_list:
#     plt.figure(figsize=(8, 6))
#     for key in error.keys():
#         if error_key in key and error_key[0] == key[0]:
#             plt.plot(error[key], label=f"{key}")
#     plt.legend()
#     plt.show()

# Visualization data
# plt.figure(figsize=(8,6))
# plt.plot(x, label="Górne łożysko silnika głównego")
# plt.xlabel("Numer próbki")
# plt.ylabel("Temperatura ['C]")
# plt.legend("Górne łożysko silnika głównego", loc='best',  fontsize=8)
# plt.title("Wykres temperatury górnego łożyska silnika głównego")
# plt.show()
#
# plt.figure(figsize=(8,6))
# plt.plot(x, label="Górne łożysko silnika głównego")
# plt.xlabel("Numer próbki")
# plt.ylabel("Temperatura ['C]")
# plt.legend("Górne łożysko silnika głównego", loc='best',  fontsize=8)
# plt.title("Wykres temperatury górnego łożyska silnika głównego")
# plt.show()

# ===================================================== HOLT ==================================== ##
# train_data = x[:len(x) - forecast_day]
# test_data = x[len(x) - forecast_day:]
# from statsmodels.tsa.holtwinters import Holt
#
# test_predictions = test_data.copy()
#
# smoothing_list = [0,0.2,0.6,1]
# smoothing_trend_list = [0,0.2,0.6,1,2,5]
# for smoothing in smoothing_list:
#     fitted_model = Holt(np.asarray(train_data)).fit(smoothing_level=smoothing, smoothing_trend=1)
#     fcast = fitted_model.forecast(forecast_day)
#     from Error_Calc import error_Calc
#     error = error.append(error_Calc(test_data['Close'], fcast, fr"alpha={smoothing}"))
#
# error_key_list = ['MAE', 'MAPE', 'MSE', 'RMSE']
# for error_key in error_key_list:
#     plt.figure(figsize=(8, 6))
#     for key in error.keys():
#         if error_key in key and error_key[0] == key[0]:
#             plt.plot(error[key], label=f"{key}")
#     plt.legend()
#     plt.show()
#
#
#
# smoothing_trend_list = [0.2,0.6,1,2,5]
# for smoothing in smoothing_trend_list:
#     print(smoothing)
#     fitted_model = Holt(np.asarray(train_data)).fit(smoothing_level=1, smoothing_trend=0.2)
#     fcast = fitted_model.forecast(forecast_day)
#     from Error_Calc import error_Calc
#     error = error.append(error_Calc(test_data['Close'], fcast, fr"alpha={smoothing}"))
#
# error_key_list = ['MAE', 'MAPE', 'MSE', 'RMSE']
# for error_key in error_key_list:
#     plt.figure(figsize=(8, 6))
#     for key in error.keys():
#         if error_key in key and error_key[0] == key[0]:
#             plt.plot(error[key], label=f"{key}")
#     plt.legend()
#     plt.show()
#
# plt.figure(figsize=(16,8))
# plt.plot(train_data, label='TRAIN')
# plt.plot(test_data, label='TEST')
# plt.plot(test_predictions["test_predictions"], label='PREDICTION')
# plt.title('Train, Test and Predicted Test using Holt Winters')
# plt.show()

#=============================== Build the LSTM Model =======================================##
p = NeuralNetwork(z)
forecast_day = 20
p.prepare_data(forecast_day)
p.prediction_Data_Prepare(60)

from LSTM_Model import lstm_Model
units = ([50,25,10],[200,100,50],[500,200,100],[1000,500,400], [2000,1500,1000],[2000,1000,1])
units = ([10,10,1],[20,10,10])
epochs = 2
for unit in units:
    model = lstm_Model(unit,p.x, p.y, epochs=epochs,batch_size= 128,verbose= 32,values=p.x)
    p.forecast(forecast_day, model)
    ##========================== Serialize and plotting ======================================##

    # organize the results in a data frame
    df_past = p._data[['Close']]
    df_past.rename(columns={'index': 'Date'}, inplace=True)
    # df_past['Date'] = pd.to_datetime(df_past['Date'])
    df_past['Forecast'] = np.nan

    # df_future = pd.DataFrame(columns=['Date', 'Close', 'Forecast'])
    # df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=future_sample)
    # df_future['Forecast'] = p.y_future.flatten()
    # df_future['Close'] = np.nan
    prediciton_value = pd.DataFrame(p.y_future).set_axis(p._data.index[len(p._data)-forecast_day:])
    train_data = pd.DataFrame(p._data[:len(p._data)-forecast_day])
    ver_data = pd.DataFrame(p._data[len(p._data)-forecast_day:])
    # results = df_past.append(df_future).set_index('Date')

    plt.figure(figsize=(16,8))
    plt.plot(prediciton_value, label="Forecasting")
    plt.plot(train_data, label = "train_data")
    plt.plot(ver_data, label = 'Verification Data')
    plt.legend()
    plt.show()
    plt.plot()
    from Error_Calc import error_Calc
    error = error.append(error_Calc(p.y_future, ver_data, parameter_name=unit))
    # plot the results
    # results.plot(title=company + "LSTM neural network prediction")


error_key_list = ['MAE', 'MAPE', 'MSE', 'RMSE']
for error_key in error_key_list:
    plt.figure(figsize=(8, 6))
    for key in error.keys():
        if error_key in key and error_key[0] == key[0]:
            plt.plot(error[key], label=f"{key}")
    plt.legend()
    plt.show()

# ==================================================== ARIMA MODEL ================================================ #
# from ARIMA_Model import ARIMA_Model, arimaModel
# from Error_Calc import error_Calc
# z = x['Close'].tolist()
# p = ARIMA_Model(x['Close'][int(len(x))-730:len(x)])
# p.test_stationarity()
#
# p.eliminateTred()
# p.split_data(365)
# # model_autoARIMA = arimaModel(p.train_data)
# #Prediction
# order_list = [(2,1,0),(1,1,1),(1,0,1),(2,0,2),(2,2,2)]
# for order in order_list:
#     p.buildModel(p.train_data, order)
#
#     # Plot model training data and training data
#     # plt.plot(p.train_data, label="Train Data")
#     # plt.plot(p.fitted.fittedvalues, color='red', label= 'Forecast')
#     # plt.show()
#
#     # Forecasting
#     # fcast = p.fitted.get_forecast(len(p.test_data),).summary_frame()
#     fcast = p.fitted.predict(start=len(p.train_data), end=len(p.train_data)+len(p.test_data)-1,dynamic=False)
#
#     # fc_series = pd.Series(fcast['mean'], ).set_axis(p.test_data.index)
#     # lower_series = pd.Series(fcast['mean_ci_lower'].set_axis(p.test_data.index))
#     # upper_series = pd.Series(fcast['mean_ci_upper'].set_axis(p.test_data.index))
#     fc_series = pd.Series(fcast).set_axis(p.test_data.index)
#     # Plot
#     # plt.figure(figsize=(10, 5), dpi=100)
#     # plt.plot(p.train_data, label='training data')
#     # plt.plot(p.test_data, color='blue', label='Actual Stock Price')
#     # plt.plot(fc_series, color='orange', label=f'Predicted Stock Price for order {order}')
#     # # plt.fill_between(lower_series.index, lower_series, upper_series,
#     # #                  color='k', alpha=.10)
#     # plt.title('ARIMA prediction model')
#     # plt.xlabel('Time')
#     # plt.ylabel('Value')
#     # plt.legend(loc='upper left', fontsize=8)
#     # plt.show()
#     error = error.append(error_Calc(p.test_data, fc_series, parameter_name=order))
#
# error_key_list = ['MAE', 'MAPE', 'MSE', 'RMSE']
# for error_key in error_key_list:
#     plt.figure(figsize=(8, 6))
#     for key in error.keys():
#         if error_key in key and error_key[0] == key[0]:
#             plt.plot(error[key], label=f"{key}")
#     plt.legend()
#     plt.show()

# ========================================================= Holt Winters ============================================ #
# from Holt_Winters import Hold_Winters
# from Error_Calc import error_Calc
#
#
# holt = Hold_Winters(x['Close'], forecast_day)
# seasonal_periods_list = [2,4]
# for seasonal_periods in seasonal_periods_list:
#     holt.prediction(seasonal_periods = seasonal_periods, future_day=forecast_day)
#     error = error.append(error_Calc(holt.test_airline["Values"],holt.test_predictions, parameter_name=seasonal_periods))
#
# error_key_list = ['MAE', 'MAPE', 'MSE', 'RMSE']
# for error_key in error_key_list:
#     plt.figure(figsize=(8, 6))
#     for key in error.keys():
#         if error_key in key and error_key[0] == key[0]:
#             plt.plot(error[key], label=f"{key}")
#     plt.legend()
#     plt.show()