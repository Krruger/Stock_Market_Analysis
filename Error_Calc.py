import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import math
import matplotlib.pyplot as plt

def error_Calc(test_data, compared_value, parameter_name):
    df_error = pd.DataFrame([], columns={f"MSE {parameter_name}", f"MAE {parameter_name}", f"RMSE {parameter_name}", f"MAPE {parameter_name}"})
    for x in range(len(test_data)):
        x += 1
        mse = mean_squared_error(test_data[:x], compared_value[:x])
        mae = mean_absolute_error(test_data[:x], compared_value[:x])
        rmse = math.sqrt(mean_squared_error(test_data[:x], compared_value[:x]))
        mape = np.mean(np.abs(test_data[:x] - compared_value[:x]) / test_data[:x])
        print(mape)
        df_error = df_error.append({
            f"MSE {parameter_name}": mse,
            f"MAE {parameter_name}": mae,
            f"RMSE {parameter_name}": rmse,
            f"MAPE {parameter_name}": mape,
        }, ignore_index=True)
    plt.plot(df_error[f"MSE {parameter_name}"], label="MSE")
    plt.plot(df_error[f"MAE {parameter_name}"], label="MAE")
    plt.plot(df_error[f"RMSE {parameter_name}"], label="RMSE")
    plt.plot(df_error[f"MAPE {parameter_name}"], label="MAPE")
    plt.show()
    return df_error