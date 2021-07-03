# 评估模型结果
from sklearn import metrics
import pandas as pd
import numpy as np


def NSE(real, pred):
    return metrics.r2_score(real, pred)


def RMSE(real, pred):
    return metrics.mean_squared_error(real, pred) ** 0.5


def MSE(real, pred):
    return metrics.mean_squared_error(real, pred)


def MAE(real, pred):
    return metrics.mean_absolute_error(real, pred)


def evaluate(train_real, train_pred, test_real, test_pred, path):
    print("train mae:{:.2},test mae:{:.2}".format(MAE(train_real, train_pred),
                                                  MAE(test_real, test_pred)))
    print("train mse:{:.2},test mse:{:.2}".format(MSE(train_real, train_pred),
                                                  MSE(test_real, test_pred)))
    print("train rmse:{:.2},test rmse:{:.2}".format(RMSE(train_real, train_pred),
                                                    RMSE(test_real, test_pred)))
    print("train nse:{:.2},test nse:{:.2}".format(NSE(train_real, train_pred),
                                                  NSE(test_real, test_pred)))
    col1 = ['mae', 'mse', 'rmse', 'nse']
    train_index = [MAE(train_real, train_pred), MSE(train_real, train_pred), RMSE(train_real, train_pred),
                   NSE(train_real, train_pred)]
    test_index = [MAE(test_real, test_pred), MSE(test_real, test_pred), RMSE(test_real, test_pred),
                  NSE(test_real, test_pred)]
    df = pd.DataFrame(np.array([col1, train_index, test_index]).T, columns=['index', 'train', 'test'])
    df.to_csv(path,index=False)
