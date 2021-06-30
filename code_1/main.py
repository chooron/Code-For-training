from scipy.optimize import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
# read data
data = pd.read_excel("01.xlsx")
data.dropna(axis=0)
pop1 = data['美国人口'].to_numpy()
pop2 = data['中国人口'].to_numpy()
year = data['年份'].to_numpy()

train_year = year[:-5]
index = year - year[0]
train_index = index[:-5]

test_index = train_index[-1] + np.array(list(range(0, 50, 2)))
test_year = train_year[-1] + np.array(list(range(0, 50, 2)))
# %%
# 马尔萨斯指数增长模型
train_pop1 = pop1[:-5]
pop0 = train_pop1[0]


# 马尔萨斯指数增长模型
def malthusFun(t, r):
    return pop0 * np.exp(r * t)


# 改进马尔萨斯指数增长模型
def malthusFun_pro(t, a, b):
    return (pop0 * a) / (b * pop0 + (a - b * pop0) * np.exp(-a * t))


mf_para = curve_fit(malthusFun, train_index, train_pop1, maxfev=1000000)[0]
# a, b = curve_fit(malthusFun_pro, index, pop, maxfev=1000000)[0]
plt.plot(test_year, malthusFun(test_index, mf_para), "*")
plt.plot(year, pop1)
plt.scatter(train_year, malthusFun(train_index, mf_para))
plt.show()


# %%
# logistic模型
def logisticFun(t, K, P0, r):
    exp_value = np.exp(r * (t))
    return (K * exp_value * P0) / (K + (exp_value - 1) * P0)


lf_para = curve_fit(logisticFun, train_index, train_pop1, maxfev=1000000)[0]
plt.plot(test_year, logisticFun(test_index, lf_para[0], lf_para[1], lf_para[2]), "*")
plt.plot(year, pop1)
plt.scatter(train_year, logisticFun(train_index, lf_para[0], lf_para[1], lf_para[2]))
plt.show()


# %%
# GM11
def gm11_model(k):
    from greytheory import GreyTheory
    temp_pop1 = train_pop1
    result = []
    for i in range(k):
        grey = GreyTheory()
        gm11 = grey.gm11
        # 参数设置
        gm11.alpha = 0.5
        # 加入数据
        for d in temp_pop1:
            gm11.add_pattern(d, "train")
        res = 0
        gm11.forecast()
        for forecast in gm11.analyzed_results:  # 从预测结果里判断，如果不等于_TAG_FORECAST_HISTORY则代表是预测值，因为只预测一个，所以预测结果列表里只有一个是预测值，其他可能是卷积值和历史值对应的预测值
            if forecast.tag != gm11._TAG_FORECAST_HISTORY:
                res = forecast.forecast_value
        temp_pop1 = np.append(temp_pop1, res)
        result += [res]
    return result


result = gm11_model(5)
plt.plot(year[-5:], result, "*")
plt.plot(year, pop1)
plt.show()

# %%
# ARIMA模型
from pmdarima import auto_arima

model = auto_arima(train_pop1, error_action='ignore', trace=True,
                   suppress_warnings=True, maxiter=10,
                   seasonal=False, m=12)
plt.plot(year[-5:], model.predict(5), "*")
plt.plot(year, pop1)
plt.show()