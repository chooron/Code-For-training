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
plt.plot(test_year, logisticFun(test_index, lf_para[0], lf_para[1], lf_para[2]), "-")
plt.plot(year, pop1)
plt.scatter(train_year, logisticFun(train_index, lf_para[0], lf_para[1], lf_para[2]))
plt.show()
