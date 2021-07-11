from scipy.optimize import *
import matplotlib.pyplot as plt
import numpy as np
from code_1.gen_sample import load_data
from code_1 import evaluate as eval
from sklearn.preprocessing import MinMaxScaler
from code_1.utils.plot_utils import plot_rela_pred
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']
path = 'fig/logistic/'
# %%
scale = MinMaxScaler()
# read data
dataName = "中国人口"
isScale = False
year, pop = load_data(dataName)
train_year = year[:-5]
test_year = year[-5:]
index = year - year[0]
train_index = index[:-5]
test_index = train_index[-1] + np.array(list(range(10, 60, 10)))  # 由于数据不等长故重新生成一个index

train_pop = pop[:-5]
if isScale: train_pop = scale.fit_transform(train_pop.reshape(-1, 1))


# %%
# logistic模型
def logisticFun(t, K, P0, r):
    exp_value = np.exp(r * (t))
    return (K * exp_value * P0) / (K + (exp_value - 1) * P0)


lf_para = curve_fit(logisticFun, train_index, train_pop, maxfev=1000000)[0]
print('------logistic模型结果-------')
train_pred = logisticFun(train_index, lf_para[0], lf_para[1], lf_para[2])
test_pred = logisticFun(test_index, lf_para[0], lf_para[1], lf_para[2])
eval.evaluate(pop[:-5], train_pred, pop[-5:], test_pred, path=path + dataName + '(index).csv')
total_pred = np.concatenate((train_pred, test_pred))
df = pd.DataFrame()
df['year'] = year
df['real'] = pop
df['pred'] = total_pred
df.to_csv(path + dataName + '(pred).csv', index=False)

plt.plot([train_year[-1]] + list(test_year), [train_pred[-1]] + list(test_pred), "-")
plt.plot(year, pop)
plt.plot(train_year, train_pred)
plt.axvline(year[-6], color='black', linestyle='--')
plt.legend(['预测值', '实际值', '拟合值'])
plt.xlabel('年份')
plt.ylabel('人口数量($\mathregular{10^6}$)')
plt.savefig(path + "logistic模型({}).png".format(dataName))
plt.show()

plot_rela_pred(pop[:-5], train_pred, time=train_year, fig_savepath=path + '{}(train).png'.format(dataName),
               measurement_time='a',
               measurement_unit='$\mathregular{10^6}$')
plot_rela_pred(pop[-5:], test_pred, time=test_year, fig_savepath=path + '{}(test).png'.format(dataName),
               measurement_time='a',
               measurement_unit='$\mathregular{10^6}$')
