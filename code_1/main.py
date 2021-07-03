from scipy.optimize import *
import matplotlib.pyplot as plt
import numpy as np
from code_1.gen_sample import load_data
from code_1 import evaluate as eval
from sklearn.preprocessing import MinMaxScaler
from utils.plot_utils import plot_rela_pred
import pandas as pd
plt.rcParams['font.sans-serif'] = ['SimHei']
path = '../fig/malthus/'
# %%
scale = MinMaxScaler()
# read data
dataName = "美国人口"
isScale = False
year, pop = load_data(dataName)
train_year = year[:-5]
index = year - year[0]
train_index = index[:-5]

test_index = train_index[-1] + np.array(list(range(10, 60, 10)))
test_year = train_year[-1] + np.array(list(range(10, 60, 10)))

# %%
# 马尔萨斯指数增长模型
train_pop = pop[:-5]
if isScale: train_pop = scale.fit_transform(train_pop.reshape(-1, 1))
pop0 = train_pop[0]


# 马尔萨斯指数增长模型(数据可能不能进行归一化处理)
def malthusFun(t, r):
    return pop0 * np.exp(r * t)


# 改进马尔萨斯指数增长模型
def malthusFun_pro(t, a, b):
    return (pop0 * a) / (b * pop0 + (a - b * pop0) * np.exp(-a * t))


mf_para = curve_fit(malthusFun, train_index, train_pop, maxfev=1000000)[0]

print('------马尔萨斯指数增长模型结果-------')
train_pred = malthusFun(train_index, mf_para)
test_pred = malthusFun(test_index, mf_para)
if isScale:
    train_pred = scale.inverse_transform(train_pred.reshape(-1, 1))
    test_pred = scale.inverse_transform(test_pred.reshape(-1, 1))
eval.evaluate(pop[:-5], train_pred, pop[-5:], test_pred, path=path + dataName + '(index).csv')
total_pred = np.concatenate((train_pred, test_pred))
df = pd.DataFrame()
df['year'] = year
df['real'] = pop
df['pred'] = total_pred
df.to_csv(path + dataName + '(pred).csv', index=False)


# a, b = curve_fit(malthusFun_pro, index, pop, maxfev=1000000)[0]
plt.plot([train_year[-1]] + list(test_year), [train_pred[-1]] + list(test_pred), "-")
plt.plot(year, pop)
plt.plot(train_year, train_pred)
plt.axvline(year[-6], color='black', linestyle='--')
plt.legend(['预测值', '实际值', '拟合值'])
plt.xlabel('年份')
plt.ylabel('人口数量($\mathregular{10^6}$)')
figure1 = plt.gcf()
figure1.savefig("../fig/malthus/马尔萨斯指数增长模型({}).png".format(dataName))
plt.show()

plot_rela_pred(pop[:-5], train_pred, time=train_year, fig_savepath=r'../fig/malthus/{}(train).png'.format(dataName),
               measurement_time='a',
               measurement_unit='$\mathregular{10^6}$')
plot_rela_pred(pop[-5:], test_pred, time=test_year, fig_savepath=r'../fig/malthus/{}(test).png'.format(dataName),
               measurement_time='a',
               measurement_unit='$\mathregular{10^6}$')
