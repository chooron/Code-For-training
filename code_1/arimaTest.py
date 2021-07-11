import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from code_1.evaluate import evaluate
from code_1.gen_sample import load_data
from code_1.utils.plot_utils import plot_rela_pred

plt.rcParams['font.sans-serif'] = ['SimHei']
path = 'fig/arima/'
# %%
# read data
dataName = "美国人口"
year, pop = load_data(dataName)
train_year = year[:-5]
test_year = year[-5:]

index = year - year[0]
train_index = index[:-5]
train_pop = pop[:-5]
test_pop = pop[-5:]

# %%
# ARIMA模型
from pmdarima import auto_arima

model = auto_arima(train_pop, error_action='ignore', trace=True,
                   suppress_warnings=True, maxiter=100,
                   seasonal=False, m=12)

train_pred = model.arima_res_.fittedvalues
test_pred = model.predict(5)

total_pred = np.concatenate((train_pred, test_pred))
df = pd.DataFrame()
df['year'] = year
df['real'] = pop
df['pred'] = total_pred
df.to_csv(path + dataName + '(pred).csv', index=False)

evaluate(pop[:-5], train_pred, pop[-5:], test_pred, path=path + dataName + '(index).csv')
plt.plot([train_year[-1]] + list(test_year), [train_pred[-1]] + list(test_pred), "-")
plt.plot(year, pop)
plt.plot(train_year, train_pred)
plt.axvline(year[-6], color='black', linestyle='--')
plt.legend(['预测值', '实际值', '拟合值'])
plt.xlabel('年份')
plt.ylabel('人口数量($\mathregular{10^6}$)')
figure1 = plt.gcf()
figure1.savefig(path + dataName + "ARIMA模型({}).png".format(dataName))
plt.show()

total_pred = np.concatenate((train_pred, test_pred))
df = pd.DataFrame()
df['year'] = year
df['real'] = pop
df['pred'] = total_pred
df.to_csv(path + dataName + '(pred).csv', index=False)

plot_rela_pred(pop[:-5], train_pred, time=train_year, fig_savepath=path + '{}(train).png'.format(dataName),
               measurement_time='a',
               measurement_unit='$\mathregular{10^6}$')
plot_rela_pred(pop[-5:], test_pred, time=test_year, fig_savepath=path + '{}(test).png'.format(dataName),
               measurement_time='a',
               measurement_unit='$\mathregular{10^6}$')
