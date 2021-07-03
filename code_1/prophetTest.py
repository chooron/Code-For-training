# facebook prophet库预测
import pandas as pd
from fbprophet import Prophet
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from code_1.gen_sample import load_data
from code_1.evaluate import evaluate
from utils.plot_utils import plot_rela_pred
import numpy as np

path = '../fig/prophet/'
plt.rcParams['font.sans-serif'] = ['SimHei']
dataName = '美国人口'
year, pop = load_data(dataName)
scaler = MinMaxScaler()
df = pd.read_csv("data.csv")[['年份', dataName]]
train_year = year[:-5]
test_year = year[-5:]

fake_year = df.iloc[0, 0] + range(0, len(df), 1)
fake_year = fake_year[:-5]
train_df = pd.DataFrame(fake_year, columns=['ds'])
for i, _ in enumerate(range(len(fake_year))):
    train_df.iloc[i, 0] = datetime(year=fake_year[i], month=1, day=1)

pop = pop.reshape(1, -1).T
train_pop = pop[:-5]
train_pop_scaled = scaler.fit_transform(train_pop)
train_df['y'] = train_pop_scaled
# train_df['cap'] = 1.5 zh
train_df['cap'] = 2.8

model = Prophet(growth='logistic')
model.fit(train_df)
future = model.make_future_dataframe(periods=5, freq='y')
# future['cap'] = 1.5 zh
future['cap'] = 2.8
future.tail()
result = model.predict(future)
pred = scaler.inverse_transform(result['yhat'].values.reshape(-1, 1))
train_pred = pred[:-5]
test_pred = pred[-5:]

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
figure1.savefig(path + dataName + "prophet模型({}).png".format(dataName))
plt.show()

plot_rela_pred(np.squeeze(pop[:-5]), np.squeeze(train_pred), time=train_year, fig_savepath=path + '{}(train).png'.format(dataName),
               measurement_time='a',
               measurement_unit='$\mathregular{10^6}$')
plot_rela_pred(np.squeeze(pop[-5:]), np.squeeze(test_pred), time=test_year, fig_savepath=path + '{}(test).png'.format(dataName),
               measurement_time='a',
               measurement_unit='$\mathregular{10^6}$')

