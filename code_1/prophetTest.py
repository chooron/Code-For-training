# facebook prophet库预测
import pandas as pd
from fbprophet import Prophet
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']

scaler = MinMaxScaler()
df = pd.read_csv("data.csv")[['年份', '美国人口']][:-5]
df['ds'] = datetime
pop = df['美国人口'].values.reshape(1, -1).T
pop_scaled = scaler.fit_transform(pop)
for i, d in enumerate(range(len(df))):
    df.iloc[i, 2] = datetime(year=df.iloc[i, 0], month=1, day=1)
df['y'] = pop_scaled
df = df.drop(['年份', '美国人口'], axis=1)
df['cap'] = 8.5
model = Prophet(growth='logistic')
model.fit(df)
future = model.make_future_dataframe(periods=50, freq='y')
future['cap'] = 8.5
future.tail()
result = model.predict(future)
pred = scaler.inverse_transform(result['yhat'].values.reshape(-1, 1))
plt.plot(df['ds'], pop)
plt.plot(result['ds'],pred)
plt.legend(['实际值','预测值'])
plt.title("美国人口预测")
plt.show()
