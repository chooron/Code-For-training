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
train_pop1 = pop1[:-5]
pop0 = train_pop1[0]

# %%
# ARIMA模型
from pmdarima import auto_arima

model = auto_arima(train_pop1, error_action='ignore', trace=True,
                   suppress_warnings=True, maxiter=10,
                   seasonal=False, m=12)
plt.plot(year[-5:], model.predict(5), "*")
plt.plot(year, pop1)
plt.show()