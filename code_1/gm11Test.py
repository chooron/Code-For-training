# %%
# GM11
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