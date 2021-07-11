# 定义问题
# 3个站点,8个摊位,故一共有24个未知值,在计算中设置为[3,8]的维度
import pandas as pd
import numpy as np

distance = pd.read_csv('distance_result.csv')
distance = distance.drop(['start'], axis=1)
dis_pal = 2
need = [80, 70, 90, 80, 120, 70, 100, 90]
loss = [10, 8, 5, 10, 10, 8, 5, 8]


def objective(x):
    # 运费
    x = x.reshape(3, 8)
    p_array = x * distance.values
    p_cost = np.sum(p_array) * dis_pal
    # 损失
    real = np.sum(x, axis=0)
    l_cost = np.sum((need - real) * loss)
    total_cost = p_cost + l_cost
    return total_cost



