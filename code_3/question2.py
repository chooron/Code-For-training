# 问题2
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sko import GA, SA, PSO
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')

t = 20
f = 200
k = 1500
d = 4000
h = 2000
record = [459, 362, 624, 542, 509, 584, 433, 748, 815, 505, 612, 452, 434, 982, 640, 742, 565, 706, 593, 680, 926, 653,
          164, 487, 734, 608, 428, 1153, 593, 844, 527, 552, 513, 781, 474, 388, 824, 538, 862, 659, 775, 859, 755, 649,
          697, 515, 628, 954, 771, 609, 402, 960, 885, 610, 292, 837, 473, 677, 358, 638, 699, 634, 555, 570, 84, 416,
          606, 1062, 484, 120, 447, 654, 564, 339, 280, 246, 687, 539, 790, 581, 621, 724, 531, 512, 577, 496, 468, 499,
          544, 645, 764, 558, 378, 765, 666, 763, 217, 715, 310, 851]
record = np.array(record)
avg = np.average(record)
var = np.sqrt(np.var(record))


def objective(x):
    m = int(x[0])
    n = int(x[1])
    p1 = 1 - norm.cdf(m, loc=avg, scale=var) / 0.95
    w1 = t * (m / n) + k + 0.02 * h * m / n + 0.02 * m * f
    Ew1 = w1 * p1
    Eq1 = m * p1 * 0.98
    Ew2 = 0
    Eq2 = 0
    for i in range(1, m):
        num = np.floor(i / n) + 1
        w2 = t * num + d + 0.02 * f * i + 0.6 * f * (n * num - i) + 0.02 * h * i / n
        p2 = (1 / (np.sqrt(2 * np.pi) * var)) * np.exp(-((i + 1 - avg) ** 2) / (2 * var * var)) / 0.95
        Ew2 += w2 * p2
        Eq2 += (0.98 * i + (n * num - i) * 0.4) * p2
    Ew = Ew1 + Ew2
    Eq = Eq1 + Eq2
    return Ew / Eq



# %%
pso_model = PSO.PSO(func=objective, n_dim=2, lb=1, ub=1000, verbose=True, max_iter=50)
history = pd.DataFrame(pso_model.gbest_y_hist)
plt.plot(history.index,history.values.squeeze(),'-*')
plt.legend(['value'])
plt.savefig('q2_mini.png')
plt.show()
pso_x, pso_y = pso_model.run()
best_m = int(pso_x[0])
best_n = int(pso_x[1])