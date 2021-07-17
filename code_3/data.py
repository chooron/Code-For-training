# 数据分析
import matplotlib.pyplot as plt

plt.style.use('ggplot')
import pandas as pd
from scipy import stats
import numpy as np

record = [459, 362, 624, 542, 509, 584, 433, 748, 815, 505, 612, 452, 434, 982, 640, 742, 565, 706, 593, 680, 926, 653,
          164, 487, 734, 608, 428, 1153, 593, 844, 527, 552, 513, 781, 474, 388, 824, 538, 862, 659, 775, 859, 755, 649,
          697, 515, 628, 954, 771, 609, 402, 960, 885, 610, 292, 837, 473, 677, 358, 638, 699, 634, 555, 570, 84, 416,
          606, 1062, 484, 120, 447, 654, 564, 339, 280, 246, 687, 539, 790, 581, 621, 724, 531, 512, 577, 496, 468, 499,
          544, 645, 764, 558, 378, 765, 666, 763, 217, 715, 310, 851]
record = np.array(record)
avg = np.average(record)
var = np.sqrt(np.var(record))



fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(2, 1, 1)
s = pd.DataFrame(record)
s.hist(bins=30, alpha=0.5, ax=ax1, log=True)
s.plot(kind='kde', secondary_y=True, ax=ax1)
plt.grid()

ax2 = fig.add_subplot(2, 1, 2)
stats.probplot(record, dist="norm", plot=plt)
plt.grid()

plt.hist(record, bins=20, log=True)
plt.hist(record, kind='kde')

plt.savefig('distrbution.png')
plt.show()

