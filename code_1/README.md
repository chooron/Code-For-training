### 人口预测
-------------------------
选用方法：

- 马尔萨斯人口预测模型
- logistics人口预测模型
- GM11灰色系统模型
- ARIMA模型
- prophet模型
- VLSW-LSTM模型

结论：

- 马尔萨斯、灰色系统模型在人口预测中效果较差
- ARIMA和经过数据归一化过的prophet模型预测结果较好，logistics次之
- VLSW方法尽管能得到较多的数据集(60+)，但仍不能满足LSTM模型训练需求，故效果不好
