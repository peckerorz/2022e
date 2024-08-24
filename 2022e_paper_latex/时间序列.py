import pandas as pd
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.stattools import adfuller as ADF
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import itertools
import numpy as np
import seaborn as sns
#6004010256,6004020375,6004020887,6004020918,6004020524,6004020656
#filename=['6004010256.xlsx','6004020375.xlsx','6004020887.xlsx','6004020918.xlsx','6004020504.xlsx','6004020656.xlsx']
filename=['6004020375.xlsx']
#获取数据

# 设置文件目录
input_dir = 'output_materials'
# 遍历文件目录中的每个文件

for file_name in os.listdir(input_dir):
    if file_name in filename:
        print(f'{file_name}')
        # 读取预测编码对应的Excel文件
        file_path = os.path.join(input_dir, file_name)
        df = pd.read_excel(file_path)
        time_series=df['周数']
        train=df['需求量'][:70]
        test=df['需求量'][70:100]

        # .diff(1)做一个时间间隔
        diff_1 = train.diff(1)  # 1阶差分
        # 对一阶差分数据在划分时间间隔
        diff_2 = diff_1.diff(1)  # 2阶差分
        """
        plt.figure(figsize=(12, 6))
        plt.plot(train)
        plt.xticks(rotation=45)  # 旋转45度
        plt.show()

        # .diff(1)做一个时间间隔
        diff_1 = train.diff(1)  # 1阶差分
        # 对一阶差分数据在划分时间间隔
        diff_2 = diff_1.diff(1)  # 2阶差分

        fig = plt.figure(figsize=(12, 10))
        # 原数据
        ax1 = fig.add_subplot(311)
        ax1.plot(train)
        # 1阶差分
        ax2 = fig.add_subplot(312)
        ax2.plot(diff_1)
        # 2阶差分
        ax3 = fig.add_subplot(313)
        ax3.plot(diff_2)
        plt.show()

        # 计算原始序列、一阶差分序列、二阶差分序列的单位根检验结果
        diff_1 = diff_1.fillna(0)
        diff_2 = diff_2.fillna(0)

       
        timeseries_adf = ADF(train.tolist())
        timeseries_diff1_adf = ADF(diff_1.tolist())
        timeseries_diff2_adf = ADF(diff_2.tolist())

        # 打印单位根检验结果
        print('timeseries_adf : ', timeseries_adf)
        print('timeseries_diff1_adf : ', timeseries_diff1_adf)
        print('timeseries_diff2_adf : ', timeseries_diff2_adf)

        # 绘制
        fig = plt.figure(figsize=(12, 7))

        ax1 = fig.add_subplot(211)
        fig = sm.graphics.tsa.plot_acf(train, lags=20, ax=ax1)
        ax1.xaxis.set_ticks_position('bottom')  # 设置坐标轴上的数字显示的位置，top:显示在顶部  bottom:显示在底部
        # fig.tight_layout()

        ax2 = fig.add_subplot(212)
        fig = sm.graphics.tsa.plot_pacf(train, lags=20, ax=ax2)
        ax2.xaxis.set_ticks_position('bottom')
        # fig.tight_layout()
        plt.show()"""

        # 确定pq的取值范围
        p_min = 0
        d_min = 0
        q_min = 0
        p_max = 5
        d_max = 0
        q_max = 5

        # Initialize a DataFrame to store the results,，以BIC准则
        results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min, p_max + 1)],
                                   columns=['MA{}'.format(i) for i in range(q_min, q_max + 1)])
        for p, d, q in itertools.product(range(p_min, p_max + 1),
                                         range(d_min, d_max + 1),
                                         range(q_min, q_max + 1)):
            if p == 0 and d == 0 and q == 0:
                results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
                continue
            try:
                model = sm.tsa.ARIMA(train, order=(p, d, q),
                                     # enforce_stationarity=False,
                                     # enforce_invertibility=False,
                                     )
                results = model.fit()
                results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.bic
            except:
                continue

        # 得到结果后进行浮点型转换
        results_bic = results_bic[results_bic.columns].astype(float)
        # 绘制热力图
        fig, ax = plt.subplots(figsize=(10, 8))
        ax = sns.heatmap(results_bic,
                         mask=results_bic.isnull(),
                         ax=ax,
                         annot=True,
                         fmt='.2f',
                         cmap="Purples"
                         )

        ax.set_title('BIC')
        plt.show()
        print(results_bic.stack().idxmin())
        p=0
        d=1
        q=1
        train_results = sm.tsa.arma_order_select_ic(train, ic=['aic', 'bic'], trend='n', max_ar=8, max_ma=8)

        print('AIC', train_results.aic_min_order)
        print('BIC', train_results.bic_min_order)

        model = sm.tsa.ARIMA(train, order=(p, d, q))
        """        results = model.fit()
        forecast = results.forecast(steps=5)"""
        # 假设model是已经拟合好的ARIMA模型
        results = model.fit()
        resid = results.resid  # 获取残差

        # 绘制
        # 查看测试集的时间序列与数据(只包含测试集)
        fig, ax = plt.subplots(figsize=(12, 5))

        ax = sm.graphics.tsa.plot_acf(resid, lags=1, ax=ax)

        plt.show()
        predict_sunspots = results.predict(dynamic=False)
        print(predict_sunspots)
        # 查看测试集的时间序列与数据(只包含测试集)
        plt.figure(figsize=(12, 6))
        plt.plot(test)
        plt.xticks(rotation=45)  # 旋转45度
        plt.plot(predict_sunspots)
        plt.show()
