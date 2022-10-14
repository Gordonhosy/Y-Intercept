import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from backtester import backtester
from strat import *

data = pd.read_csv("data.csv")

ticker_dict = {}

for ticker in data.ticker.unique():
        data_tmp = data.loc[data["ticker"] == ticker]
        data_tmp['ret'] = data_tmp['last'].pct_change()
        data_tmp['log_ret'] = np.log(data_tmp['last']) - np.log(data['last'].shift(1))
        data_tmp['signal'] = data_tmp['log_ret']
        ticker_dict[ticker] = data_tmp.set_index('date')


ticker_test = '1332 JT'
test = ticker_dict[ticker_test]
x = MA(test, "signal", "pos", "log_ret", 0, window=10, thre=0)


a = backtester(ticker_test, ticker_dict[ticker_test].dropna().log_ret, x.pos)
np.mean(a.daily_log_ret)
b = a.cal_sharpe()
a.plot_cum_lr(figsize = 4)
