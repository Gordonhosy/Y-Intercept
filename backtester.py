import matplotlib.pyplot as plt
import numpy as np

class backtester:
    
    def __init__(self, ticker, r_log_ret, position):
        self.ticker = ticker
        self.r_log_ret = r_log_ret[ticker]
        self.position = position[ticker]
        self.daily_log_ret = self.cal_daily_log_ret()
        
    
    def cal_daily_log_ret(self):
        daily_log_ret = self.position.shift(1)*self.r_log_ret
        if type(self.ticker) == str:
            return daily_log_ret.dropna()
        else:
            return (daily_log_ret.sum(axis = 1)).dropna()

    def cal_sharpe(self, N = 252, rf = 0):
        mean = np.mean(self.daily_log_ret)*N - rf
        sigma = np.std(self.daily_log_ret)*np.sqrt(N)
        return mean/sigma
    
    def plot_daily_lr(self, figsize = 6):
        plt.figure(figsize = figsize*np.array([1.618, 1]))
        plt.xlabel("Date")
        plt.ylabel("Log_Return")
        plt.title("Daily Log Return")
        plt.plot(self.daily_log_ret)
        plt.xticks(self.r_log_ret.index, rotation = 45)
        plt.locator_params(axis='x', nbins=10)
        
    def plot_cum_lr(self, figsize = 6):
        plt.figure(figsize = figsize*np.array([1.618, 1]))
        plt.xlabel("Date")
        plt.ylabel("Log_Return")
        plt.title("Cumulative Log Return")
        plt.plot(self.daily_log_ret.cumsum())
        plt.xticks(self.r_log_ret.index, rotation = 45)
        plt.locator_params(axis='x', nbins=10)