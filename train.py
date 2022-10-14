import numpy as np
import pandas as pd

def cal_sharpe_df(df, series_name, N=365, rf=0):
    mean = df[series_name].mean()*N -rf
    sigma = df[series_name].std()*np.sqrt(N)
    return mean/sigma

def test(df, func, trad_name="PX_LAST", window_param=[10, 60, 10], thred_param =[0.25, 5, 0.5], onlyL = False, trading_resolution =252, MM = True, cost = 0,start_date = None, end_date = None):
    result_list = []
    for window in list(np.arange(window_param[0], window_param[1], window_param[2])):
        for thred in list(np.arange(thred_param[0], thred_param[1], thred_param[2])):
            temp = func(df, "Signal", "Pos", trad_name, cost, window, thred, MM=MM, onlyL=onlyL)
            temp = temp.loc[start_date:end_date]
            sharpe = cal_sharpe_df(temp, "Pnl", trading_resolution)
            result_list.append([window, thred, sharpe])

    return result_list

