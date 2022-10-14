import numpy as np
import pandas as pd


def MA(data, signal_name, pos_name, ret_name, cost, window, thre, signal_2=False, signal_2_name=None, thre_2=None,
       MM=False, onlyL=False, Pnl=True, pos_series=False):

    if signal_2:
        data = data[[signal_name, signal_2_name, ret_name]]
    else:
        data = data[[signal_name, ret_name]]

    data["MA"] = data[signal_name].rolling(window=window).mean()
    data = data.dropna()
    dataArray = data.values
    pos_list = []

    if signal_2:
        signal_num = 3
    else:
        signal_num = 2

    for i in dataArray:
        if signal_2:
            if i[signal_num] > thre and i[1] < thre_2:
                pos_list.append(-1)
            elif i[signal_num] < -thre and i[1] < thre_2:
                pos_list.append(1)
            else:
                pos_list.append(0)
        else:
            if i[signal_num] > thre:
                pos_list.append(-1)
            elif i[signal_num] < -thre:
                pos_list.append(1)
            else:
                pos_list.append(0)

    data[pos_name] = pos_list

    if MM:
        data[pos_name] = data[pos_name] * -1

    if onlyL:   
        data[pos_name].replace({-1: 0}, inplace=True)

    if Pnl:
        data["Pnl"] = data[pos_name].shift(1) * data[ret_name] - abs(
            data[pos_name] - data[pos_name].shift(1)) * cost
        return data
    elif pos_series:
        return data[pos_name].tolist()
    else:
        return data