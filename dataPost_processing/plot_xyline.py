from matplotlib import pyplot as plt
import datetime
import os
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib import font_manager, rcParams, pyplot as plt
from matplotlib.ticker import FuncFormatter



def plot_xyline(
    xdata = [],
    ydata = [],
    ydata_ifscatter = [], #
    data_labels = [], #
    data_linewidths = [],
    data_linestyles = [],
    y_max_min=[], #
    y_size = 5, #
    y_ticks_decimal = 2,
    fig_size = (5, 2),
    figaxe = [0.12, 0.2, 0.7, 0.7],
    x_y_labels = [],
    title = "",
    filename = "",
    outputdir = "",
    xdata_dateformat = '', #
    xdata_numberformat = 0, #
    xdata_customticks = [],
):
    """
    :param xdata: x轴数据，每个都要有
    :param ydata: y轴数据，对应每个x
    :param ydata_ifscatter: 是否某个y数据用scatter点图表示，参数格式为['True','False']，对应每个是否改变
    :param data_labels: 每条线数据的label，按照data数据顺序
    :param data_linewidths: 线条宽度 []，每个对应
    :param data_linestyles: 线条样式 []，每个对应
    :param y_max_min: Y轴的最大最小值，用于设置y轴限度
    :param y_size: 以Y轴范围划分为多少个提示虚线，默认5
    :param y_ticks_decimal: # 保留多少y刻度的小数位数，在数值很小时，刻度划分可能会因保留位数而不均匀
    :param fig_size: 图像大小，matplotlib
    :param figaxe:  图像轴绘制范围
    :param x_y_labels: 图像绘制的xy名称
    :param title: 图像标题
    :param filename: 图像保存的名称 不带拓展名
    :param outputdir: 输出的路径
    :param xdata_dateformat: 若X轴数据是日期，设置其显示格式 如%m-%d 可选参数
    :param xdata_numberformat: 若x轴数据是数值，设置其ticks数值间隔 可选参数
    :param xdata_customticks: 让x轴制定显示几个信息，如31*7个x轴数据，显示为[2002, 2005, 2008, 2011, 2014, 2017, 2020]，输入参数格式为[xdata(一个),显示信息个数，[显示信息数组]]

    :return:
    """


    def format_y_ticks(y, pos):
        return f'{y:.{y_ticks_decimal}f}'  # 保留两位小数
    # 字体设置
    font_path = r"D:\Fonts\times+simsun.ttf"
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    rcParams['font.family'] = 'sans-serif'  # 使用字体中的无衬线体
    rcParams['font.sans-serif'] = prop.get_name()  # 根据名称设置字体
    rcParams['font.size'] = 13  # 设置字体大小
    rcParams['axes.unicode_minus'] = False  # 使坐标轴刻度标签正常显示正负号

    max_value = y_max_min[0]
    min_value = y_max_min[1]
    size = y_size
    step = (max_value - min_value) / (size - 1) if size > 1 else 0
    y_ = [round(min_value + step * i, y_ticks_decimal) for i in range(size)]

    fig = plt.figure(figsize=fig_size, dpi=200) # (8, 3)
    ax = fig.add_axes(figaxe) # [0.15, 0.2, 0.73, 0.7]

    lines = []
    for i in range(len(xdata)):
        if ydata_ifscatter[i] == True: # 判断当前序号数据是画线图还是点
            linex = plt.scatter(xdata[i], ydata[i], s=data_linewidths[i], linewidth=data_linewidths[i], label=data_labels[i],
                                c='red')  # 缺失值空缺太多，会导致line画不出来，则使用scatter散点图
        else:
            linex, = ax.plot(
                xdata[i], ydata[i],
                linewidth=data_linewidths[i], label=data_labels[i], linestyle=data_linestyles[i])  # 要用legend画图例，这里必须,=
        lines.append(linex)

    plt.legend(handles=lines, loc='upper left', fontsize=7, bbox_to_anchor=(1, 1))
    ax.set_ylabel(x_y_labels[1], fontsize=7)
    ax.set_xlabel(x_y_labels[0], fontsize=7)
    plt.title(title, fontsize=7)
    plt.ylim(min_value, max_value)  # y轴 高度范围
    plt.yticks(y_)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_y_ticks))
    yticks = plt.yticks()[0]
    # 为每个y轴刻度画虚线
    for ytick in yticks:
        plt.axhline(y=ytick, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
    plt.xticks(fontsize=7)  # xticks必须在这个位置才生效
    plt.yticks(fontsize=7)
    if xdata_dateformat != '': ax.xaxis.set_major_formatter(mdates.DateFormatter(xdata_dateformat))  # 设置日期显示间隔
    if xdata_numberformat != 0: plt.xticks(np.arange(0, len(xdata[0])+1, xdata_numberformat))  # 设置数值显示间隔
    if xdata_customticks != []:
        custom_ticks = np.linspace(0, len(xdata_customticks[0]) - 1, xdata_customticks[1], dtype=int)  # 选择7个位置
        year_labels = xdata_customticks[2]
        plt.xticks(custom_ticks, year_labels)
    plt.savefig(outputdir + f"{filename}.png")


if __name__ == "__main__":
    pass
