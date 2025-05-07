"""
Author: Yaohan Xian
GitHub: https://github.com/Sm0keNMirrors
Last update: 2025年1月8日
"""
import matplotlib.dates as mdates
import numpy as np
from matplotlib import font_manager, rcParams, pyplot as plt
from matplotlib.ticker import FuncFormatter
import random

def plot_bars(
    bardatas = [],
    ifstackbars = False,
    bar_width = 0.2,
    bar_labels = [],

    xlabels=[],

    fig_size=(5, 2),
    figaxe=[0.12, 0.2, 0.7, 0.7],
    x_y_labels = [],
    y_max_min=[],  #
    y_size=5,  #
    y_ticks_decimal=0,
    x_ticks_decimal=0,
    fontsize = 13,
    legendsize = None,
    sticksize = None,
    title = "",

    filename="",
    outputdir="",
    font_dir="",


):
    """
    绘制基本柱状图
    :param bardatas: 柱形高度，也就是y轴数值。
    :param ifstackbars: 布尔值，是否绘制堆叠柱状图，若绘制必须为二维数组或列表。
    :param bar_width: 柱宽度

    :param fig_size: 图像大小，matplotlib
    :param figaxe:  图像轴绘制范围
    :param x_y_labels: 图像绘制的xy名称
    :param y_max_min: Y轴的最大最小值，用于设置y轴限度
    :param y_size: 以Y轴范围划分为多少个提示虚线，默认5
    :param y_ticks_decimal: # 保留多少y刻度的小数位数，在数值很小时，刻度划分可能会因保留位数而不均匀。
    :param fontsize: 字体大小
    :param legendsize: legend字体大小，不设置时默认与fontsize相等
    :param sticksize: stick字体大小，不设置时默认与fontsize相等
    :param title: 图像标题

    :param filename: 图像保存的名称，不带拓展名。
    :param outputdir: 输出的路径。
    :param font_dir: 指定特殊绘图整体的字体类型，如想中英文同时存在必须导入times+simsun.ttf，不指名则为matplotlib默认字体。

    :param xlabels: 每个柱对应的labels，字符串列表。
    :return:
    """

    def format_y_ticks(y,pos):
        if y_ticks_decimal == 100:
            return f'{y*100}%'  # 特殊情形 显示为百分率
        else:
            return f'{y:.{y_ticks_decimal}f}'  # 保留x位小数

    def format_x_ticks(x,pos):
        if x_ticks_decimal == 100:
            return f'{x*100}%'  # 特殊情形 显示为百分率
        else:
            return f'{x:.{x_ticks_decimal}f}'  # 保留x位小数

    def generate_random_color_codes(n):
        color_codes = []
        for _ in range(n):
            # 随机生成6个十六进制数，以组成颜色代号
            color = '#' + ''.join(random.choices('0123456789ABCDEF', k=6))
            color_codes.append(color)
        return color_codes

    # 字体设置
    if font_dir != "":
        font_dir = r"D:\Fonts\times+simsun.ttf"
        font_manager.fontManager.addfont(font_dir)
        prop = font_manager.FontProperties(fname=font_dir)
        rcParams['font.sans-serif'] = prop.get_name()  # 根据名称设置字体
    else:
        prop = None
        rcParams['font.family'] = 'SimHei'  # 中文字体
        rcParams['font.family'] = 'sans-serif'  # 使用字体中的无衬线体
    rcParams['font.size'] = fontsize  # 设置字体大小
    rcParams['axes.unicode_minus'] = False  # 使坐标轴刻度标签正常显示正负号

    fig = plt.figure(figsize=fig_size, dpi=200)  # (8, 3)
    ax = fig.add_axes(figaxe)  # [0.15, 0.2, 0.73, 0.7]

    # bar_locations = [(bar_width + bar_width / 2)]* len(xlabels)
    xticks_name = xlabels  # 横坐标区域名称
    max_calcu_flag = False
    if y_max_min == []: max_calcu_flag = True
    if ifstackbars == False: # 不堆叠
        bar = ax.bar(xticks_name, bardatas, color=generate_random_color_codes(1),width = bar_width,label=bar_labels[0])
        if max_calcu_flag == True:  # 不设立时自动计算
            y_max_min = [np.max(bardatas), 0]
    else:
        barcolors = generate_random_color_codes(len(bardatas))
        for bardata in bardatas:
            index = bardatas.index(bardata)
            print(barcolors,bardatas,bar_labels)
            if index == 0 : # 首次 无bottom
                bar = ax.bar(xticks_name, bardatas[index], color=barcolors[index], width=bar_width,
                             label=bar_labels[index])
                bottom = np.array(bardatas[index])
            else:
                bar = ax.bar(xticks_name, bardatas[index],bottom=bottom, color=barcolors[index], width=bar_width,
                             label=bar_labels[index]) # 上一组数据作为bottom

                bottom += np.array(bardatas[index])
                if max_calcu_flag == True:  # 不设立时自动计算
                    y_max_min = [np.max(bottom),0]


    max_value = y_max_min[0]
    min_value = y_max_min[1]
    size = y_size
    step = (max_value - min_value) / (size - 1) if size > 1 else 0
    y_ = [round(min_value + step * i, y_ticks_decimal) for i in range(size)]

    if legendsize == None: legendsize = fontsize
    ax.legend(loc='upper left', fontsize=legendsize, bbox_to_anchor=(1, 1))

    ax.set_ylabel(x_y_labels[1], fontsize=fontsize)
    ax.set_xlabel(x_y_labels[0], fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.ylim(0, max_value)  # y轴 高度范围
    plt.yticks(y_)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_y_ticks))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_x_ticks))
    yticks = plt.yticks()[0]
    # 为每个y轴刻度画虚线
    for ytick in yticks:
        plt.axhline(y=ytick, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
    plt.xticks(xlabels[::3],fontsize=fontsize)  # xticks必须在这个位置才生效
    plt.yticks(fontsize=fontsize)
    plt.savefig(outputdir + f"{filename}.png")



if __name__ == "__main__":
    pass

    plot_bars(
        bardatas=[1,2,3,4,5],
        xlabels=['a','b','c','d','e'],
        bar_width = 0.5,
        bar_labels = ['实例1'],
        ifstackbars= False,
        x_y_labels=['X轴','y轴'],
        y_max_min = [5,1],
        title="单bar测试",
        filename="test1",
        fontsize=9,
        outputdir=r"E:\xyhfiles_runtest\\",
        font_dir=r"D:\Fonts\times+simsun.ttf",
    )

    plot_bars(
        bardatas=[[1,2,3,4,5],[1,2,1,1,1],[2,3,1,2,3]],
        xlabels=['a','b','c','d','e'],
        bar_width = 0.5,
        bar_labels = ['实例1','实例2','实例3'],
        ifstackbars= True,
        x_y_labels=['X轴','y轴'],
        title="多bar堆叠测试",
        filename="testmore",
        y_ticks_decimal = 100,
        fontsize=6,
        legendsize = 6.5,
        outputdir=r"E:\xyhfiles_runtest\\",
        font_dir=r"D:\Fonts\times+simsun.ttf",
    )
