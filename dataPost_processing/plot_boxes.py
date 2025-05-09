
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams, pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates

def plot_boxes(
    boxes_datas = [],
    boxes_labels = [],

    if_lineboxex = False,

    x_y_labels = [],
    box_props = dict(facecolor='lightblue', color='blue'),

    fig_size = (5, 2),
    figaxe = [0.12, 0.2, 0.7, 0.7],
    fontsize=13,

    title = "",
    filename="",
    outputdir="",
    font_dir="",

    xdata_dateformat = '', #
    xdata_numberformat = 0, #
    xdata_customticks = [],
):
    """

    :param if_lineboxex: 是否改为绘制由平均值线条，和25 75分位值作为上下限fill的图像，
    :param box_props:
    :param fontsize:
    :param x_y_labels:
    :param boxes_datas: 每个箱线图的数据，数组列表 列表的列表，不能有NAN 否则无法绘图
    :param fig_size:
    :param figaxe:
    :param boxes_labels: 每个箱线图的横坐标表示
    :return:
    """

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

    if if_lineboxex == False:
        boxex = ax.boxplot(
            boxes_datas,
            notch=False,
            patch_artist=True,
            labels=boxes_labels,
            boxprops = box_props,
        )
    else:
        mean_values = np.nanmean(boxes_datas, axis=1) # 均值
        print(boxes_datas,mean_values)
        q25 = np.nanpercentile(boxes_datas, 25, axis=1) # 25 75分位
        q75 = np.nanpercentile(boxes_datas, 75, axis=1)
        plt.plot(boxes_labels, mean_values, label='Mean', color='blue',linewidth=0.7)
        # 填充上下四分位区间
        plt.fill_between(boxes_labels, q25, q75, color='lightblue', alpha=0.4, label='25% - 75% Range')
        if xdata_dateformat != '': ax.xaxis.set_major_formatter(mdates.DateFormatter(xdata_dateformat))  # 设置日期显示间隔
        if xdata_numberformat != 0: plt.xticks(np.arange(0, len(boxes_datas[0]) + 1, xdata_numberformat))  # 设置数值显示间隔

    # 添加标题和标签
    plt.title(title)
    ax.set_ylabel(x_y_labels[1], fontsize=fontsize)
    ax.set_xlabel(x_y_labels[0], fontsize=fontsize)

    # 显示图形
    plt.grid(axis='y')  # 添加网格
    plt.savefig(outputdir + f"{filename}.png")
