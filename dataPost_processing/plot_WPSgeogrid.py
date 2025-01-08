"""
Author: Yaohan Xian
GitHub: https://github.com/Sm0keNMirrors
Last update: 2025年1月8日
"""

import numpy as np
import netCDF4 as nc
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.io.shapereader import Reader
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os


def plot_WPSgeogrid(   # 绘制geogrid的嵌套结果，让用户检查是否合适并进行调整或者继续执行
    geo_em_files_dir = "",  # WPS输出的geo文件路径
    shpfiles_dir = "",
    geogrid_picout_dir = "",
):
    """
    可视化输出WPS中geogrid输出的geoem的嵌套层
    :param geo_em_files_dir:  所有domain的geoem所在的文件夹
    :param shpfiles_dir: shp文件所在路径
    :param geogrid_picout_dir: 输出的绘图结果所在的路径
    :return:
    """
    def getArrayVertices(ARR):
        top_left = ARR[0, 0]
        top_right = ARR[0, ARR.shape[1] - 1]
        bottom_left = ARR[ARR.shape[0] - 1, 0]
        bottom_right = ARR[ARR.shape[0] - 1, ARR.shape[1] - 1]
        return [top_left, top_right, bottom_left, bottom_right]

    shpfiles_pr = os.listdir(shpfiles_dir)
    shpfiles = []
    for f in shpfiles_pr:
        if f[-4:] == ".shp":
            shpfiles.append(shpfiles_dir+f)

    # --创建画图空间
    proj = ccrs.PlateCarree()  # 创建坐标系
    fig = plt.figure(figsize=(8, 6))  # 创建页面
    ax = fig.subplots(1, 1, subplot_kw={'projection': proj})

    geo_file_list_ = os.listdir(geo_em_files_dir)  # geo文件列表
    geo_file_list = []
    for x in geo_file_list_:
        if "geo_em" in x:
            geo_file_list.append(x)
    geo_file_list.sort()
    domains_n = len(geo_file_list)  # 嵌套区域数
    domain_rectangles = []
    re_color = ['yellow']*domains_n
    for i in geo_file_list:
        i_n = geo_file_list.index(i)
        geof = nc.Dataset(geo_em_files_dir + i)
        LAT = np.array(geof.variables['XLAT_C'][:][0])  # 读取纬度范围
        LON = np.array(geof.variables['XLONG_C'][:][0])  # 读取经度范围
        LAT_vert = getArrayVertices(LAT)  # 获取经纬度数组的四个顶点数据列表
        LON_vert = getArrayVertices(LON)
        RE = Rectangle((LON_vert[0], LAT_vert[0]), LON_vert[1] - LON_vert[0], LAT_vert[2] - LAT_vert[0], linewidth=1.3,
                       linestyle='-', zorder=2,
                       edgecolor=re_color[i_n], facecolor='none', transform=ccrs.PlateCarree())
        ax.text(LON_vert[0], LAT_vert[0], f"d0{i_n}", transform=ccrs.PlateCarree(), fontsize=15, c='k')
        if i_n == 0:
            d01_vert = [LON_vert[0], LON_vert[1], LAT_vert[0], LAT_vert[2]]
        domain_rectangles.append(RE)  # 将所有矩形存放数组

    for shp_dir in shpfiles:
        shp = cfeat.ShapelyFeature(Reader(shp_dir).geometries(), proj, edgecolor='k', facecolor='none')
        ax.add_feature(shp, lw=0.5, zorder=2)
    ax.set_extent(d01_vert)  # 可根据需求自行定义
    for i in domain_rectangles:
        ax.add_patch(i)
    # --设置网格点属性
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1.2, color='k', alpha=0.5, linestyle='--')
    plt.savefig(geogrid_picout_dir+"geogrid.png")


if __name__ == "__main__":
    plot_WPSgeogrid(
        geo_em_files_dir=r"E:\WPSoutput\\",
        shpfiles_dir=r"E:\CMAQdata_chengdu202208\shps\\",
        geogrid_picout_dir=r"E:\WPSoutput\\"
    )
    pass