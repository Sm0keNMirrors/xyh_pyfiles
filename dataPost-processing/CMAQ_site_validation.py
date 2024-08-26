import datetime

import netCDF4 as nt
import os
import h5py
import numpy as np
import xlrd
import xlwt
import pandas as pd

import xlutils.copy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import netCDF4 as nc
from matplotlib import ticker
from windrose import WindroseAxes
# from WRF_CMAQ.packcodes import WRFCMAQ_ToolFunctions as WCt


def getNearestPos(station_lat, station_lon, XLAT, XLONG):
    """
    得到距离站点最近的经纬度索引值
    :param station_lat:
    :param station_lon:
    :param XLAT:
    :param XLONG:
    :return:
    """
    difflat = station_lat - XLAT  # 经纬度数组与站点经纬度相减，找出最小的
    difflon = station_lon - XLONG
    rad = np.multiply(difflat, difflat) + np.multiply(difflon, difflon)  # difflat * difflat + difflon * difflon 计算最小的距离
    aa = np.where(rad == np.min(rad))  # 查询最小的距离的点在哪里，也就是站点位置
    ind = np.squeeze(np.array(aa))

    return ind

def generate_date_list(start_date: datetime, n: int) -> list:
    """从 start_date 开始，生成 n 天的日期列表"""
    return [start_date + datetime.timedelta(days=i) for i in range(n)]

def getAirStationsFromLatLon(uplat, downlat, leftlon, rightlon, station_info_csv=""):
    """
    从站点信息xls文件中，返回在某一个矩形经纬度范围内的站点信息，格式为验证代码中的字典格式
    :param uplat:
    :param downlat:
    :param leftlon:
    :param rightlon:
    :return: airStations: 空气质量站点信息
    """
    # airStation_infofile_dir = r"E:\全国空气质量\_站点列表\站点信息2016起.xls"
    airStation_infofile_dir = station_info_csv
    airStation_infofile = xlrd.open_workbook(airStation_infofile_dir)  # 修复了8月10日气象站数据有确实的问题，导致了绘图的错误
    sheet = airStation_infofile.sheet_by_index(0)
    airStations = {}

    for line in range(1, sheet.nrows):
        stationinfo = sheet.row_values(line, 0, 5)
        if stationinfo[3] != '':
            if leftlon < float(stationinfo[3]) < rightlon and downlat < float(stationinfo[4]) < uplat:
                airStations.update({stationinfo[1]: [float(stationinfo[3]), float(stationinfo[4]), stationinfo[0],stationinfo[2]]})
                                        # 数据格式：站点名：经度 纬度 站点代号 所在城市
    return airStations

def CMAQ_site_validation(
    start_date='YYYY-MM-DD', # 验证的开始时期，确定观测值数据获取
    daycount=5, # 验证持续时间，
    simdata_inithour=16, # 模拟值数据列表的起始位置，由于模拟数据为utc0，则以第16为数组的开始位置才能完整的取到一天，可修改
    GRIDCRO2D_file_dir = "",
    Combine_file_dir = "",
    target_substances = [], # 目标污染物，若输入多个，则是几个污染物类型的和，如不同ISAMtag相加的总量
    target_substance_obs = '', # 观测数据中的污染物名称，用于获取，不同于CMAQ的输出nc，一般验证只会获取一类
    Molar_mass = 0, # ppm为单位的物质的摩尔质量，为0时则不进行转换，如PM2.5，直接为ug
    airstation_files_dir = "",
    airstation_infofile_dir = "",
    out_dir = "",
    result_csv_name = "",
    suffix = ""
):
    """
    使用全国公开csv逐小时基本污染物逐小时浓度数据验证CAMQ模拟结果
    :param start_date:
    :param daycount:
    :param GRIDCRO2D_file_dir:
    :param Combine_file_dir:
    :param target_substances:
    :param airstation_files_dir:
    :param airstation_infofile_dir:
    :param out_dir:
    :param result_csv_name:
    :param suffix:
    :return:
    """
    year = str(datetime.datetime.strptime(start_date, '%Y-%m-%d')).split('-')[0] # 年份
    CMAQGRIDCRO2D_file_dir = GRIDCRO2D_file_dir  # CMAQ经纬度对应网格文件，用于得到站点位置
    CMAQoutISAMCombine_file_dir = Combine_file_dir  # 合并后的污染物浓度文件
    # airStation_file_dir = f"{airstation_files_dir}\\站点_{year}0101-{year}1231\\"  # 目标天的空气质量站点数据所在文件夹
    if os.path.exists(out_dir) is False: os.mkdir(out_dir)

    GRIDCRO2D = nc.Dataset(CMAQGRIDCRO2D_file_dir)
    CMAQoutf = nt.Dataset(CMAQoutISAMCombine_file_dir, "r")  # 打开CMAQ输出的NC格式文件
    var_lon = np.array(GRIDCRO2D.variables['LON'][:][0])
    var_lat = np.array(GRIDCRO2D.variables['LAT'][:][0])
    lonmin, latmax, lonmax, latmin = (var_lon.min(), var_lat.max(),
                                      var_lon.max(), var_lat.min())
    airStation_locations = getAirStationsFromLatLon(latmax, latmin, lonmin, lonmax,airstation_infofile_dir)  # 获得模拟区域内的所有站点信息

    labels = ['站点', 'MAE', 'R', 'IOA', 'NMB', 'NME','MFE', 'MFB','FE', 'FB', '站点经度', '站点纬度', '城市']
    result_csv_data = pd.DataFrame(columns=labels) # 创建记录结果的csv

    airStation_csv_list = [] # 验证时间内的所有csv
    validation_dates = generate_date_list(datetime.datetime.strptime(start_date, '%Y-%m-%d'),daycount)
    for date in validation_dates:
        filedate = f'{str(date).split("-")[0]}{str(date).split("-")[1]}{str(date).split("-")[2]}'
        airStation_csv_list.append(f"{airstation_files_dir}china_sites_{filedate}.csv")


    xls_write_num = -1  # 写入xls的验证参数的行，数据有效的时候才+1，否则会有空行
    for airstation in airStation_locations.values(): # 从站点列表处理每一个站点
        stname = list(airStation_locations.keys())[list(airStation_locations.values()).index(airstation)]  # 通过值k获取字典dic对应键的公式： list(dic.keys())[list(dic.values()).index(k)]
        # 读取经纬度与网格关系数据
        CMAQXLAT = np.array(GRIDCRO2D.variables['LAT'][:][0])
        CMAQXLONG = np.array(GRIDCRO2D.variables['LON'][:][0])
        nearpos = getNearestPos(airstation[1], airstation[0], CMAQXLAT, CMAQXLONG)  # 从WRF得到站点格子
        nearlat = nearpos[1]  # 与WRF获取最近点不同
        nearlon = nearpos[2]

        pollution1_station_pre = []
        subdata_shape = np.array(CMAQoutf.variables[target_substances[0]][:]).shape  # 获得数据shape来存放all
        substance = np.zeros(subdata_shape,'float64')
        for sub in target_substances:
            substance += np.array(CMAQoutf.variables[sub][:])
        target = substance[:, 0, nearlat, nearlon]
        if Molar_mass != 0 : target = target * Molar_mass / 22.4 * 1000
        substance_station = target
        pollution1_pre = substance_station

        # 计算24h滑动平均
        # pollution1 = []
        # for x in range(16,760):
        #     moving_avg_list = substance_station[x-12:x+12]
        #     pollution1.append(np.mean(moving_avg_list))

        # 计算日均值或几个小时的均值
        # pollution1 = []
        # for x in range(16, 760,24):
        #     day_avg_list = substance_station[x:x + 24]
        #     pollution1.append(np.mean(day_avg_list))

        # pollution1 = substance_station[16:760]   # 得到UTC+8后的8月1-8月31
        pollution1 = substance_station[simdata_inithour:simdata_inithour+24*daycount]  # 得到UTC+8后的8月1-8月31
        # pollution1 = substance_station[0:744]   # 得到UTC+8后的8月1-8月31
        # print('污染物列表的长度：   ', len(pollution1))
        # print(pollution1)

        # 读取观测值csv列表来获取整个观测值数组
        for m in airStation_csv_list:
            airStation_csv = pd.read_csv(m)
            target_col = airStation_csv.columns.get_loc(airstation[2])# 找出站点代号所在的列
            matching_rows = airStation_csv[airStation_csv[target_col] == target_substance_obs] # 获取对应物质所在列
            pollution1_station_pre = airStation_csv.loc[matching_rows, target_col]
            pollution1_station_pre = pollution1_station_pre.replace('', np.nan).astype(int).tolist()

            # print(pollution1_station)
            # print('list长度：   ', len(pollution1_station))

        # 判断站点数据是否是全空的
        # print((np.isnan(np.array(pollution1_station))).all())
        if (np.isnan(np.array(pollution1_station_pre))).all():
            continue
        xls_write_num += 1

        pollution1_station = pollution1_station_pre  # 原始小时值情形

        # 某h均值计算情形
        # pollution1_station = []
        # for x in range(0, 744, 24):
        #     day_avg_list = pollution1_station_pre[x:x + 24]
        #     day_avg_list_nonan = []
        #     for i in day_avg_list:
        #         if np.isnan(i) == False:
        #             day_avg_list_nonan.append(i)
        #     pollution1_station.append(np.mean(day_avg_list_nonan))

        # 某h滑动均值计算情形
        # pollution1_station = []
        # for x in range(16, 760):
        #     day_avg_list = pollution1_station_pre[x-12:x + 12]
        #     day_avg_list_nonan = []
        #     for i in day_avg_list:
        #         if np.isnan(i) == False:
        #             day_avg_list_nonan.append(i)
        #     pollution1_station.append(np.mean(day_avg_list_nonan))

        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        fig2 = plt.figure(figsize=(5, 2), dpi=200)

        xdate = []  # 生成对应的时间序列，否则横坐标无法正常转化为日期 744小时
        for x in range(1, 10):
            for b in range(0, 10):
                xdate.append(
                    datetime.datetime.strptime(year + '-01-' + '0' + str(x) + '-' + '0' + str(b), '%Y-%m-%d-%H'))
            for b in range(10, 24):
                xdate.append(datetime.datetime.strptime(year + '-01-' + '0' + str(x) + '-' + str(b), '%Y-%m-%d-%H'))
        for x in range(10, 32):
            for b in range(0, 10):
                xdate.append(datetime.datetime.strptime(year + '-01-' + str(x) + '-' + '0' + str(b), '%Y-%m-%d-%H'))
            for b in range(10, 24):
                xdate.append(datetime.datetime.strptime(year + '-01-' + str(x) + '-' + str(b), '%Y-%m-%d-%H'))

        # xdate = []  # 生成对应的时间序列，否则横坐标无法正常转化为日期 31天 日均
        # for x in range(1, 10):
        #     xdate.append(datetime.datetime.strptime(year + '-01-' + '0' + str(x), '%Y-%m-%d'))
        # for x in range(10, 32):
        #     xdate.append(datetime.datetime.strptime(year + '-01-' + str(x), '%Y-%m-%d'))

        print('两者长度比较 ：   ', len(pollution1_station), len(pollution1))
        ax2 = fig2.add_subplot(111)
        Hour = range(1, len(pollution1_station))  # 横坐标，初始为小时
        line1, = ax2.plot(xdate, pollution1_station, linewidth=0.5, label='Obs', color='#258080')  # 要用legend画图例，这里必须,=
        line2, = ax2.plot(xdate, pollution1, linewidth=0.5, label='Sim', color='red')
        ax2.set_ylabel('PM2.5 Concentration', fontsize=5)
        ax2.set_xlabel('Date', fontsize=5)
        plt.title('PM2.5 Concentration', fontsize=5)
        plt.xticks(fontsize=5)  # xticks必须在这个位置才生效
        plt.yticks(fontsize=5)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))  # 设置不显示年份
        plt.legend((line1, line2), ('Obs', 'Sim'), loc='upper right', frameon=False, framealpha=0.5,
                   fontsize=5)
        """
        计算精度系数
        """
        substanceMAE = WCt.calcuMAE(pollution1, pollution1_station)
        substanceR = WCt.calcuR(pollution1, pollution1_station)
        substanceIOA = WCt.calcuIOA(pollution1, pollution1_station)
        substanceNMB = WCt.calcuNMB(pollution1, pollution1_station)
        substanceNME = WCt.calcuNME(pollution1, pollution1_station)
        # substanceMFE = WCt.calcuMFE(pollution1, pollution1_station)
        # substanceMFB = WCt.calcuMFB(pollution1, pollution1_station)
        substanceFE = WCt.calcuFE(pollution1, pollution1_station)
        substanceFB = WCt.calcuFB(pollution1, pollution1_station)
        sheet.write(xls_write_num + 1, 0, stname)
        sheet.write(xls_write_num + 1, 1, substanceMAE)
        sheet.write(xls_write_num + 1, 2, substanceR)
        sheet.write(xls_write_num + 1, 3, substanceIOA)
        sheet.write(xls_write_num + 1, 4, substanceNMB)
        sheet.write(xls_write_num + 1, 5, substanceNME)
        sheet.write(xls_write_num + 1, 6, substanceFE)
        sheet.write(xls_write_num + 1, 7, substanceFB)
        sheet.write(xls_write_num + 1, 8, k[1])
        sheet.write(xls_write_num + 1, 9, k[0])
        sheet.write(xls_write_num + 1, 10, k[3])
        plt.savefig(out_dir + k[3] + suffix)  # 不以站点名命名，以所在区县命名

    Xfile.save(result_xls_name)


if __name__ == '__main__':
    pass
