import datetime
import netCDF4 as nt
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
import netCDF4 as nc
from tqdm import tqdm


def calcuRSME(modData, obsData):
    """
    计算RSME
    :param modData: 模拟数据(列表)
    :param obsData: 观测数据(列表)
    :return: RSME
    """
    N = len(modData) if len(modData) < len(obsData) else len(obsData)  # 不过限度  # 数据总个数
    ALL = 0
    NAN_count = 0
    for i in range(0, N):
        if np.isnan(obsData[i]) == True:  # 判断nan
            NAN_count += 1
            continue
        ALL += (modData[i] - obsData[i]) ** 2
    return np.sqrt(ALL / (N-NAN_count))

def calcuMAE(modData, obsData):
    """
    计算MAE
    :param modData:
    :param obsData:
    :return:
    """
    N = len(modData) if len(modData) < len(obsData) else len(obsData)  # 不过限度  # 数据总个数
    ALL = 0
    NAN_count = 0
    for i in range(0, N):
        if np.isnan(obsData[i]) == True:  # 判断nan
            NAN_count += 1
            continue
        ALL += abs(modData[i] - obsData[i])
    return ALL / (N-NAN_count)

def calcuMB(modData, obsData):
    N = len(modData) if len(modData) < len(obsData) else len(obsData)  # 不过限度
    ALL = 0
    NAN_count = 0
    for i in range(0, N):
        if np.isnan(obsData[i]) == True:  # 判断nan
            NAN_count += 1
            continue
        ALL += modData[i] - obsData[i]
    return ALL / (N-NAN_count)

def calcuR(modData, obsData):
    N = len(modData)  # 数据总个数
    ALL1 = 0
    ALL2 = 0
    ALL3 = 0
    obsData = list(obsData)
    modData = list(modData)

    while 1:  # 直到mean算出之前一直筛查
        if np.isnan(np.mean(obsData)) == False:
            break
        for i in obsData:
            if np.isnan(i) == True:  # 去掉NAN再求平均
                modData.pop(obsData.index(i))  # 这个必须在前，去除之前找到index
                obsData.pop(obsData.index(i))
    # print(len(modData))
    # print(len(obsData))

    modMean = np.mean(modData)
    obsMean = np.mean(obsData)
    N2 = len(modData) if len(modData) < len(obsData) else len(obsData)  # 不过限度
    for i in range(0, N2):
        if np.isnan(obsData[i]) == True:  # 判断nan
            continue
        ALL1 += (modData[i] - modMean) * (obsData[i] - obsMean)
        ALL2 += (modData[i] - modMean) ** 2
        ALL3 += (obsData[i] - obsMean) ** 2
    return ALL1 / np.sqrt(ALL2 * ALL3)

def calcuIOA(modData,obsData):
    N = len(modData)  # 数据总个数
    ALL1 = 0
    ALL2 = 0
    obsData = list(obsData)
    modData = list(modData)
    while 1:  # 直到mean算出之前一直筛查
        if np.isnan(np.mean(obsData)) == False:
            break
        for i in obsData:
            if np.isnan(i) == True:  # 去掉NAN再求平均
                modData.pop(obsData.index(i))  # 这个必须在前，去除之前找到index
                obsData.pop(obsData.index(i))
    modMean = np.mean(modData)
    obsMean = np.mean(obsData)
    N2 = len(modData) if len(modData) < len(obsData) else len(obsData)  # 不过限度
    for i in range(0, N2):
        if np.isnan(obsData[i]) == True:  # 判断nan
            continue
        ALL1 += (modData[i] - obsData[i]) ** 2
        ALL2 += (abs(modData[i] - obsMean) + abs(obsData[i] - obsMean)) ** 2

    return 1 - ALL1/ALL2

def calcuNMB(modData, obsData):
    N = len(modData) if len(modData) < len(obsData) else len(obsData)  # 不过限度
    ALL1 = 0
    ALL2 = 0
    for i in range(0, N):
        if np.isnan(obsData[i]) == True:  # 判断nan
            continue
        ALL1 += modData[i] - obsData[i]
        ALL2 += obsData[i]
    return ALL1 / ALL2

def calcuNME(modData, obsData):
    N = len(modData) if len(modData) < len(obsData) else len(obsData)  # 不过限度
    ALL1 = 0
    ALL2 = 0
    for i in range(0, N):
        if np.isnan(obsData[i]) == True:  # 判断nan
            continue
        ALL1 += abs(modData[i] - obsData[i])
        ALL2 += obsData[i]
    return ALL1 / ALL2

def calcuMFE(modData, obsData):
    N = len(modData) if len(modData) < len(obsData) else len(obsData)  # 不过限度
    ALL = 0
    A = 0
    B = 0
    NAN_count = 0
    for i in range(0, N):
        if np.isnan(obsData[i]) == True:  # 判断nan
            NAN_count += 1
            continue
        A = abs(modData[i] - obsData[i])
        B = obsData[i] + modData[i] / 2
        ALL += A / B
    return ALL / (N-NAN_count) * 100

def calcuMFB(modData, obsData):
    N = len(modData) if len(modData) < len(obsData) else len(obsData)  # 不过限度
    ALL = 0
    A = 0
    B = 0
    NAN_count = 0
    for i in range(0, N):
        if np.isnan(obsData[i]) == True:  # 判断nan
            NAN_count += 1
            continue
        A = modData[i] - obsData[i]
        B = obsData[i] + modData[i] / 2
        ALL += A / B
    return ALL / (N-NAN_count) * 100

def calcuFE(modData, obsData):
    N = len(modData) if len(modData) < len(obsData) else len(obsData)  # 不过限度
    ALL = 0
    A = 0
    B = 0
    NAN_count = 0
    for i in range(0, N):
        if np.isnan(obsData[i]) == True:  # 判断nan
            NAN_count += 1
            continue
        A = abs(modData[i] - obsData[i])
        B = obsData[i] + modData[i]
        ALL += A / B
    return ALL * 2 / (N-NAN_count) * 100

def calcuFB(modData, obsData):
    N = len(modData) if len(modData) < len(obsData) else len(obsData)  # 不过限度
    ALL = 0
    A = 0
    B = 0
    NAN_count = 0
    for i in range(0, N):
        if np.isnan(obsData[i]) == True:  # 判断nan
            NAN_count += 1
            continue
        A = modData[i] - obsData[i]
        B = obsData[i] + modData[i]
        ALL += A / B
    return ALL * 2 / (N-NAN_count) * 100

def calcuGE(modData, obsData):
    N = len(modData) if len(modData) < len(obsData) else len(obsData)  # 不过限度
    ALL = 0
    A = 0
    B = 0
    NAN_count = 0
    for i in range(0, N):
        if np.isnan(obsData[i]) == True:  # 判断nan
            NAN_count += 1
            continue
        A = modData[i] - obsData[i]
        B = obsData[i] + modData[i]
        ALL += abs(A / B)
    return ALL * 2 / (N-NAN_count) * 100

def missing_hour_fill(incomplete_list):
    full_list = list(range(24))
    # incomplete_list = [0, 1, 3, 4, 5, 10, 15, 23]  # 示例不完整列表
    incomplete_array = np.array(incomplete_list)
    result_array = np.full(len(full_list), np.nan)
    result_array[incomplete_array] = incomplete_array
    return result_array

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

def generate_date_list(start_date_str,n):
    start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
    """从 start_date 开始，生成 n 天的日期列表"""
    return [start_date + datetime.timedelta(days=i) for i in range(n)]

def generate_date_list_withhour(start_date_str,n):
    start_date_str += '-00'
    start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d-%H')
    date_list = []
    for i in range(n+1):  # 包括起始时间，所以是 745
        current_date = start_date + datetime.timedelta(hours=i)
        date_list.append(current_date)
    return date_list

def getAirStationsFromLatLon(uplat, downlat, leftlon, rightlon, station_info_csv=""):
    """
    从站点信息csv文件中，返回在某一个矩形经纬度范围内的站点信息，格式为验证代码中的字典格式
    :param uplat:
    :param downlat:
    :param leftlon:
    :param rightlon:
    :return: airStations: 空气质量站点信息
    """
    # airStation_infofile_dir = r"E:\全国空气质量\_站点列表\站点信息2016起.xls"
    airStation_infofile_dir = station_info_csv
    airStation_infofile = pd.read_csv(airStation_infofile_dir)
    airStations = {}

    # 去掉经纬度信息不符合规则的站点
    airStation_infofile['经度'] = pd.to_numeric(airStation_infofile['经度'], errors='coerce')
    airStation_infofile['纬度'] = pd.to_numeric(airStation_infofile['纬度'], errors='coerce')
    filtered_rows = airStation_infofile[
        (airStation_infofile['经度'].notna()) &
        (airStation_infofile['纬度'].notna())
        ]

    final_filtered_rows  = filtered_rows[
        (filtered_rows['经度'] > leftlon) & (filtered_rows['经度'] < rightlon) &
        (filtered_rows['纬度'] > downlat) & (filtered_rows['纬度'] < uplat)
        ]
    for row in final_filtered_rows.index.tolist():
        airStations.update({airStation_infofile.at[row,'监测点名称']:
                                [float(airStation_infofile.at[row,'经度']),
                                 float(airStation_infofile.at[row,'纬度']),
                                 airStation_infofile.at[row,'监测点编码'],
                                 airStation_infofile.at[row,'城市']]})
                                        # 数据格式：站点名：经度 纬度 站点代号 所在城市
    return airStations

def CMAQ_site_validation(
    start_date='YYYY-MM-DD',
    daycount=5, #
    simdata_inithour=16, #
    GRIDCRO2D_file_dir = "",
    Combine_file_dir = "",
    target_substances = [], #
    target_substance_obs = '', #
    Molar_mass = 0, #
    airstation_files_dir = "",
    airstation_infofile_dir = "",
    out_dir = "", #
    result_csv_name = "", #
    suffix = "",
):
    """
    使用全国公开csv逐小时基本污染物逐小时浓度数据验证CAMQ模拟结果

    :param start_date: 验证的开始时期，确定观测值数据获取
    :param daycount: 验证持续天数
    :param simdata_inithour: 模拟值数据列表的起始位置，由于模拟数据为utc0，则以第16为数组的开始位置才能完整的取到一天，可修改
    :param GRIDCRO2D_file_dir: CMAQ目标验证domian的任意GRIDCRO2D文件
    :param Combine_file_dir:  CMAQ包含最终目标物质浓度的combine后文件
    :param target_substances: 目标污染物，若输入多个，则是几个污染物类型的和，如不同ISAMtag相加的总量
    :param target_substance_obs: 观测数据中的污染物名称，用于获取，不同于CMAQ的输出nc，一般验证只会获取一类
    :param Molar_mass: ppm为单位的物质的摩尔质量，为0时则不进行转换，如PM2.5，直接为ug
    :param airstation_files_dir: 包含观测站数据的csv文件所在文件夹
    :param airstation_infofile_dir: 包含观测站信息的csv文件路径
    :param out_dir: 验证结果输出的文件夹
    :param result_csv_name: 包含验证参数计算结果的输出csv文件名称
    :param suffix: 验证过程后缀，用于区分
    :return:
    """
    year = str(datetime.datetime.strptime(start_date, '%Y-%m-%d')).split('-')[0] # 年份
    CMAQGRIDCRO2D_file_dir = GRIDCRO2D_file_dir  # CMAQ经纬度对应网格文件，用于得到站点位置
    CMAQoutISAMCombine_file_dir = Combine_file_dir  # 合并后的污染物浓度文件
    # airStation_file_dir = f"{airstation_files_dir}\\站点_{year}0101-{year}1231\\"  # 目标天的空气质量站点数据所在文件夹
    if os.path.exists(out_dir) is False: os.mkdir(out_dir)
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

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
    validation_dates = generate_date_list(start_date,daycount)
    for date in validation_dates:
        date2 = str(date).split(' ')[0]
        filedate = f'{date2.split("-")[0]}{date2.split("-")[1]}{date2.split("-")[2]}'
        airStation_csv_list.append(f"{airstation_files_dir}china_sites_{filedate}.csv")


    csv_row = 1 # csv非表头的第1行开始写入
    for airstation in tqdm(airStation_locations.values(),desc='处理所有有效站点...'): # 从站点列表处理每一个站点
        stname = list(airStation_locations.keys())[list(airStation_locations.values()).index(airstation)]  # 通过值k获取字典dic对应键的公式： list(dic.keys())[list(dic.values()).index(k)]
        # 读取经纬度与网格关系数据
        CMAQXLAT = np.array(GRIDCRO2D.variables['LAT'][:][0])
        CMAQXLONG = np.array(GRIDCRO2D.variables['LON'][:][0])
        nearpos = getNearestPos(airstation[1], airstation[0], CMAQXLAT, CMAQXLONG)  # 从WRF得到站点格子
        nearlat = nearpos[1]  # 与WRF获取最近点不同
        nearlon = nearpos[2]

        subdata_shape = np.array(CMAQoutf.variables[target_substances[0]][:]).shape  # 获得数据shape来存放all
        substance = np.zeros(subdata_shape,'float64')
        for sub in target_substances:
            substance += np.array(CMAQoutf.variables[sub][:])
        target = substance[:, 0, nearlat, nearlon]
        if Molar_mass != 0 : target = target * Molar_mass / 22.4 * 1000
        substance_station = target
        pollution1 = substance_station[simdata_inithour:simdata_inithour+24*daycount]  # 得到UTC+8后的8月1-8月31

        # 读取观测值csv列表来获取整个观测值数组
        pollution1_station_pre = []
        for m in airStation_csv_list:
            airStation_csv = pd.read_csv(m)
            matching_rows = airStation_csv[airStation_csv['type'] == target_substance_obs].index.tolist() # 获取对应物质所在列
            matching_hours = airStation_csv.loc[matching_rows, 'hour'].astype(int) # 查看有哪些小时
            fill_hours = list(missing_hour_fill(matching_hours))
            for hour in fill_hours: # 检测从某日csv文件获取的是否有缺失小时，有则顺序补充nan
                if hour == np.nan:
                    pollution1_station_pre.append(np.nan)
                else:
                    target_row = airStation_csv[(airStation_csv['hour'] == hour) & (airStation_csv['type'] == target_substance_obs)]
                    if airstation[2] in airStation_csv.columns: # 判断目标站点是否存在，不存在添加空值
                        pollution1_station_pre.append(target_row[airstation[2]].values[0])
                    else:
                        pollution1_station_pre.append(np.nan)
            # pollution1_station_pre1 = airStation_csv.loc[matching_rows, target_col]
            # pollution1_station_pre += pollution1_station_pre1.replace('', np.nan).astype(int).tolist()

        # 判断站点数据是否是全空的，是则跳过对这个站点的结果输出
        # print((np.isnan(np.array(pollution1_station))).all())
        if (np.isnan(np.array(pollution1_station_pre))).all():
            continue

        pollution1_station = pollution1_station_pre  # 原始小时值情形

        fig2 = plt.figure(figsize=(5, 2), dpi=200)
        xdate = generate_date_list_withhour(start_date,daycount*24)
        xdate.pop() # 会多一个h
        ax2 = fig2.add_subplot(111)
        line1, = ax2.plot(xdate, pollution1_station, linewidth=0.5, label='Obs', color='#258080')  # 要用legend画图例，这里必须,=
        line2, = ax2.plot(xdate, pollution1, linewidth=0.5, label='Sim', color='red')
        ax2.set_ylabel('μg/m$^{3}$', fontsize=10)
        ax2.set_xlabel('Date', fontsize=10)
        plt.title(f'{target_substance_obs} Concentration validation at site {stname}', fontsize=10)
        plt.xticks(fontsize=5)  # xticks必须在这个位置才生效
        plt.yticks(fontsize=5)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))  # 设置不显示年份
        plt.legend((line1, line2), ('Obs', 'Sim'), loc='upper right', frameon=False, framealpha=0.5,
                   fontsize=5)
        plt.savefig(f'{out_dir}{stname}_{suffix}.png')
        plt.close()
        """
        计算精度系数
        """
        substanceMAE = calcuMAE(pollution1, pollution1_station)
        substanceR = calcuR(pollution1, pollution1_station)
        substanceIOA = calcuIOA(pollution1, pollution1_station)
        substanceNMB = calcuNMB(pollution1, pollution1_station)
        substanceNME = calcuNME(pollution1, pollution1_station)
        substanceMFE = calcuMFE(pollution1, pollution1_station)
        substanceMFB = calcuMFB(pollution1, pollution1_station)
        substanceFE = calcuFE(pollution1, pollution1_station)
        substanceFB = calcuFB(pollution1, pollution1_station)
        result_csv_data.at[csv_row, '站点'] = stname
        result_csv_data.at[csv_row, 'MAE'] = substanceMAE
        result_csv_data.at[csv_row, 'R'] = substanceR
        result_csv_data.at[csv_row, 'IOA'] = substanceIOA
        result_csv_data.at[csv_row, 'NMB'] = substanceNMB
        result_csv_data.at[csv_row, 'NME'] = substanceNME
        result_csv_data.at[csv_row, 'MFE'] = substanceMFE
        result_csv_data.at[csv_row, 'MFB'] = substanceMFB
        result_csv_data.at[csv_row, 'FE'] = substanceFE
        result_csv_data.at[csv_row, 'FB'] = substanceFB
        result_csv_data.at[csv_row, '站点经度'] = airstation[1]
        result_csv_data.at[csv_row, '站点纬度'] = airstation[0]
        result_csv_data.at[csv_row, '城市'] = airstation[3]
        csv_row += 1


    result_csv_data.to_csv(f'{out_dir}{result_csv_name}_{suffix}.csv',encoding='utf-8')


if __name__ == '__main__':
    CMAQ_site_validation(
        start_date='2020-08-01',
        daycount=30,
        simdata_inithour=16,
        GRIDCRO2D_file_dir="E:\Emission_update\GRIDCRO2D_2020213.nc",
        Combine_file_dir="E:\Emission_update\CD202008_MEIAT-IA_d03_combine_PM2503_IAave.nc",
        target_substances=['O3'],
        target_substance_obs='O3',
        Molar_mass=48,
        airstation_files_dir=r'E:\全国空气质量\全国站点小时浓度csv_files\\',
        airstation_infofile_dir="E:\全国空气质量\_站点列表\站点列表-2022.02.13起.csv",
        out_dir=r'E:\Emission_update\\validation\\',
        result_csv_name='validationPara',
        suffix='IAave'
    )
    pass
