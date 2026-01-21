"""
Author: Yaohan Xian
GitHub: https://github.com/Sm0keNMirrors
Last update:
"""

import datetime
import os
import sys
import pypinyin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
import netCDF4 as nc
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde


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

def calculate_figRowCol_size(x):
    import math
    # 对 x 进行开方，找到一个接近的行数
    n = int(math.ceil(math.sqrt(x)))  # 行数
    m = int(math.ceil(x / n))          # 列数
    return n, m

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
    airstation_selected = [],
    result_pic_types=[],
    result_pic_combine = [],
    out_dir = "", #
    result_csv_name = "", #
    suffix = "",
):
    import datetime
    import os
    import sys
    import pypinyin
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib
    import netCDF4 as nc
    from tqdm import tqdm
    from sklearn.linear_model import LinearRegression
    from scipy.stats import gaussian_kde
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
    :param airstation_selected: 选择性得仅输出名字为这些站点的验证结果，当[]时或不指定则输出范围内的全部站点
    :param result_pic_types: 输出哪些类型的结果图，'line'：双折线图，'scatter':散点回归线图
    :param result_pic_combine: 是否将结果图画为一个总图，对应上面的'line'，'scatter'
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
    CMAQoutf = nc.Dataset(CMAQoutISAMCombine_file_dir, "r")  # 打开CMAQ输出的NC格式文件
    var_lon = np.array(GRIDCRO2D.variables['LON'][:][0])
    var_lat = np.array(GRIDCRO2D.variables['LAT'][:][0])
    lonmin, latmax, lonmax, latmin = (var_lon.min(), var_lat.max(),
                                      var_lon.max(), var_lat.min())
    airStation_locations_ = getAirStationsFromLatLon(latmax, latmin, lonmin, lonmax,airstation_infofile_dir)  # 获得模拟区域内的所有站点信息
    airStation_locations = {}
    if airstation_selected != []:
        for x in airStation_locations_:
            if x in airstation_selected:
                airStation_locations.update({x:airStation_locations_[x]})
    else:
        airStation_locations = airStation_locations_
    print(airStation_locations)


    labels = ['站点', 'MAE','RMSE', 'R', 'IOA', 'NMB', 'NME','MFE', 'MFB','FE', 'FB', '站点经度', '站点纬度', '城市']
    result_csv_data = pd.DataFrame(columns=labels) # 创建记录结果的csv

    airStation_csv_list = [] # 验证时间内的所有csv
    validation_dates = generate_date_list(start_date,daycount)
    for date in validation_dates:
        date2 = str(date).split(' ')[0]
        filedate = f'{date2.split("-")[0]}{date2.split("-")[1]}{date2.split("-")[2]}'
        airStation_csv_list.append(f"{airstation_files_dir}china_sites_{filedate}.csv")


    if result_pic_combine != []:
        if 'scatter' in result_pic_combine:
            nn,mm = calculate_figRowCol_size(len(list(airStation_locations.keys())))
            fig_c, axs_c = plt.subplots(nn, mm, figsize=(18, 18), dpi=300,)  # 创建 子图布局
    csv_row = 1 # csv非表头的第1行开始写入
    plotcount = 0 # 绘制子图的计数
    for airstation in tqdm(airStation_locations.values(),desc='处理所有有效站点...'): # 从站点列表处理每一个站点
        stname = list(airStation_locations.keys())[list(airStation_locations.values()).index(airstation)]  # 通过值k获取字典dic对应键的公式： list(dic.keys())[list(dic.values()).index(k)]
        # 读取经纬度与网格关系数据
        stcity = airStation_locations[stname][3]
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
                    # print(target_row[airstation[2]].values)
                    if airstation[2] in airStation_csv.columns and np.array(target_row[airstation[2]]).size > 0 : # 判断目标站点是否存在，不存在添加空值
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


        #绘制不同类型的结果图
        if 'line' in result_pic_types:
            # if os.path.exists(f'{out_dir}pics_line\\') is False: os.mkdir(f'{out_dir}pics_line\\')
            # fig2 = plt.figure(figsize=(5, 2), dpi=200)
            # xdate = generate_date_list_withhour(start_date,daycount*24)
            # xdate.pop() # 会多一个h
            # ax2 = fig2.add_subplot(111)
            # line1, = ax2.plot(xdate, pollution1_station, linewidth=0.5, label='Obs', color='#258080')  # 要用legend画图例，这里必须,=
            # line2, = ax2.plot(xdate, pollution1, linewidth=0.5, label='Sim', color='red')
            # ax2.set_ylabel('μg/m$^{3}$', fontsize=10)
            # ax2.set_xlabel('Date', fontsize=10)
            # plt.title(f'{target_substance_obs} Concentration validation at site {stname}', fontsize=10)
            # plt.xticks(fontsize=5)  # xticks必须在这个位置才生效
            # plt.yticks(fontsize=5)
            # ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))  # 设置不显示年份
            # plt.legend((line1, line2), ('Obs', 'Sim'), loc='upper right', frameon=False, framealpha=0.5,
            #            fontsize=5)
            # fig2.savefig(f'{out_dir}pics_line\\{stname}_{suffix}.png')
            # fig2.close()

            if os.path.exists(f'{out_dir}pics_line\\') is False:
                os.mkdir(f'{out_dir}pics_line\\')

            fig2 = plt.figure(figsize=(5, 2), dpi=200)
            xdate = generate_date_list_withhour(start_date, daycount * 24)
            xdate.pop()  # 会多一个h
            ax2 = fig2.add_subplot(111)

            # ====== ② 单位：ug/m3 -> ppb（仅适用于气态；25℃, 1atm: ppb = ug/m3 * 24.45 / MW）======
            import numpy as np
            import pandas as pd

            def _to_ppb(arr_ugm3, mw):
                a = np.asarray(arr_ugm3, dtype=np.float32)
                return a * (24.45 / mw)

            # 常见污染物分子量（g/mol）
            _mw_map = {
                "O3": 48.0,
                "NO2": 46.0055,
                "NO": 30.0061,
                "SO2": 64.066,
                "CO": 28.0101,
                "NH3": 17.031,
                "HCHO": 30.026,
            }

            # 从 target_substance_obs 中提取类似 O3/NO2/SO2 这类 key
            # _key = re.sub(r"[^A-Za-z0-9]+", "", str(target_substance_obs).upper())
            _key = "O3"
            mw = _mw_map.get(_key, None)

            # 处理缺测：统一转为 float，并用 NaN 掩膜
            obs = pd.to_numeric(pd.Series(pollution1_station), errors="coerce").to_numpy(dtype=np.float32)
            sim = pd.to_numeric(pd.Series(pollution1), errors="coerce").to_numpy(dtype=np.float32)

            if mw is not None:
                obs_ppb = _to_ppb(obs, mw)
                sim_ppb = _to_ppb(sim, mw)
                ylab = "ppb"
            else:
                # 若识别不到分子量（例如 PM2.5），ppb 不适用：退回原单位，避免输出错误单位
                obs_ppb = obs
                sim_ppb = sim
                ylab = "μg/m$^{3}$"

            # ====== ③ 指标：IOA / MB / RMSE（基于有效值 mask）======
            mask = np.isfinite(obs_ppb) & np.isfinite(sim_ppb)
            if np.any(mask):
                O = obs_ppb[mask]
                P = sim_ppb[mask]
                Obar = np.mean(O)

                # IOA (Willmott)
                denom = np.sum((np.abs(P - Obar) + np.abs(O - Obar)) ** 2)
                ioa = 1.0 - (np.sum((P - O) ** 2) / denom) if denom > 0 else np.nan

                # Mean Bias & RMSE
                mb = np.mean(P - O)
                rmse = np.sqrt(np.mean((P - O) ** 2))
            else:
                ioa, mb, rmse = np.nan, np.nan, np.nan

            # ====== ① 优化绘图：更清爽的轴、网格、legend、日期 ======
            line1, = ax2.plot(xdate, obs_ppb, linewidth=0.9, label='Obs', color='#258080')
            line2, = ax2.plot(xdate, sim_ppb, linewidth=0.9, label='Sim', color='red', alpha=0.85)

            ax2.set_ylabel(ylab, fontsize=9)
            ax2.set_xlabel('Date', fontsize=9)

            # 只显示月-日
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax2.tick_params(axis='x', labelsize=6, rotation=0)
            ax2.tick_params(axis='y', labelsize=6)

            # 细网格
            ax2.grid(True, which='major', linewidth=0.3, alpha=0.35)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)

            # 曲线图例（右上角）
            ax2.legend(loc='upper right', frameon=False, fontsize=6, handlelength=2.2, borderaxespad=0.2)

            # 指标文字（图上方，类似 legend）
            metric_txt = f"IOA={ioa:.2f}   MB={mb:.2f} {ylab}   RMSE={rmse:.2f} {ylab}"
            ax2.text(
                0.0, 1.08, metric_txt,
                transform=ax2.transAxes,
                ha='left', va='bottom',
                fontsize=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8)
            )

            # 标题（略紧凑）
            ax2.set_title(f'{stname}',loc='right', fontsize=6)

            fig2.tight_layout()
            fig2.savefig(f'{out_dir}pics_line\\{stname}_{suffix}.png')
            plt.close(fig2)


        if 'scatter' in result_pic_types:
            if os.path.exists(f'{out_dir}pics_scatter\\') is False: os.mkdir(f'{out_dir}pics_scatter\\')
            plt.figure(figsize=(6, 6) ,dpi=200)
            observed = np.array(pollution1_station)
            simulated = np.array(pollution1) # 转为数组后才能进行掩码处理
            # 去除 NaN 值
            mask = ~np.isnan(observed)  # 创建一个掩码，选择观测值不是 NaN 的索引
            observed_clean = observed[mask]
            simulated_clean = simulated[mask]
            xy = np.vstack([observed_clean, simulated_clean])
            z = gaussian_kde(xy)(xy)
            scatter = plt.scatter(observed_clean, simulated_clean, c=z, cmap='viridis', alpha=0.6)
            model = LinearRegression()
            model.fit(observed_clean.reshape(-1, 1), simulated_clean)
            x_fit = np.linspace(min(observed_clean), max(observed_clean), 100).reshape(-1, 1)
            y_fit = model.predict(x_fit)
            slope = model.coef_[0]
            intercept = model.intercept_
            equation = f'y = {slope:.2f}x + {intercept:.2f}'
            # plt.text(0.9 * max(observed_clean), 0.9 * max(simulated_clean), equation, fontsize=12, color='red',
            #          ha='center', transform=plt.gca().transAxes)
            plt.plot(x_fit, y_fit, color='red', linewidth=2, label=equation)
            # cbar = plt.colorbar(scatter)
            # cbar.set_label('scatter density')
            plt.xlabel('Obs')
            plt.ylabel('Mod')
            plt.title(f'Scatter and regression line at site {stname}')
            plt.legend()
            # plt.grid()

            plt.savefig(f'{out_dir}pics_scatter\\{stname}_{suffix}.png')
            plt.close()
            if 'scatter' in result_pic_combine:
                row = plotcount // mm  # 绘图行列
                col = plotcount % mm  # 绘图行列
                axs_c[row, col].scatter(observed_clean, simulated_clean,s=10, c=z, cmap='viridis', alpha=0.6)
                axs_c[row, col].plot(x_fit, y_fit, color='red', linewidth=2, label=equation)
                stcity_eng = ''.join(pypinyin.lazy_pinyin(stcity))
                axs_c[row, col].text(0.01, 0.99, stcity_eng, transform=axs_c[row, col].transAxes,
                      fontsize=18, verticalalignment='top', horizontalalignment='left')
                axs_c[row, col].legend(loc='lower right')
                axs_c[row, col].set_xlim(0,250)
                axs_c[row, col].set_ylim(0, 250)
        """
        计算精度系数
        """
        substanceMAE = calcuMAE(sim_ppb, obs_ppb)
        substanceRSME = calcuRSME(sim_ppb, obs_ppb)
        substanceR = calcuR(sim_ppb, obs_ppb)
        substanceIOA = calcuIOA(sim_ppb, obs_ppb)
        substanceNMB = calcuNMB(sim_ppb, obs_ppb)
        substanceNME = calcuNME(sim_ppb, obs_ppb)
        substanceMFE = calcuMFE(sim_ppb, obs_ppb)
        substanceMFB = calcuMFB(sim_ppb, obs_ppb)
        substanceFE = calcuFE(sim_ppb, obs_ppb)
        substanceFB = calcuFB(sim_ppb, obs_ppb)
        result_csv_data.at[csv_row, '站点'] = stname
        result_csv_data.at[csv_row, 'MAE'] = substanceMAE
        result_csv_data.at[csv_row, 'RMSE'] = substanceRSME
        result_csv_data.at[csv_row, 'R'] = substanceR
        result_csv_data.at[csv_row, 'IOA'] = substanceIOA
        result_csv_data.at[csv_row, 'NMB'] = substanceNMB
        result_csv_data.at[csv_row, 'NME'] = substanceNME
        result_csv_data.at[csv_row, 'MFE'] = substanceMFE
        result_csv_data.at[csv_row, 'MFB'] = substanceMFB
        result_csv_data.at[csv_row, 'FE'] = substanceFE
        result_csv_data.at[csv_row, 'FB'] = substanceFB
        result_csv_data.at[csv_row, '站点纬度'] = airstation[1]
        result_csv_data.at[csv_row, '站点经度'] = airstation[0]
        result_csv_data.at[csv_row, '城市'] = airstation[3]
        csv_row += 1
        plotcount+=1

    result_csv_data.to_csv(f'{out_dir}{result_csv_name}_{suffix}.csv',encoding='utf-8')
    # 总体绘图的一些设置
    fig_c.delaxes(axs_c[nn-1, mm-1])  # 删除一些子图
    fig_c.delaxes(axs_c[nn-1, mm-2])
    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08)
    fig_c.supxlabel('Observed O$_{3}$ (μg/m$^{3}$)', fontsize=25)
    fig_c.supylabel('Simulated O$_{3}$ (μg/m$^{3}$)', fontsize=25)
    fig_c.savefig(f'{out_dir}\\scatter_combine_{suffix}.png')

if __name__ == '__main__':
    # CMAQ_site_validation(
    #     start_date='2020-08-01',
    #     daycount=30,
    #     simdata_inithour=16,
    #     GRIDCRO2D_file_dir="E:\Emission_update\GRIDCRO2D_2020213.nc",
    #     Combine_file_dir="E:\Emission_update\CD202008_MEIAT-IA_d03_combine_PM2503_IAave.nc",
    #     target_substances=['O3'],
    #     target_substance_obs='O3',
    #     Molar_mass=48,
    #     airstation_files_dir=r'E:\全国空气质量\全国站点小时浓度csv_files\\',
    #     airstation_infofile_dir="E:\全国空气质量\_站点列表\站点列表-2022.02.13起.csv",
    #     result_pic_types=['line','scatter'],
    #     out_dir=r'E:\Emission_update\\validation\\',
    #     result_csv_name='validationPara',
    #     suffix='IAave'
    # )

    CMAQ_site_validation(   # CQMLEM
        start_date='2020-01-02',
        daycount=27,
        simdata_inithour=16,
        GRIDCRO2D_file_dir="E:\SichuanCMAQPMtrends\GRIDCRO2D_d03.nc",
        Combine_file_dir="E:\CQemis_ML2hourly\COMBINE_ACONC_v532_gcc_20200101_202001.nc",
        target_substances=['PM25_TOT'],
        target_substance_obs='PM2.5',
        Molar_mass=0,
        airstation_files_dir=r'E:\全国空气质量\全国站点小时浓度csv_files\\',
        airstation_infofile_dir="E:\全国空气质量\_站点列表\站点列表-2022.02.13起.csv",
        result_pic_types=['line','scatter'],
        out_dir=r'E:\CQemis_ML2hourly\validation\\',
        result_csv_name='validationPara',
        suffix='CQMLEM'
    )

    # CMAQ_site_validation(   # CQMLEM_origin
    #     start_date='2020-01-02',
    #     daycount=27,
    #     simdata_inithour=40,
    #     GRIDCRO2D_file_dir="E:\SichuanCMAQPMtrends\GRIDCRO2D_d03.nc",
    #     Combine_file_dir="E:\SichuanCMAQPMtrends\cctmCombine\CCTM_ISAM_PM25_2020_v54_newem_region.nc",
    #     target_substances=['PM25_BCO','PM25_ICO','PM25_OTH','PM25_CDEM','PM25_CQEM','PM25_SCBEEM','PM25_SCBWEM'],
    #     target_substance_obs='PM2.5',
    #     Molar_mass=0,
    #     airstation_files_dir=r'E:\全国空气质量\全国站点小时浓度csv_files\\',
    #     airstation_infofile_dir="E:\全国空气质量\_站点列表\站点列表-2022.02.13起.csv",
    #     result_pic_types=['line','scatter'],
    #     out_dir=r'E:\CQemis_ML2hourly\validation_origin\\',
    #     result_csv_name='validationPara',
    #     suffix='CQMLEM_orign'
    # )

    pass
