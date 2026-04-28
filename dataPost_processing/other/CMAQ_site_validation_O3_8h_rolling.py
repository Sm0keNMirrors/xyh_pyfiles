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
    return ALL1 / ALL2 * 100

def calcuNME(modData, obsData):
    N = len(modData) if len(modData) < len(obsData) else len(obsData)  # 不过限度
    ALL1 = 0
    ALL2 = 0
    for i in range(0, N):
        if np.isnan(obsData[i]) == True:  # 判断nan
            continue
        ALL1 += abs(modData[i] - obsData[i])
        ALL2 += obsData[i]
    return ALL1 / ALL2 * 100

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

def chinese_to_pinyin(
    text: str,
    mode: str = "abbr",          # "abbr"=首字母简写, "full"=全拼
    uppercase: bool = True,      # 是否大写（对 abbr 更常用）
    separator: str = "",         # full 模式拼音之间的分隔符，如 "_" / " " / ""
    keep_non_chinese: bool = False,  # 是否保留非中文字符（数字/字母/符号）
) -> str:
    """
    将中文字符串转换为拼音（全拼 or 首字母简写）。

    参数：
        text: 输入字符串（中文为主）
        mode:
            - "abbr": 每个汉字取拼音首字母（缩写）
            - "full": 每个汉字取完整拼音（全拼）
        uppercase: 是否转大写（abbr常用；full也可用）
        separator: full 模式下拼音间的连接符（abbr 模式会忽略该参数）
        keep_non_chinese: 是否保留非中文字符

    返回：
        转换后的拼音字符串
    """
    from pypinyin import pinyin, Style
    import re

    if not isinstance(text, str):
        text = str(text)

    if mode not in ("abbr", "full"):
        raise ValueError("mode must be 'abbr' or 'full'")

    style = Style.FIRST_LETTER if mode == "abbr" else Style.NORMAL
    py_list = pinyin(text, style=style, strict=False)

    out = []
    for ch, item in zip(text, py_list):
        token = item[0]

        is_chinese = bool(re.match(r'[\u4e00-\u9fff]', ch))

        if is_chinese:
            out.append(token.upper() if uppercase else token.lower())
        else:
            if keep_non_chinese:
                # 非中文字符：原样保留（字母按 uppercase 处理）
                if token.isalpha():
                    out.append(token.upper() if uppercase else token.lower())
                else:
                    out.append(token)
            # 否则忽略

    if mode == "full":
        return separator.join(out)
    else:
        return "".join(out)



def rolling_mean_8h_keep_length(arr, min_periods=1):
    """
    对逐小时序列计算 8h 滑动平均，返回与原序列等长的结果。
    默认使用当前时刻及其之前连续最多 8 小时做平均：
    y[t] = mean(x[max(0, t-7):t+1])
    NaN 自动跳过；若窗口内全为 NaN，则结果为 NaN。
    """
    s = pd.Series(np.asarray(arr, dtype=np.float32))
    return s.rolling(window=8, min_periods=min_periods).mean().to_numpy(dtype=np.float32)


def CMAQ_site_validation(
    start_date='YYYY-MM-DD',
    daycount=5, #
    simdata_inithour=16, #
    GRIDCRO2D_file_dir = "",
    Combine_file_dir = "",
    target_substances = [], #
    target_substance_obs = '', #
    Molar_mass = 0, #
    ifdaily=False,
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
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib
    import netCDF4 as nc
    """
    使用全国公开csv观测数据验证CMAQ模拟结果（支持 O3 逐小时 8h 滑动平均验证）

    更新说明：
    不再逐站点分别计算验证参数，而是先对所有有效站点的逐小时观测序列做平均，
    对应模拟格点的逐小时序列也做平均，得到总体观测序列和总体模拟序列，
    再基于这两条总体序列计算最终验证参数。

    函数参数保持原有调用方式不变。
    输出新增：
    1. 总体逐小时/逐日观测-模拟序列 csv
    2. 总体验证参数 csv
    3. return {'obs_series': ..., 'sim_series': ..., 'metrics': ...}
    """
    year = str(datetime.datetime.strptime(start_date, '%Y-%m-%d')).split('-')[0]
    if out_dir and (os.path.exists(out_dir) is False):
        os.mkdir(out_dir)
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    GRIDCRO2D = nc.Dataset(GRIDCRO2D_file_dir)
    CMAQoutf = nc.Dataset(Combine_file_dir, "r")

    var_lon = np.array(GRIDCRO2D.variables['LON'][:][0])
    var_lat = np.array(GRIDCRO2D.variables['LAT'][:][0])
    lonmin, latmax, lonmax, latmin = (var_lon.min(), var_lat.max(), var_lon.max(), var_lat.min())

    airStation_locations_ = getAirStationsFromLatLon(latmax, latmin, lonmin, lonmax, airstation_infofile_dir)
    airStation_locations = {}
    if airstation_selected != []:
        for x in airStation_locations_:
            if x in airstation_selected:
                airStation_locations.update({x: airStation_locations_[x]})
    else:
        airStation_locations = airStation_locations_
    print(airStation_locations)

    airStation_csv_list = []
    validation_dates = generate_date_list(start_date, daycount)
    for date in validation_dates:
        date2 = str(date).split(' ')[0]
        filedate = f'{date2.split("-")[0]}{date2.split("-")[1]}{date2.split("-")[2]}'
        airStation_csv_list.append(f"{airstation_files_dir}china_sites_{filedate}.csv")

    substance = None
    for sub in target_substances:
        arr = np.array(CMAQoutf.variables[sub][:], dtype=np.float32)
        if substance is None:
            substance = arr
        else:
            substance += arr

    CMAQXLAT = np.array(GRIDCRO2D.variables['LAT'][:][0])
    CMAQXLONG = np.array(GRIDCRO2D.variables['LON'][:][0])
    air_csv_data = [pd.read_csv(m) for m in airStation_csv_list]

    all_obs_series = []
    all_sim_series = []
    used_station_names = []
    used_station_meta = []

    for stname, airstation in tqdm(airStation_locations.items(), desc='汇总所有有效站点序列...'):
        nearpos = getNearestPos(airstation[1], airstation[0], CMAQXLAT, CMAQXLONG)
        nearlat = nearpos[1]
        nearlon = nearpos[2]

        target = substance[:, 0, nearlat, nearlon]
        if Molar_mass != 0:
            target = target * Molar_mass / 22.4 * 1000
        sim_station = np.asarray(target[simdata_inithour:simdata_inithour + 24 * daycount], dtype=np.float32)

        obs_station_pre = []
        station_col = airstation[2]
        for df in air_csv_data:
            subdf = df[df['type'] == target_substance_obs]
            if subdf.empty or station_col not in subdf.columns:
                obs_station_pre.extend([np.nan] * 24)
                continue
            subdf = subdf[['hour', station_col]].copy()
            subdf['hour'] = pd.to_numeric(subdf['hour'], errors='coerce')
            subdf = subdf.dropna(subset=['hour'])
            subdf['hour'] = subdf['hour'].astype(int)
            subdf[station_col] = pd.to_numeric(subdf[station_col], errors='coerce')
            hour_map = dict(zip(subdf['hour'], subdf[station_col]))
            for hour in range(24):
                obs_station_pre.append(hour_map.get(hour, np.nan))

        obs_station = np.asarray(obs_station_pre, dtype=np.float32)
        if np.isnan(obs_station).all():
            continue

        n_common = min(len(obs_station), len(sim_station))
        obs_station = obs_station[:n_common]
        sim_station = sim_station[:n_common]

        all_obs_series.append(obs_station)
        all_sim_series.append(sim_station)
        used_station_names.append(stname)
        used_station_meta.append({
            '站点': stname,
            '站点经度': airstation[0],
            '站点纬度': airstation[1],
            '城市': airstation[3]
        })

    if len(all_obs_series) == 0:
        raise ValueError('没有找到可用于验证的有效站点观测序列。')

    all_obs_array = np.vstack(all_obs_series)
    all_sim_array = np.vstack(all_sim_series)

    overall_obs_hourly = np.nanmean(all_obs_array, axis=0)
    overall_sim_hourly = np.nanmean(all_sim_array, axis=0)

    xdate = generate_date_list_withhour(start_date, daycount * 24)
    xdate.pop()
    n_common = min(len(xdate), len(overall_obs_hourly), len(overall_sim_hourly))
    xdate = xdate[:n_common]
    overall_obs_hourly = overall_obs_hourly[:n_common]
    overall_sim_hourly = overall_sim_hourly[:n_common]

    obs_eval = np.asarray(overall_obs_hourly, dtype=np.float32)
    sim_eval = np.asarray(overall_sim_hourly, dtype=np.float32)
    x_eval = np.asarray(xdate)

    # O3_8h 表示逐小时 8h 滑动平均，不改变序列长度
    if str(target_substance_obs).upper() == 'O3_8H':
        obs_eval = rolling_mean_8h_keep_length(obs_eval)
        sim_eval = rolling_mean_8h_keep_length(sim_eval)

    if ifdaily:
        n_day_valid = (min(len(obs_eval), len(sim_eval)) // 24)
        obs_eval = obs_eval[:n_day_valid * 24].reshape(n_day_valid, 24)
        sim_eval = sim_eval[:n_day_valid * 24].reshape(n_day_valid, 24)
        obs_eval = np.nanmean(obs_eval, axis=1)
        sim_eval = np.nanmean(sim_eval, axis=1)
        x_eval = np.asarray(xdate[:n_day_valid * 24])[::24]
        xlabel = 'Date (Daily)'
    else:
        xlabel = 'Date'

    mask = np.isfinite(obs_eval) & np.isfinite(sim_eval)
    obs_valid = obs_eval[mask]
    sim_valid = sim_eval[mask]

    if len(obs_valid) == 0:
        raise ValueError('总体观测序列与总体模拟序列没有可配对的有效数据，无法计算验证参数。')

    substanceMAE = calcuMAE(sim_eval, obs_eval)
    substanceRSME = calcuRSME(sim_eval, obs_eval)
    substanceR = calcuR(sim_eval, obs_eval)
    substanceIOA = calcuIOA(sim_eval, obs_eval)
    substanceNMB = calcuNMB(sim_eval, obs_eval)
    substanceNME = calcuNME(sim_eval, obs_eval)
    substanceMFE = calcuMFE(sim_eval, obs_eval)
    substanceMFB = calcuMFB(sim_eval, obs_eval)
    substanceFE = calcuFE(sim_eval, obs_eval)
    substanceFB = calcuFB(sim_eval, obs_eval)
    substanceMB = calcuMB(sim_eval, obs_eval)

    metrics_dict = {
        '站点数': len(used_station_names),
        'MAE': substanceMAE,
        'RMSE': substanceRSME,
        'R': substanceR,
        'IOA': substanceIOA,
        'MB': substanceMB,
        'NMB': substanceNMB,
        'NME': substanceNME,
        'MFE': substanceMFE,
        'MFB': substanceMFB,
        'FE': substanceFE,
        'FB': substanceFB,
    }

    sequence_df = pd.DataFrame({
        'datetime': x_eval,
        'obs_overall': obs_eval,
        'sim_overall': sim_eval,
    })
    if str(target_substance_obs).upper() == 'O3_8H':
        sequence_df.rename(columns={'obs_overall': 'obs_overall_O3_8h', 'sim_overall': 'sim_overall_O3_8h'}, inplace=True)
    metrics_df = pd.DataFrame([metrics_dict])
    stations_df = pd.DataFrame(used_station_meta)

    if out_dir:
        sequence_csv = f'{out_dir}{result_csv_name}_overall_series_{suffix}.csv' if result_csv_name else f'{out_dir}overall_series_{suffix}.csv'
        metrics_csv = f'{out_dir}{result_csv_name}_{suffix}.csv' if result_csv_name else f'{out_dir}validation_metrics_{suffix}.csv'
        stations_csv = f'{out_dir}{result_csv_name}_used_stations_{suffix}.csv' if result_csv_name else f'{out_dir}used_stations_{suffix}.csv'
        sequence_df.to_csv(sequence_csv, index=False, encoding='utf-8-sig')
        metrics_df.to_csv(metrics_csv, index=False, encoding='utf-8-sig')
        stations_df.to_csv(stations_csv, index=False, encoding='utf-8-sig')

    if 'line' in result_pic_types:
        if os.path.exists(f'{out_dir}pics_line\\') is False:
            os.mkdir(f'{out_dir}pics_line\\')
        fig2 = plt.figure(figsize=(6, 2.5), dpi=200)
        ax2 = fig2.add_subplot(111)
        ax2.plot(x_eval, obs_eval, linewidth=0.9, label='Overall Obs')
        ax2.plot(x_eval, sim_eval, linewidth=0.9, label='Overall Sim', alpha=0.85)
        ax2.set_xlabel(xlabel, fontsize=9)
        ylabel = target_substance_obs if target_substance_obs else 'Concentration'
        if str(target_substance_obs).upper() == 'O3_8H':
            ylabel = 'O3_8h rolling mean'
        ax2.set_ylabel(ylabel, fontsize=9)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax2.tick_params(axis='x', labelsize=6, rotation=0)
        ax2.tick_params(axis='y', labelsize=6)
        ax2.grid(True, which='major', linewidth=0.3, alpha=0.35)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.legend(loc='upper right', frameon=False, fontsize=6)
        metric_txt = f"N={len(used_station_names)}  R={substanceR:.2f}  IOA={substanceIOA:.2f}  MB={substanceMB:.2f}  RMSE={substanceRSME:.2f}"
        ax2.text(0.0, 1.08, metric_txt, transform=ax2.transAxes, ha='left', va='bottom', fontsize=6,
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8))
        plot_title = 'Overall Site-Averaged Validation'
        if str(target_substance_obs).upper() == 'O3_8H':
            plot_title = 'Overall Site-Averaged 8h Rolling O3 Validation'
        ax2.set_title(plot_title, loc='right', fontsize=7)
        fig2.tight_layout()
        plt.savefig(f'{out_dir}pics_line\\overall_{suffix}.png', bbox_inches='tight', dpi=600)
        plt.close(fig2)

    if 'scatter' in result_pic_types:
        if os.path.exists(f'{out_dir}pics_scatter\\') is False:
            os.mkdir(f'{out_dir}pics_scatter\\')
        plt.figure(figsize=(6, 6), dpi=200)
        observed_clean = np.asarray(obs_valid, dtype=np.float32)
        simulated_clean = np.asarray(sim_valid, dtype=np.float32)
        if len(observed_clean) > 1:
            xy = np.vstack([observed_clean, simulated_clean])
            z = gaussian_kde(xy)(xy)
            plt.scatter(observed_clean, simulated_clean, c=z, cmap='viridis', alpha=0.6)
            model = LinearRegression()
            model.fit(observed_clean.reshape(-1, 1), simulated_clean)
            x_fit = np.linspace(min(observed_clean), max(observed_clean), 100).reshape(-1, 1)
            y_fit = model.predict(x_fit)
            slope = model.coef_[0]
            intercept = model.intercept_
            equation = f'y = {slope:.2f}x + {intercept:.2f}'
            plt.plot(x_fit, y_fit, color='red', linewidth=2, label=equation)
            plt.legend()
        else:
            plt.scatter(observed_clean, simulated_clean, alpha=0.6)
        plt.xlabel('Overall Obs')
        plt.ylabel('Overall Sim')
        plt.title('Overall Scatter and Regression')
        plt.savefig(f'{out_dir}pics_scatter\\overall_{suffix}.png')
        plt.close()

    return {
        'obs_series': obs_eval,
        'sim_series': sim_eval,
        'metrics': metrics_dict,
        'used_stations': used_station_names,
    }


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

    # CMAQ_site_validation(   # CQMLEM
    #     start_date='2020-01-02',
    #     daycount=27,
    #     simdata_inithour=16,
    #     GRIDCRO2D_file_dir="E:\SichuanCMAQPMtrends\GRIDCRO2D_d03.nc",
    #     Combine_file_dir="E:\CQemis_ML2hourly\COMBINE_ACONC_v532_gcc_20200101_202001.nc",
    #     target_substances=['PM25_TOT'],
    #     target_substance_obs='PM2.5',
    #     Molar_mass=0,
    #     airstation_files_dir=r'E:\全国空气质量\全国站点小时浓度csv_files\\',
    #     airstation_infofile_dir="E:\全国空气质量\_站点列表\站点列表-2022.02.13起.csv",
    #     result_pic_types=['line','scatter'],
    #     out_dir=r'E:\CQemis_ML2hourly\validation\\',
    #     result_csv_name='validationPara',
    #     suffix='CQMLEM'
    # )

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
