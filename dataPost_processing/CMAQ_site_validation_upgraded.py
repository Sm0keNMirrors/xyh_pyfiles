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



def CMAQ_site_validation(
    start_date='YYYY-MM-DD',
    daycount=5,
    simdata_inithour=16,
    GRIDCRO2D_file_dir="",
    Combine_file_dir="",
    target_substances=None,
    target_substance_obs='',
    Molar_mass=0,
    ifdaily=False,
    validation_freq=None,
    convert_to_ppb=False,
    validate_mda8=False,
    mda8_min_hours=6,
    metric_method="manual",
    metric_names=None,
    obs_unit="ug/m3",
    sim_unit="ppm",
    temperature_K=298.15,
    pressure_atm=1.0,
    airstation_files_dir="",
    airstation_infofile_dir="",
    airstation_selected=None,
    result_pic_types=None,
    result_pic_combine=None,
    out_dir="",
    result_csv_name="",
    suffix="",
    save_site_timeseries=True,
    scatter_xlim=None,
    scatter_ylim=None,
    scatter_density=True,
    close_fig=True,
):
    """
    使用全国公开逐小时站点浓度数据验证 CMAQ 模拟结果。

    升级内容：
    1) 站点筛选方法逻辑保持不变，仅优化代码组织；
    2) validation_freq 可选 hourly / daily；
    3) convert_to_ppb=True 时，气态污染物可转 ppb；
    4) validate_mda8=True 时，执行每日 MDA8 验证，适合 O3；
    5) metric_method 可选 manual / vectorized / sklearn；
    6) 优化输出指标表、逐站时间序列表、line / scatter / scatter_combine 图。


    :param start_date: 验证开始日期，格式为 'YYYY-MM-DD'。
    :param daycount: 验证持续天数。
    :param simdata_inithour: 从 CMAQ 模拟结果中截取验证时段的起始小时索引。
    :param GRIDCRO2D_file_dir: CMAQ 网格经纬度文件 GRIDCRO2D 路径。
    :param Combine_file_dir: CMAQ combine 后浓度文件路径。
    :param target_substances: CMAQ 中参与验证的目标变量名列表，多个变量会相加。
    :param target_substance_obs: 观测 CSV 中对应污染物的 type 名称。
    :param Molar_mass: 气态污染物分子量，设为 0 时自动识别或不转换。
    :param ifdaily: 旧参数，True 表示日均验证，False 表示逐小时验证。
    :param validation_freq: 验证时间尺度，可选 'hourly' 或 'daily'。
    :param convert_to_ppb: 是否将可转换的气态污染物浓度转换为 ppb。
    :param validate_mda8: 是否计算并验证每日 MDA8，通常用于 O3。
    :param mda8_min_hours: 计算 MDA8 时每个 8 小时窗口所需的最少有效小时数。
    :param metric_method: 指标计算方法，可选 'manual'、'vectorized' 或 'sklearn'。
    :param metric_names: 需要输出的评价指标名称列表。None时则输出所有参数。
    :param obs_unit: 观测数据原始单位，可选如 'ug/m3' 或 'ppb'。
    :param sim_unit: CMAQ 模拟数据原始单位，可选如 'ppm'、'ppb' 或 'ug/m3'。
    :param temperature_K: ppb 与 ug/m3 转换时使用的温度，单位 K。
    :param pressure_atm: ppb 与 ug/m3 转换时使用的气压，单位 atm。
    :param airstation_files_dir: 中国环境站点逐小时观测 CSV 文件所在文件夹。
    :param airstation_infofile_dir: 中国环境站点信息 CSV 文件路径。
    :param airstation_selected: 指定只验证的站点名称列表，空列表表示验证范围内全部站点。
    :param result_pic_types: 输出单站图类型列表，可包含 'line'、'scatter'、'csv'。
    :param result_pic_combine: 输出合并图类型列表，可包含 'scatter'。
    :param out_dir: 验证结果输出文件夹。
    :param result_csv_name: 站点评价指标表输出文件名前缀。
    :param suffix: 输出文件名后缀。
    :param save_site_timeseries: 是否输出所有站点的验证时间序列 CSV。
    :param scatter_xlim: 散点图 x 轴范围，None 表示自动。
    :param scatter_ylim: 散点图 y 轴范围，None 表示自动。
    :param scatter_density: 散点图是否按点密度着色。
    :param close_fig: 保存图片后是否自动关闭 figure。
    :return: 返回包含各站点评价指标的 DataFrame。


    """
    import datetime
    import os
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib
    import netCDF4 as nc
    from tqdm import tqdm
    from sklearn.linear_model import LinearRegression
    from scipy.stats import gaussian_kde

    if target_substances is None:
        target_substances = []
    if airstation_selected is None:
        airstation_selected = []
    if result_pic_types is None:
        result_pic_types = []
    if result_pic_combine is None:
        result_pic_combine = []
    if metric_names is None:
        metric_names = ["MAE", "RMSE", "R", "R2", "IOA", "MB", "NMB", "NME", "MFE", "MFB", "FE", "FB", "GE"]

    if validation_freq is None:
        validation_freq = "daily" if ifdaily else "hourly"
    validation_freq = str(validation_freq).lower()
    if validation_freq not in ("hourly", "daily"):
        raise ValueError("validation_freq must be 'hourly' or 'daily'")
    if validate_mda8:
        validation_freq = "daily"

    if not target_substances:
        raise ValueError("target_substances 不能为空")
    if not start_date or start_date == "YYYY-MM-DD":
        raise ValueError("请提供 start_date='YYYY-MM-DD'")
    if not GRIDCRO2D_file_dir:
        raise ValueError("GRIDCRO2D_file_dir 不能为空")
    if not Combine_file_dir:
        raise ValueError("Combine_file_dir 不能为空")
    if not airstation_files_dir:
        raise ValueError("airstation_files_dir 不能为空")
    if not airstation_infofile_dir:
        raise ValueError("airstation_infofile_dir 不能为空")
    if not out_dir:
        raise ValueError("out_dir 不能为空")

    os.makedirs(out_dir, exist_ok=True)
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    gas_mw_map = {
        "O3": 48.0, "NO2": 46.0055, "NO": 30.0061, "NOX": 46.0055,
        "SO2": 64.066, "CO": 28.0101, "NH3": 17.031,
        "HCHO": 30.026, "FORM": 30.026,
    }
    particle_keys = {"PM25", "PM2.5", "PM10", "PM1", "PM25_TOT", "PM2_5"}

    def _norm_key(name):
        key = re.sub(r"[^A-Za-z0-9.]+", "", str(name).upper())
        aliases = {"PM25": "PM25", "PM2.5": "PM2.5", "OZONE": "O3"}
        return aliases.get(key, key)

    def _get_mw():
        if Molar_mass not in (None, 0):
            return float(Molar_mass)
        return gas_mw_map.get(_norm_key(target_substance_obs or target_substances[0]), None)

    def _is_particle():
        key_obs = _norm_key(target_substance_obs)
        keys_mod = {_norm_key(x) for x in target_substances}
        return key_obs in particle_keys or bool(keys_mod & particle_keys)

    def _ppb_factor_from_ugm3(mw):
        r_atm = 0.082057
        molar_volume_l = r_atm * float(temperature_K) / float(pressure_atm)
        return molar_volume_l / float(mw)

    def _convert_units(obs_arr, sim_arr):
        obs = np.asarray(obs_arr, dtype=float)
        sim = np.asarray(sim_arr, dtype=float)
        mw = _get_mw()
        can_convert = (mw is not None) and (not _is_particle())

        if can_convert:
            if str(sim_unit).lower() in ("ppm", "ppmv"):
                sim_ugm3 = sim * float(mw) / 22.4 * 1000.0
            elif str(sim_unit).lower() in ("ppb", "ppbv"):
                sim_ugm3 = sim / _ppb_factor_from_ugm3(mw)
            else:
                sim_ugm3 = sim

            if str(obs_unit).lower() in ("ppb", "ppbv"):
                obs_ugm3 = obs / _ppb_factor_from_ugm3(mw)
            else:
                obs_ugm3 = obs

            if convert_to_ppb:
                fac = _ppb_factor_from_ugm3(mw)
                return obs_ugm3 * fac, sim_ugm3 * fac, "ppb"
            return obs_ugm3, sim_ugm3, r"μg/m$^{3}$"

        return obs, sim, r"μg/m$^{3}$"

    def _hourly_time_index():
        return pd.date_range(start=pd.Timestamp(start_date), periods=daycount * 24, freq="h")

    def _daily_mean(values):
        values = np.asarray(values, dtype=float)[:daycount * 24]
        n_day = len(values) // 24
        if n_day == 0:
            return np.array([], dtype=float)
        arr2d = values[:n_day * 24].reshape(n_day, 24)
        with np.errstate(all="ignore"):
            return np.nanmean(arr2d, axis=1)

    def _daily_mda8(values):
        values = np.asarray(values, dtype=float)[:daycount * 24]
        n_day = len(values) // 24
        out = []
        for d in range(n_day):
            day_vals = values[d * 24:(d + 1) * 24]
            roll = []
            for h in range(17):
                win = day_vals[h:h + 8]
                if np.isfinite(win).sum() >= mda8_min_hours:
                    roll.append(np.nanmean(win))
            out.append(np.nanmax(roll) if roll else np.nan)
        return np.asarray(out, dtype=float)

    def _aggregate(obs_hour, sim_hour):
        obs_hour = np.asarray(obs_hour, dtype=float)
        sim_hour = np.asarray(sim_hour, dtype=float)
        t_hour = _hourly_time_index()
        n = min(len(obs_hour), len(sim_hour), len(t_hour))
        obs_hour, sim_hour, t_hour = obs_hour[:n], sim_hour[:n], t_hour[:n]

        if validate_mda8:
            obs_use = _daily_mda8(obs_hour)
            sim_use = _daily_mda8(sim_hour)
            t_use = pd.date_range(start=pd.Timestamp(start_date), periods=len(obs_use), freq="D")
            label = "Daily MDA8"
        elif validation_freq == "daily":
            obs_use = _daily_mean(obs_hour)
            sim_use = _daily_mean(sim_hour)
            t_use = pd.date_range(start=pd.Timestamp(start_date), periods=len(obs_use), freq="D")
            label = "Daily mean"
        else:
            obs_use, sim_use, t_use = obs_hour, sim_hour, t_hour
            label = "Hourly"

        n2 = min(len(obs_use), len(sim_use), len(t_use))
        return np.asarray(obs_use[:n2], dtype=float), np.asarray(sim_use[:n2], dtype=float), pd.DatetimeIndex(t_use[:n2]), label

    def _paired_clean(sim, obs):
        sim = np.asarray(sim, dtype=float)
        obs = np.asarray(obs, dtype=float)
        n = min(len(sim), len(obs))
        sim, obs = sim[:n], obs[:n]
        mask = np.isfinite(sim) & np.isfinite(obs)
        return sim[mask], obs[mask], mask

    def _metrics_vectorized(sim, obs):
        sim, obs, _ = _paired_clean(sim, obs)
        out = {m: np.nan for m in metric_names}
        out["N"] = int(len(obs))
        if len(obs) == 0:
            return out

        diff = sim - obs
        abs_diff = np.abs(diff)
        obs_mean = np.mean(obs)

        if "MAE" in out:
            out["MAE"] = float(np.mean(abs_diff))
        if "RMSE" in out:
            out["RMSE"] = float(np.sqrt(np.mean(diff ** 2)))
        if "MB" in out:
            out["MB"] = float(np.mean(diff))
        if "R" in out and len(obs) >= 2 and np.std(obs) > 0 and np.std(sim) > 0:
            out["R"] = float(np.corrcoef(sim, obs)[0, 1])
        if "R2" in out and len(obs) >= 2 and np.std(obs) > 0 and np.std(sim) > 0:
            out["R2"] = float(np.corrcoef(sim, obs)[0, 1] ** 2)

        denom = np.sum((np.abs(sim - obs_mean) + np.abs(obs - obs_mean)) ** 2)
        if "IOA" in out:
            out["IOA"] = float(1.0 - np.sum(diff ** 2) / denom) if denom > 0 else np.nan

        sum_obs = np.sum(obs)
        if sum_obs != 0:
            if "NMB" in out:
                out["NMB"] = float(np.sum(diff) / sum_obs * 100.0)
            if "NME" in out:
                out["NME"] = float(np.sum(abs_diff) / sum_obs * 100.0)

        den_mean = (obs + sim) / 2.0
        m = np.isfinite(den_mean) & (den_mean != 0)
        if np.any(m):
            if "MFE" in out:
                out["MFE"] = float(np.mean(np.abs(diff[m]) / den_mean[m]) * 100.0)
            if "MFB" in out:
                out["MFB"] = float(np.mean(diff[m] / den_mean[m]) * 100.0)

        den_sum = obs + sim
        m = np.isfinite(den_sum) & (den_sum != 0)
        if np.any(m):
            if "FE" in out:
                out["FE"] = float(np.mean(2.0 * np.abs(diff[m]) / den_sum[m]) * 100.0)
            if "FB" in out:
                out["FB"] = float(np.mean(2.0 * diff[m] / den_sum[m]) * 100.0)
            if "GE" in out:
                out["GE"] = float(np.mean(2.0 * np.abs(diff[m] / den_sum[m])) * 100.0)

        return out

    def _metrics_manual(sim, obs):
        sim_clean, obs_clean, _ = _paired_clean(sim, obs)
        out = {m: np.nan for m in metric_names}
        out["N"] = int(len(obs_clean))
        if len(obs_clean) == 0:
            return out

        calc_map = {
            "MAE": calcuMAE, "RMSE": calcuRSME, "R": calcuR, "IOA": calcuIOA,
            "NMB": calcuNMB, "NME": calcuNME, "MFE": calcuMFE, "MFB": calcuMFB,
            "FE": calcuFE, "FB": calcuFB, "GE": calcuGE,
        }
        for name in metric_names:
            if name in calc_map:
                try:
                    out[name] = float(calc_map[name](sim_clean, obs_clean))
                except Exception:
                    out[name] = np.nan
        if "MB" in metric_names:
            try:
                out["MB"] = float(calcuMB(sim_clean, obs_clean))
            except Exception:
                out["MB"] = np.nan
        if "R2" in metric_names and np.isfinite(out.get("R", np.nan)):
            out["R2"] = float(out["R"] ** 2)
        return out

    def _metrics_sklearn(sim, obs):
        out = _metrics_vectorized(sim, obs)
        sim_clean, obs_clean, _ = _paired_clean(sim, obs)
        if len(obs_clean) == 0:
            return out
        try:
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            if "MAE" in out:
                out["MAE"] = float(mean_absolute_error(obs_clean, sim_clean))
            if "RMSE" in out:
                try:
                    out["RMSE"] = float(mean_squared_error(obs_clean, sim_clean, squared=False))
                except TypeError:
                    out["RMSE"] = float(np.sqrt(mean_squared_error(obs_clean, sim_clean)))
            if "R2" in out and len(obs_clean) >= 2:
                out["R2"] = float(r2_score(obs_clean, sim_clean))
        except Exception:
            pass
        return out

    def _calc_metrics(sim, obs):
        method = str(metric_method).lower()
        if method == "manual":
            return _metrics_manual(sim, obs)
        if method in ("vectorized", "numpy", "np"):
            return _metrics_vectorized(sim, obs)
        if method in ("sklearn", "scikit-learn"):
            return _metrics_sklearn(sim, obs)
        raise ValueError("metric_method must be 'manual', 'vectorized', or 'sklearn'")

    def _read_obs_for_station(air_csv_data, station_col):
        vals = []
        for df in air_csv_data:
            subdf = df[df['type'] == target_substance_obs] if 'type' in df.columns else pd.DataFrame()
            if subdf.empty or station_col not in subdf.columns:
                vals.extend([np.nan] * 24)
                continue
            subdf = subdf[['hour', station_col]].copy()
            subdf['hour'] = pd.to_numeric(subdf['hour'], errors='coerce')
            subdf[station_col] = pd.to_numeric(subdf[station_col], errors='coerce')
            subdf = subdf.dropna(subset=['hour'])
            subdf['hour'] = subdf['hour'].astype(int)
            hour_map = dict(zip(subdf['hour'], subdf[station_col]))
            vals.extend([hour_map.get(h, np.nan) for h in range(24)])
        return np.asarray(vals, dtype=float)

    def _safe_station_filename(stname):
        pin = chinese_to_pinyin(stname, mode="abbr", uppercase=True, keep_non_chinese=True)
        pin = re.sub(r"[^A-Za-z0-9_.-]+", "_", pin)
        return pin or "station"

    def _plot_line(time_use, obs_use, sim_use, stname, stcity, metrics, unit_label, validation_label):
        pics_dir = os.path.join(out_dir, "pics_line")
        os.makedirs(pics_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5.8, 2.4), dpi=220)
        ax.plot(time_use, obs_use, linewidth=1.0, label="Obs", color="#258080")
        ax.plot(time_use, sim_use, linewidth=1.0, label="Sim", color="red", alpha=0.85)
        ax.set_ylabel(unit_label, fontsize=9)
        ax.set_xlabel("Date", fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.tick_params(axis='x', labelsize=7, rotation=0)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(True, which='major', linewidth=0.3, alpha=0.35)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper right', frameon=False, fontsize=7, handlelength=2.2, borderaxespad=0.2)

        metric_txt = []
        for k in ["IOA", "MB", "RMSE", "R", "NMB"]:
            if k in metrics and np.isfinite(metrics[k]):
                u = f" {unit_label}" if k in ("MB", "RMSE", "MAE") else ""
                metric_txt.append(f"{k}={metrics[k]:.2f}{u}")
        ax.text(
            0.0, 1.08, "   ".join(metric_txt),
            transform=ax.transAxes,
            ha='left', va='bottom',
            fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8)
        )
        ax.set_title(f"{chinese_to_pinyin(stcity, mode='full')} | {validation_label}", loc='right', fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(pics_dir, f"{_safe_station_filename(stname)}_{suffix}.png"), bbox_inches="tight", dpi=600)
        if close_fig:
            plt.close(fig)

    def _scatter_core(ax, obs_use, sim_use, stname, stcity, unit_label, metrics):
        sim_clean, obs_clean, _ = _paired_clean(sim_use, obs_use)
        if len(obs_clean) == 0:
            ax.text(0.5, 0.5, "No valid pair", transform=ax.transAxes, ha="center", va="center")
            return

        z = None
        if scatter_density and len(obs_clean) >= 3:
            try:
                xy = np.vstack([obs_clean, sim_clean])
                z = gaussian_kde(xy)(xy)
            except Exception:
                z = None

        if z is not None:
            ax.scatter(obs_clean, sim_clean, c=z, cmap='viridis', alpha=0.65, s=16)
        else:
            ax.scatter(obs_clean, sim_clean, alpha=0.65, s=16)

        lim_min = min(float(np.nanmin(obs_clean)), float(np.nanmin(sim_clean)), 0.0)
        lim_max = max(float(np.nanmax(obs_clean)), float(np.nanmax(sim_clean)))
        if lim_max == lim_min:
            lim_max = lim_min + 1.0

        if scatter_xlim is not None:
            ax.set_xlim(scatter_xlim)
            x_line = np.linspace(scatter_xlim[0], scatter_xlim[1], 100)
        else:
            ax.set_xlim(lim_min, lim_max * 1.05)
            x_line = np.linspace(lim_min, lim_max * 1.05, 100)

        if scatter_ylim is not None:
            ax.set_ylim(scatter_ylim)
        else:
            ax.set_ylim(lim_min, lim_max * 1.05)

        ax.plot(x_line, x_line, color="black", linestyle="--", linewidth=1.0, label="1:1")

        if len(obs_clean) >= 2 and np.nanstd(obs_clean) > 0:
            model = LinearRegression()
            model.fit(obs_clean.reshape(-1, 1), sim_clean)
            y_fit = model.predict(x_line.reshape(-1, 1))
            ax.plot(x_line, y_fit, color='red', linewidth=1.5, label=f"y={model.coef_[0]:.2f}x+{model.intercept_:.2f}")

        ax.set_xlabel(f"Obs ({unit_label})")
        ax.set_ylabel(f"Sim ({unit_label})")
        ax.grid(True, linewidth=0.3, alpha=0.35)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        stats = []
        for k in ["R", "RMSE", "NMB"]:
            if k in metrics and np.isfinite(metrics[k]):
                stats.append(f"{k}={metrics[k]:.2f}")
        ax.set_title(f"{chinese_to_pinyin(stcity, mode='full')}\n" + "  ".join(stats), fontsize=8)
        ax.legend(frameon=False, fontsize=7, loc="lower right")

    def _plot_scatter(obs_use, sim_use, stname, stcity, metrics, unit_label):
        pics_dir = os.path.join(out_dir, "pics_scatter")
        os.makedirs(pics_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=220)
        _scatter_core(ax, obs_use, sim_use, stname, stcity, unit_label, metrics)
        fig.tight_layout()
        fig.savefig(os.path.join(pics_dir, f"{_safe_station_filename(stname)}_{suffix}.png"), bbox_inches="tight", dpi=500)
        if close_fig:
            plt.close(fig)

    GRIDCRO2D = nc.Dataset(GRIDCRO2D_file_dir)
    CMAQoutf = nc.Dataset(Combine_file_dir, "r")

    try:
        var_lon = np.array(GRIDCRO2D.variables['LON'][:][0])
        var_lat = np.array(GRIDCRO2D.variables['LAT'][:][0])
        lonmin, latmax, lonmax, latmin = var_lon.min(), var_lat.max(), var_lon.max(), var_lat.min()

        # ① 站点筛选方法逻辑不改：先按 CMAQ 范围筛选，再按 airstation_selected 保留指定站点。
        airStation_locations_all = getAirStationsFromLatLon(latmax, latmin, lonmin, lonmax, airstation_infofile_dir)
        if airstation_selected != []:
            airStation_locations = {name: info for name, info in airStation_locations_all.items() if name in airstation_selected}
        else:
            airStation_locations = airStation_locations_all

        print(f"[INFO] selected stations: {len(airStation_locations)}")
        if len(airStation_locations) == 0:
            print("[WARN] 没有筛选到站点，请检查 GRIDCRO2D 范围、站点信息文件或 airstation_selected。")

        validation_dates = generate_date_list(start_date, daycount)
        airStation_csv_list = []
        for date in validation_dates:
            date2 = str(date).split(' ')[0]
            filedate = ''.join(date2.split('-'))
            airStation_csv_list.append(os.path.join(airstation_files_dir, f"china_sites_{filedate}.csv"))

        air_csv_data = []
        for fp in airStation_csv_list:
            if os.path.exists(fp):
                air_csv_data.append(pd.read_csv(fp))
            else:
                print(f"[WARN] obs csv not found: {fp}")
                air_csv_data.append(pd.DataFrame(columns=["type", "hour"]))

        substance = None
        for sub in target_substances:
            if sub not in CMAQoutf.variables:
                raise KeyError(f"{sub} 不在 Combine_file_dir 的变量中")
            arr = np.array(CMAQoutf.variables[sub][:], dtype=np.float32)
            substance = arr if substance is None else substance + arr

        CMAQXLAT = np.array(GRIDCRO2D.variables['LAT'][:][0])
        CMAQXLONG = np.array(GRIDCRO2D.variables['LON'][:][0])

        result_rows = []
        series_rows = []
        combine_scatter_data = []

        for stname, airstation in tqdm(airStation_locations.items(), desc='处理所有有效站点...'):
            stlon = float(airstation[0])
            stlat = float(airstation[1])
            station_col = airstation[2]
            stcity = airstation[3]

            nearpos = getNearestPos(stlat, stlon, CMAQXLAT, CMAQXLONG)
            nearlat = int(nearpos[1])
            nearlon = int(nearpos[2])

            sim_raw = substance[:, 0, nearlat, nearlon]
            sim_hour_raw = np.asarray(sim_raw[simdata_inithour:simdata_inithour + 24 * daycount], dtype=float)
            obs_hour_raw = _read_obs_for_station(air_csv_data, station_col)

            if np.all(~np.isfinite(obs_hour_raw)):
                continue

            obs_hour, sim_hour, unit_label = _convert_units(obs_hour_raw, sim_hour_raw)
            obs_use, sim_use, time_use, validation_label = _aggregate(obs_hour, sim_hour)

            if len(obs_use) == 0 or np.all(~np.isfinite(obs_use)):
                continue

            metrics = _calc_metrics(sim_use, obs_use)

            row = {
                "站点": stname,
                "城市": stcity,
                "站点经度": stlon,
                "站点纬度": stlat,
                "站点编码": station_col,
                "最近网格ROW": nearlat,
                "最近网格COL": nearlon,
                "验证频率": validation_label,
                "单位": unit_label,
                "指标算法": metric_method,
                "是否ppb转换": bool(convert_to_ppb),
                "是否MDA8": bool(validate_mda8),
            }
            for name in ["N"] + metric_names:
                if name in metrics:
                    row[name] = metrics[name]
            result_rows.append(row)

            if save_site_timeseries or "csv" in result_pic_types:
                for t, obs_v, sim_v in zip(time_use, obs_use, sim_use):
                    series_rows.append({
                        "站点": stname, "城市": stcity, "datetime": t,
                        "obs": obs_v, "sim": sim_v, "unit": unit_label,
                        "validation_freq": validation_label,
                    })

            if "line" in result_pic_types:
                _plot_line(time_use, obs_use, sim_use, stname, stcity, metrics, unit_label, validation_label)

            if "scatter" in result_pic_types:
                _plot_scatter(obs_use, sim_use, stname, stcity, metrics, unit_label)

            if "scatter" in result_pic_combine:
                combine_scatter_data.append((obs_use, sim_use, stname, stcity, metrics, unit_label))

        result_csv_data = pd.DataFrame(result_rows)
        result_path = os.path.join(out_dir, f"{result_csv_name or 'validation_metrics'}_{suffix}.csv")
        result_csv_data.to_csv(result_path, index=False, encoding='utf-8-sig')
        print(f"[OK] metrics csv -> {result_path}")

        if save_site_timeseries or "csv" in result_pic_types:
            ts_dir = os.path.join(out_dir, "csv_timeseries")
            os.makedirs(ts_dir, exist_ok=True)
            ts_path = os.path.join(ts_dir, f"site_timeseries_{suffix}.csv")
            pd.DataFrame(series_rows).to_csv(ts_path, index=False, encoding='utf-8-sig')
            print(f"[OK] site time series csv -> {ts_path}")

        if "scatter" in result_pic_combine and len(combine_scatter_data) > 0:
            nn, mm = calculate_figRowCol_size(len(combine_scatter_data))
            fig_c, axs_c = plt.subplots(nn, mm, figsize=(4.2 * mm, 4.2 * nn), dpi=220, squeeze=False)
            for i, (obs_use, sim_use, stname, stcity, metrics, unit_label) in enumerate(combine_scatter_data):
                rr, cc = i // mm, i % mm
                _scatter_core(axs_c[rr, cc], obs_use, sim_use, stname, stcity, unit_label, metrics)

            for j in range(len(combine_scatter_data), nn * mm):
                rr, cc = j // mm, j % mm
                fig_c.delaxes(axs_c[rr, cc])

            fig_c.supxlabel(f"Observed ({combine_scatter_data[0][5]})", fontsize=14)
            fig_c.supylabel(f"Simulated ({combine_scatter_data[0][5]})", fontsize=14)
            fig_c.tight_layout()
            combine_path = os.path.join(out_dir, f"scatter_combine_{suffix}.png")
            fig_c.savefig(combine_path, bbox_inches="tight", dpi=500)
            if close_fig:
                plt.close(fig_c)
            print(f"[OK] combined scatter -> {combine_path}")

        return result_csv_data

    finally:
        try:
            GRIDCRO2D.close()
        except Exception:
            pass
        try:
            CMAQoutf.close()
        except Exception:
            pass


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
