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
    grid_match_method="nearest",
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
    stream_obs_csv=True,
    timeseries_output_mode="per_station",
    max_combine_scatter_sites=60,
    combine_scatter_sample_size=2000,
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
    6) 优化输出指标表、逐站时间序列表、line / scatter / scatter_combine 图；
    7) grid_match_method 控制站点-格点匹配方法：
       - 'nearest' / 'A'：最近格点；
       - 'mean3x3' / '3x3' / 'B'：3×3 格点平均；
       - 'bilinear' / 'idw' / 'C'：双线性插值，失败时自动回退距离加权插值。
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
    timeseries_output_mode = str(timeseries_output_mode).lower()
    if timeseries_output_mode not in ("per_station", "single", "none"):
        raise ValueError("timeseries_output_mode must be 'per_station', 'single', or 'none'")
    if timeseries_output_mode == "none":
        save_site_timeseries = False
    if metric_names is None:
        metric_names = ["MAE", "RMSE", "R", "R2", "IOA", "MB", "NMB", "NME", "MFE", "MFB", "FE", "FB", "GE"]

    if validation_freq is None:
        validation_freq = "daily" if ifdaily else "hourly"
    validation_freq = str(validation_freq).lower()
    if validation_freq not in ("hourly", "daily"):
        raise ValueError("validation_freq must be 'hourly' or 'daily'")
    if validate_mda8:
        validation_freq = "daily"

    # 站点-格点匹配方法：
    # nearest / closest / A: 最近格点法；
    # mean3x3 / 3x3 / B: 以最近格点为中心的 3×3 格点平均法；
    # bilinear / idw / distance_weighted / C: 双线性插值；若网格不规则或无法形成包围格点，则回退到距离加权插值。
    grid_match_method = str(grid_match_method).strip().lower()
    grid_match_alias = {
        "a": "nearest", "nearest": "nearest", "closest": "nearest", "最近格点": "nearest",
        "b": "mean3x3", "3x3": "mean3x3", "mean3x3": "mean3x3", "3×3": "mean3x3", "3*3": "mean3x3",
        "c": "bilinear", "bilinear": "bilinear", "idw": "bilinear", "distance_weighted": "bilinear",
        "距离加权": "bilinear", "双线性": "bilinear",
    }
    if grid_match_method not in grid_match_alias:
        raise ValueError(
            "grid_match_method must be one of: "
            "'nearest'/'A', 'mean3x3'/'3x3'/'B', 'bilinear'/'idw'/'C'"
        )
    grid_match_method = grid_match_alias[grid_match_method]

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

    def _as_2d_grid(arr, name="grid"):
        """
        将 CMAQ GRIDCRO2D 的 LAT/LON 统一压缩为二维 ROW×COL。
        兼容常见维度：
        - (TSTEP, LAY, ROW, COL)
        - (LAY, ROW, COL)
        - (ROW, COL)
        """
        arr = np.asarray(arr)
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"{name} squeeze 后仍不是二维，当前 shape={arr.shape}，请检查 GRIDCRO2D 变量维度")
        return arr

    def _get_nearest_row_col(stlat, stlon, lat2d, lon2d):
        """返回最近格点的 ROW/COL，并自动兼容 LAT/LON 多余的 TSTEP/LAY 维度。"""
        lat2d = _as_2d_grid(lat2d, "LAT")
        lon2d = _as_2d_grid(lon2d, "LON")
        difflat = stlat - lat2d
        difflon = stlon - lon2d
        dist2 = difflat * difflat + difflon * difflon
        row, col = np.unravel_index(np.nanargmin(dist2), dist2.shape)
        return int(row), int(col)

    def _slice_3x3_mean(field4d, row, col):
        """
        B. 3×3 格点平均法：
        以站点最近格点为中心，取周围最多 3×3 个格点的平均。
        该方法可降低 15 km 分辨率下烟羽空间偏移 1 个格点左右造成的代表性误差。
        """
        nrow, ncol = field4d.shape[2], field4d.shape[3]
        r0, r1 = max(row - 1, 0), min(row + 2, nrow)
        c0, c1 = max(col - 1, 0), min(col + 2, ncol)
        return np.nanmean(field4d[:, 0, r0:r1, c0:c1], axis=(1, 2))

    def _idw_timeseries(field4d, stlat, stlon, lat2d, lon2d, k=4, power=2.0):
        """
        距离加权插值 IDW：
        选取距离站点最近的 k 个格点，权重为 1/d^power。
        若站点非常接近某一格点，则直接使用该格点，避免除零。
        """
        lat2d = _as_2d_grid(lat2d, "LAT")
        lon2d = _as_2d_grid(lon2d, "LON")
        dist2 = (lat2d - stlat) ** 2 + (lon2d - stlon) ** 2
        flat_idx = np.argsort(dist2.ravel())[:k]
        rows, cols = np.unravel_index(flat_idx, dist2.shape)
        d = np.sqrt(dist2[rows, cols])
        if np.nanmin(d) < 1.0e-12:
            m = int(np.nanargmin(d))
            return field4d[:, 0, rows[m], cols[m]]
        w = 1.0 / np.maximum(d, 1.0e-12) ** power
        w = w / np.nansum(w)
        vals = field4d[:, 0, rows, cols]
        return np.nansum(vals * w.reshape(1, -1), axis=1)

    def _bilinear_or_idw_timeseries(field4d, stlat, stlon, lat2d, lon2d, row, col):
        """
        C. 双线性插值 / 距离加权插值法：
        1) 优先在最近格点邻域中寻找能包围站点的 2×2 网格；
        2) 对规则经纬度网格按经纬度方向做双线性权重；
        3) 若网格弯曲、站点在边界、或找不到包围 2×2 网格，则回退到 IDW。
        """
        lat2d = _as_2d_grid(lat2d, "LAT")
        lon2d = _as_2d_grid(lon2d, "LON")
        nrow, ncol = lat2d.shape
        candidates = []
        for r in range(max(row - 1, 0), min(row + 2, nrow - 1)):
            for c in range(max(col - 1, 0), min(col + 2, ncol - 1)):
                lat_box = lat2d[r:r + 2, c:c + 2]
                lon_box = lon2d[r:r + 2, c:c + 2]
                if (
                    np.nanmin(lat_box) <= stlat <= np.nanmax(lat_box)
                    and np.nanmin(lon_box) <= stlon <= np.nanmax(lon_box)
                ):
                    # 越靠近最近格点越优先
                    candidates.append((abs(r - row) + abs(c - col), r, c))

        if candidates:
            _, r, c = sorted(candidates, key=lambda x: x[0])[0]

            # 对常见 CMAQ GRIDCRO2D 的近似规则经纬度 2D 网格：
            # x 方向用该 2×2 小格子的经度均值，y 方向用纬度均值。
            lon_w = np.nanmean(lon2d[r:r + 2, c])
            lon_e = np.nanmean(lon2d[r:r + 2, c + 1])
            lat_s = np.nanmean(lat2d[r, c:c + 2])
            lat_n = np.nanmean(lat2d[r + 1, c:c + 2])

            if abs(lon_e - lon_w) > 1.0e-12 and abs(lat_n - lat_s) > 1.0e-12:
                x = (stlon - lon_w) / (lon_e - lon_w)
                y = (stlat - lat_s) / (lat_n - lat_s)
                if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                    q11 = field4d[:, 0, r, c]
                    q21 = field4d[:, 0, r, c + 1]
                    q12 = field4d[:, 0, r + 1, c]
                    q22 = field4d[:, 0, r + 1, c + 1]
                    return (
                        q11 * (1.0 - x) * (1.0 - y)
                        + q21 * x * (1.0 - y)
                        + q12 * (1.0 - x) * y
                        + q22 * x * y
                    ), "bilinear"

        return _idw_timeseries(field4d, stlat, stlon, lat2d, lon2d), "idw"

    def _extract_sim_hour(field4d, stlat, stlon, lat2d, lon2d, method):
        """根据 grid_match_method 提取站点对应的 CMAQ 逐小时模拟序列。"""
        row, col = _get_nearest_row_col(stlat, stlon, lat2d, lon2d)

        if method == "nearest":
            series = field4d[:, 0, row, col]
            actual_method = "nearest"
        elif method == "mean3x3":
            series = _slice_3x3_mean(field4d, row, col)
            actual_method = "mean3x3"
        elif method == "bilinear":
            series, actual_method = _bilinear_or_idw_timeseries(field4d, stlat, stlon, lat2d, lon2d, row, col)
        else:
            raise ValueError(f"Unsupported grid_match_method: {method}")

        sim_hour_raw = np.asarray(series[simdata_inithour:simdata_inithour + 24 * daycount], dtype=float)
        return sim_hour_raw, row, col, actual_method

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
        """
        读取单个站点的逐小时观测值。

        内存优化：
        stream_obs_csv=True 时，air_csv_data 为 CSV 路径列表；
        每天只读取 type/hour/本站点 三列，避免一次性加载全国所有站点宽表。
        """
        vals = []

        for item in air_csv_data:
            if stream_obs_csv:
                fp = item
                if not os.path.exists(fp):
                    vals.extend([np.nan] * 24)
                    continue

                try:
                    df = pd.read_csv(
                        fp,
                        usecols=lambda c: c in {"type", "hour", station_col}
                    )
                except Exception:
                    vals.extend([np.nan] * 24)
                    continue
            else:
                df = item

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

            if stream_obs_csv:
                del df, subdf

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

        # 内存优化：默认不一次性读取所有日期的全国站点 CSV。
        # stream_obs_csv=True 时这里只保存路径，逐站处理时只读本站点列。
        if stream_obs_csv:
            air_csv_data = airStation_csv_list
        else:
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

        CMAQXLAT = _as_2d_grid(GRIDCRO2D.variables['LAT'][:], "LAT")
        CMAQXLONG = _as_2d_grid(GRIDCRO2D.variables['LON'][:], "LON")

        result_rows = []
        series_rows = []  # 仅兼容保留；默认不再累计所有站点时间序列
        combine_scatter_data = []

        ts_dir = os.path.join(out_dir, "csv_timeseries")
        if save_site_timeseries or "csv" in result_pic_types:
            os.makedirs(ts_dir, exist_ok=True)
        single_ts_path = os.path.join(ts_dir, f"site_timeseries_{suffix}.csv")
        single_ts_header_written = False
        rng_sample = np.random.default_rng(1234)

        for stname, airstation in tqdm(airStation_locations.items(), desc='处理所有有效站点...'):
            stlon = float(airstation[0])
            stlat = float(airstation[1])
            station_col = airstation[2]
            stcity = airstation[3]

            sim_hour_raw, nearlat, nearlon, actual_grid_match_method = _extract_sim_hour(
                substance, stlat, stlon, CMAQXLAT, CMAQXLONG, grid_match_method
            )
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
                "格点匹配方法": grid_match_method,
                "实际格点匹配方法": actual_grid_match_method,
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

            if "scatter" in result_pic_combine and len(combine_scatter_data) < max_combine_scatter_sites:
                obs_c = np.asarray(obs_use, dtype=float)
                sim_c = np.asarray(sim_use, dtype=float)

                if combine_scatter_sample_size is not None:
                    valid_idx = np.where(np.isfinite(obs_c) & np.isfinite(sim_c))[0]
                    if valid_idx.size > combine_scatter_sample_size:
                        keep_idx = rng_sample.choice(valid_idx, size=combine_scatter_sample_size, replace=False)
                        keep_idx.sort()
                        obs_c = obs_c[keep_idx]
                        sim_c = sim_c[keep_idx]

                combine_scatter_data.append((obs_c, sim_c, stname, stcity, metrics, unit_label))

            # 主动释放当前站点数组引用，降低长循环内存峰值。
            del obs_hour_raw, sim_hour_raw, obs_hour, sim_hour, obs_use, sim_use

        result_csv_data = pd.DataFrame(result_rows)
        result_path = os.path.join(out_dir, f"{result_csv_name or 'validation_metrics'}_{suffix}.csv")
        result_csv_data.to_csv(result_path, index=False, encoding='utf-8-sig')
        print(f"[OK] metrics csv -> {result_path}")

        if save_site_timeseries or "csv" in result_pic_types:
            if timeseries_output_mode == "single":
                print(f"[OK] site time series csv -> {single_ts_path}")
            elif timeseries_output_mode == "per_station":
                print(f"[OK] per-station time series csv -> {ts_dir}")
            else:
                print("[INFO] time series csv skipped by timeseries_output_mode='none'")

        if "scatter" in result_pic_combine and len(combine_scatter_data) > 0:
            if len(airStation_locations) > max_combine_scatter_sites:
                print(f"[WARN] scatter_combine 仅绘制前 {max_combine_scatter_sites} 个站点，避免站点过多导致内存不足。")
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
