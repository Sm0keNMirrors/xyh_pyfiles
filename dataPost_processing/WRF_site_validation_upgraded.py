"""
WRF_site_validation_upgraded.py

基于 ISD 逐小时气象站观测数据，对 WRF 模拟气象变量进行站点验证。

本版本针对原 WRF_site_validation.py 做了升级：
1. 保留原 ISD 站点筛选与 WRF 最近格点验证逻辑；
2. 参考 CMAQ_site_validation_upgraded.py 的验证指标计算方式，统一输出：
   MAE / RMSE / R / R2 / IOA / MB / NMB / NME / MFE / MFB / FE / FB / GE；
3. 新增逐站逐变量时间序列 CSV 输出；
4. 新增统一的指标 CSV 输出，也可选择另存为 Excel；
5. 优化 Obs-Sim 折线图和散点回归图绘制；
6. 优化 WRF 文件时间排序和 Times 变量读取，避免仅按文件名日期截位排序导致错序；
7. 优化 NaN、缺测值和不合理观测值处理。
"""

import datetime
import os
import re
import warnings

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression
from tqdm import tqdm


# =============================================================================
# 基础工具
# =============================================================================

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _join_dir_file(folder, filename):
    return os.path.join(folder, filename)


def _safe_station_filename(text):
    """
    将站点名转换成适合作为文件名的字符串。
    优先使用 pypinyin 的中文首字母；若环境未安装 pypinyin，则直接清理特殊字符。
    """
    if not isinstance(text, str):
        text = str(text)

    try:
        from pypinyin import pinyin, Style
        parts = []
        for ch, item in zip(text, pinyin(text, style=Style.FIRST_LETTER, strict=False)):
            if re.match(r"[\u4e00-\u9fff]", ch):
                parts.append(str(item[0]).upper())
            else:
                parts.append(ch)
        text = "".join(parts)
    except Exception:
        pass

    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "station"


def _to_float_array(values):
    return np.asarray(values, dtype=float)


def _paired_clean(sim, obs):
    """
    同时去除模拟值和观测值中的 NaN/Inf。
    返回顺序与 CMAQ 升级版保持一致：sim_clean, obs_clean, mask。
    """
    sim = _to_float_array(sim)
    obs = _to_float_array(obs)
    n = min(len(sim), len(obs))
    sim = sim[:n]
    obs = obs[:n]
    mask = np.isfinite(sim) & np.isfinite(obs)
    return sim[mask], obs[mask], mask


def _angular_difference_degree(sim, obs):
    """
    风向误差的圆周差值，范围 [-180, 180)。
    仅用于 wind_direction_circular=True 且变量为 WD 时的误差类指标。
    """
    sim = _to_float_array(sim)
    obs = _to_float_array(obs)
    return (sim - obs + 180.0) % 360.0 - 180.0


# =============================================================================
# 验证指标：参考 CMAQ_site_validation_upgraded.py 的计算与输出逻辑
# =============================================================================

def metrics_vectorized(sim, obs, metric_names=None, angular=False):
    """
    统一指标计算。

    参数：
        sim, obs:
            模拟值和观测值。
        metric_names:
            需要输出的指标名称列表。
        angular:
            True 时使用风向圆周差值计算误差类指标，适合 WD。
            R/R2 仍按原值相关性计算；NMB/NME/MFE/MFB/FE/FB/GE 的分母仍沿用 CMAQ 公式。
    """
    if metric_names is None:
        metric_names = ["MAE", "RMSE", "R", "R2", "IOA", "MB", "NMB", "NME", "MFE", "MFB", "FE", "FB", "GE"]

    sim_clean, obs_clean, _ = _paired_clean(sim, obs)
    out = {m: np.nan for m in metric_names}
    out["N"] = int(len(obs_clean))

    if len(obs_clean) == 0:
        return out

    if angular:
        diff = _angular_difference_degree(sim_clean, obs_clean)
    else:
        diff = sim_clean - obs_clean

    abs_diff = np.abs(diff)

    if "MAE" in out:
        out["MAE"] = float(np.mean(abs_diff))
    if "RMSE" in out:
        out["RMSE"] = float(np.sqrt(np.mean(diff ** 2)))
    if "MB" in out:
        out["MB"] = float(np.mean(diff))

    if len(obs_clean) >= 2 and np.std(obs_clean) > 0 and np.std(sim_clean) > 0:
        r_val = float(np.corrcoef(sim_clean, obs_clean)[0, 1])
        if "R" in out:
            out["R"] = r_val
        if "R2" in out:
            out["R2"] = float(r_val ** 2)

    obs_mean = np.mean(obs_clean)
    denom_ioa = np.sum((np.abs(sim_clean - obs_mean) + np.abs(obs_clean - obs_mean)) ** 2)
    if "IOA" in out:
        out["IOA"] = float(1.0 - np.sum(diff ** 2) / denom_ioa) if denom_ioa > 0 else np.nan

    sum_obs = np.sum(obs_clean)
    if sum_obs != 0:
        if "NMB" in out:
            out["NMB"] = float(np.sum(diff) / sum_obs * 100.0)
        if "NME" in out:
            out["NME"] = float(np.sum(abs_diff) / sum_obs * 100.0)

    den_mean = (obs_clean + sim_clean) / 2.0
    m = np.isfinite(den_mean) & (den_mean != 0)
    if np.any(m):
        if "MFE" in out:
            out["MFE"] = float(np.mean(np.abs(diff[m]) / den_mean[m]) * 100.0)
        if "MFB" in out:
            out["MFB"] = float(np.mean(diff[m] / den_mean[m]) * 100.0)

    den_sum = obs_clean + sim_clean
    m = np.isfinite(den_sum) & (den_sum != 0)
    if np.any(m):
        if "FE" in out:
            out["FE"] = float(np.mean(2.0 * np.abs(diff[m]) / den_sum[m]) * 100.0)
        if "FB" in out:
            out["FB"] = float(np.mean(2.0 * diff[m] / den_sum[m]) * 100.0)
        if "GE" in out:
            out["GE"] = float(np.mean(2.0 * np.abs(diff[m] / den_sum[m])) * 100.0)

    return out


def metrics_sklearn(sim, obs, metric_names=None, angular=False):
    """
    sklearn 模式：MAE/RMSE/R2 优先调用 sklearn，其余仍使用 vectorized 公式。
    """
    out = metrics_vectorized(sim, obs, metric_names=metric_names, angular=angular)
    sim_clean, obs_clean, _ = _paired_clean(sim, obs)

    if len(obs_clean) == 0:
        return out

    if angular:
        # sklearn 指标不适合直接处理圆周误差，因此 WD 仍使用 vectorized 的结果。
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


def calc_metrics(sim, obs, metric_method="vectorized", metric_names=None, angular=False):
    method = str(metric_method).strip().lower()
    if method in ("vectorized", "numpy", "np", "manual"):
        # 原 manual 逐项循环函数容易受 NaN 和除零影响，这里统一使用 CMAQ 升级版的向量化公式。
        return metrics_vectorized(sim, obs, metric_names=metric_names, angular=angular)
    if method in ("sklearn", "scikit-learn"):
        return metrics_sklearn(sim, obs, metric_names=metric_names, angular=angular)
    raise ValueError("metric_method must be 'vectorized', 'manual', or 'sklearn'")


# =============================================================================
# 时间工具
# =============================================================================

def generate_date_list(start_date_str, n):
    start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
    return [start_date + datetime.timedelta(days=i) for i in range(n)]


def generate_date_list_withhour(start_date_str, n):
    start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
    return [start_date + datetime.timedelta(hours=i) for i in range(n)]


def _validation_time_index(start_date, daycount, validation_freq="hourly"):
    start = pd.Timestamp(start_date)

    if validation_freq == "hourly":
        return pd.date_range(start=start, periods=daycount * 24, freq="h")

    if validation_freq == "daily":
        return pd.date_range(start=start, periods=daycount, freq="D")

    raise ValueError("validation_freq must be 'hourly' or 'daily'")


def _daily_mean(values, daycount):
    values = _to_float_array(values)
    n = min(len(values), daycount * 24)
    values = values[:n]
    n_day = n // 24
    if n_day == 0:
        return np.array([], dtype=float)
    arr2d = values[:n_day * 24].reshape(n_day, 24)
    with np.errstate(all="ignore"):
        return np.nanmean(arr2d, axis=1)


def _aggregate_series(obs_hour, sim_hour, start_date, daycount, validation_freq):
    obs_hour = _to_float_array(obs_hour)
    sim_hour = _to_float_array(sim_hour)
    n = min(len(obs_hour), len(sim_hour), daycount * 24)
    obs_hour = obs_hour[:n]
    sim_hour = sim_hour[:n]

    if validation_freq == "daily":
        obs_use = _daily_mean(obs_hour, daycount)
        sim_use = _daily_mean(sim_hour, daycount)
        n2 = min(len(obs_use), len(sim_use))
        time_use = _validation_time_index(start_date, n2, validation_freq="daily")
        return obs_use[:n2], sim_use[:n2], time_use, "Daily mean"

    time_use = _validation_time_index(start_date, daycount, validation_freq="hourly")
    n2 = min(len(obs_hour), len(sim_hour), len(time_use))
    return obs_hour[:n2], sim_hour[:n2], time_use[:n2], "Hourly"


# =============================================================================
# WRF 网格与文件读取
# =============================================================================

def getNearestPos(station_lat, station_lon, XLAT, XLONG):
    """
    返回站点最近格点的 row/col 索引。
    """
    difflat = station_lat - XLAT
    difflon = station_lon - XLONG
    dist2 = difflat * difflat + difflon * difflon
    row, col = np.unravel_index(np.nanargmin(dist2), dist2.shape)
    return int(row), int(col)


def _first_2d(arr):
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim > 2:
        return np.asarray(arr[0])
    raise ValueError(f"无法将变量压缩为二维经纬度场，shape={arr.shape}")


def _decode_wrf_times(times_var):
    """
    读取 WRF Times 变量，返回 pandas.DatetimeIndex。
    兼容 char array / bytes array。
    """
    raw = np.asarray(times_var[:])
    times = []

    if raw.ndim == 2:
        for row in raw:
            if row.dtype.kind in ("S", "U"):
                s = "".join([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in row])
            else:
                s = "".join([chr(int(x)) for x in row])
            s = s.strip().replace("_", " ")
            times.append(pd.to_datetime(s, errors="coerce"))
    elif raw.ndim == 1:
        for x in raw:
            if isinstance(x, bytes):
                s = x.decode("utf-8")
            else:
                s = str(x)
            s = s.strip().replace("_", " ")
            times.append(pd.to_datetime(s, errors="coerce"))

    out = pd.DatetimeIndex([t for t in times if pd.notna(t)])
    return out


def _parse_wrf_time_from_filename(filename):
    """
    从 wrfout_d03_YYYY-MM-DD_HH:MM:SS 形式的文件名中解析时间。
    失败时返回 pandas.NaT。
    """
    base = os.path.basename(filename)
    m = re.search(r"wrfout_[^_]+_(\d{4}-\d{2}-\d{2})[_-](\d{2}:\d{2}:\d{2})", base)
    if m:
        return pd.to_datetime(f"{m.group(1)} {m.group(2)}", errors="coerce")

    m = re.search(r"(\d{4}-\d{2}-\d{2})[_-](\d{2})", base)
    if m:
        return pd.to_datetime(f"{m.group(1)} {m.group(2)}:00:00", errors="coerce")

    return pd.NaT


def _list_wrfout_files(wrfout_files_dir, target_domain):
    files = []
    for fn in os.listdir(wrfout_files_dir):
        if fn.startswith(f"wrfout_{target_domain}"):
            files.append(os.path.join(wrfout_files_dir, fn))

    if not files:
        raise FileNotFoundError(f"在 {wrfout_files_dir} 中没有找到 wrfout_{target_domain}* 文件")

    def _sort_key(path):
        t = _parse_wrf_time_from_filename(path)
        if pd.notna(t):
            return (0, t)
        return (1, os.path.basename(path))

    return sorted(files, key=_sort_key)


def _read_wrf_file_times(path):
    try:
        with nc.Dataset(path) as ds:
            if "Times" in ds.variables:
                times = _decode_wrf_times(ds.variables["Times"])
                if len(times) > 0:
                    return times
    except Exception:
        pass

    # fallback：若文件没有 Times，就按文件名时间 + Time 维长度逐小时扩展。
    t0 = _parse_wrf_time_from_filename(path)
    if pd.isna(t0):
        return pd.DatetimeIndex([])

    try:
        with nc.Dataset(path) as ds:
            nt = len(ds.dimensions.get("Time", []))
    except Exception:
        nt = 1

    return pd.date_range(start=t0, periods=max(1, nt), freq="h")


def _calc_wrf_variables_at_grid(ds, row, col):
    """
    提取一个 WRF 文件中某个格点的 T2/RH/WS/WD 时间序列。
    """
    deg = 180.0 / np.pi

    required = ["T2", "PSFC", "Q2", "U10", "V10"]
    for var in required:
        if var not in ds.variables:
            raise KeyError(f"WRF 文件缺少变量 {var}")

    T2_K = np.asarray(ds.variables["T2"][:, row, col], dtype=float)
    PSFC = np.asarray(ds.variables["PSFC"][:, row, col], dtype=float)
    Q2 = np.asarray(ds.variables["Q2"][:, row, col], dtype=float)
    U10 = np.asarray(ds.variables["U10"][:, row, col], dtype=float)
    V10 = np.asarray(ds.variables["V10"][:, row, col], dtype=float)

    T2_C = T2_K - 273.15

    # 与原脚本保持同一计算形式，同时对 RH 做物理范围裁剪。
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        RH = 0.236 * PSFC * Q2 * np.exp((17.67 * T2_C) / (T2_K - 29.65)) ** (-1)
    RH = np.asarray(RH, dtype=float)
    RH[(RH < 0) | (RH > 150)] = np.nan
    RH = np.clip(RH, 0, 100)

    WS = np.sqrt(U10 ** 2 + V10 ** 2)
    WD = (180.0 + np.arctan2(U10, V10) * deg) % 360.0

    return {
        "T2": T2_C,
        "RH": RH,
        "WS": WS,
        "WD": WD,
    }


def _extract_wrf_timeseries_at_station(wrf_files, row, col, target_time_index):
    """
    从所有 WRF 文件中提取单站点模拟序列，并按目标验证时间对齐。
    """
    records = []
    values = {k: [] for k in ["T2", "RH", "WS", "WD"]}

    for fp in wrf_files:
        with nc.Dataset(fp, "r") as ds:
            file_times = _read_wrf_file_times(fp)
            var_data = _calc_wrf_variables_at_grid(ds, row, col)

        nt = min([len(file_times)] + [len(v) for v in var_data.values()])
        if nt == 0:
            continue

        records.extend(list(file_times[:nt]))
        for key in values:
            values[key].extend(list(var_data[key][:nt]))

    if len(records) == 0:
        # 完全读不到 Times 时，按文件顺序拼接后截取。
        out = {}
        for key in values:
            arr = _to_float_array(values[key])
            if len(arr) < len(target_time_index):
                arr = np.pad(arr, (0, len(target_time_index) - len(arr)), constant_values=np.nan)
            out[key] = arr[:len(target_time_index)]
        return out

    df = pd.DataFrame({"time": pd.DatetimeIndex(records)})
    for key in values:
        df[key] = _to_float_array(values[key])[:len(df)]

    df = df.dropna(subset=["time"]).drop_duplicates(subset=["time"]).sort_values("time")
    df = df.set_index("time")

    out = {}
    for key in ["T2", "RH", "WS", "WD"]:
        out[key] = df[key].reindex(target_time_index).to_numpy(dtype=float)

    return out


# =============================================================================
# ISD 气象站数据读取与筛选
# =============================================================================

def generate_dates(year):
    start = datetime.datetime(int(year), 1, 1, 0, 0)
    hours = 24 * (366 if pd.Timestamp(f"{year}-12-31").dayofyear == 366 else 365)
    return [(start + datetime.timedelta(hours=h)).strftime("%Y%m%d%H") for h in range(hours)]


def _normalize_station_id(stationid):
    """
    原始站点列表中的 stationid 可能被读成 int/float/str。
    原脚本逻辑是 stationid 后补一个 0，以匹配 ISD 文件名。
    """
    if pd.isna(stationid):
        return ""
    s = str(stationid).strip()
    if re.match(r"^\d+\.0$", s):
        s = s[:-2]
    if not s.endswith("0"):
        s = f"{s}0"
    return s


def _read_isd_raw_file(file_path):
    """
    读取 ISD 站点文本文件。
    原文件列顺序：
    年 月 日 时 温度 露点温度 海洋压强 风向 风速 云量 液体沉淀深度尺寸-1小时 液体沉淀深度尺寸-6小时
    """
    labels = [
        "年份", "月", "日", "时", "温度", "露点温度", "海洋压强", "风向", "风速", "云量",
        "液体沉淀深度尺寸-1小时", "液体沉淀深度尺寸-6小时"
    ]

    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=labels,
        engine="python",
        na_values=["", "NA", "nan"],
    )

    for col in labels:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["年份", "月", "日", "时"]).copy()
    df["time"] = pd.to_datetime(
        dict(
            year=df["年份"].astype(int),
            month=df["月"].astype(int),
            day=df["日"].astype(int),
            hour=df["时"].astype(int),
        ),
        errors="coerce",
    )
    df = df.dropna(subset=["time"]).drop_duplicates(subset=["time"])
    return df


def metstationFilesTo2CSV(file, outfile, year):
    """
    将气象站点数据转换为逐小时 CSV，缺测值用 -9999 填充。
    保留原函数名，便于兼容旧调用。
    """
    labels = [
        "年份", "月", "日", "时", "温度", "露点温度", "海洋压强", "风向", "风速", "云量",
        "液体沉淀深度尺寸-1小时", "液体沉淀深度尺寸-6小时"
    ]

    df_raw = _read_isd_raw_file(file)
    start = pd.Timestamp(f"{int(year)}-01-01 00:00:00")
    hours = 24 * (366 if pd.Timestamp(f"{int(year)}-12-31").dayofyear == 366 else 365)
    full_index = pd.date_range(start=start, periods=hours, freq="h")

    out = pd.DataFrame(index=full_index)
    out["年份"] = out.index.year.astype(float)
    out["月"] = out.index.month.astype(float)
    out["日"] = out.index.day.astype(float)
    out["时"] = out.index.hour.astype(float)

    df_raw = df_raw.set_index("time")
    for col in labels[4:]:
        out[col] = df_raw[col].reindex(full_index)

    out = out[labels].fillna(-9999)
    out.to_csv(outfile, encoding="utf-8-sig", index=False)
    return out


def isdsite_metdata_select(
    latmin,
    latmax,
    lonmin,
    lonmax,
    year="2020",
    metstation_files_dir="",
    metstation_infofile_dir="",
    min_file_size=1000,
):
    """
    筛选模拟区域内且有有效 ISD 数据文件的气象站。
    返回格式：
        {站点名: [lon, lat, station_id_for_file]}
    """
    metstations_infofile = pd.read_csv(metstation_infofile_dir)

    required_cols = {"Lon", "Lat", "name", "stationid"}
    missing = required_cols - set(metstations_infofile.columns)
    if missing:
        raise KeyError(f"站点信息文件缺少列: {sorted(missing)}")

    metstations_infofile["Lon"] = pd.to_numeric(metstations_infofile["Lon"], errors="coerce")
    metstations_infofile["Lat"] = pd.to_numeric(metstations_infofile["Lat"], errors="coerce")

    final_filtered_rows = metstations_infofile[
        (metstations_infofile["Lon"] > lonmin) & (metstations_infofile["Lon"] < lonmax)
        & (metstations_infofile["Lat"] > latmin) & (metstations_infofile["Lat"] < latmax)
    ]

    metstations = {}
    for row in final_filtered_rows.index.tolist():
        stid = _normalize_station_id(metstations_infofile.at[row, "stationid"])
        if not stid:
            continue
        metstations[str(metstations_infofile.at[row, "name"])] = [
            float(metstations_infofile.at[row, "Lon"]),
            float(metstations_infofile.at[row, "Lat"]),
            stid,
        ]

    valid_stations = {}
    for name, info in metstations.items():
        fp = os.path.join(metstation_files_dir, f"{info[2]}-99999-{year}")
        if os.path.exists(fp) and os.path.getsize(fp) >= min_file_size:
            valid_stations[name] = info

    return valid_stations


def _read_station_obs_for_period(
    station_file,
    target_time_index,
    save_converted_csv_path=None,
    year=None,
):
    """
    读取单个站点的验证时段观测序列，返回 T2/RH/WS/WD。
    """
    df_raw = _read_isd_raw_file(station_file).set_index("time").sort_index()
    obs = pd.DataFrame(index=target_time_index)

    # 原 ISD 文件中温度、露点温度和风速通常为 0.1 单位。
    temp_raw = pd.to_numeric(df_raw["温度"], errors="coerce").replace([-9999, 9999], np.nan)
    dew_raw = pd.to_numeric(df_raw["露点温度"], errors="coerce").replace([-9999, 9999], np.nan)
    ws_raw = pd.to_numeric(df_raw["风速"], errors="coerce").replace([-9999, 9999], np.nan)
    wd_raw = pd.to_numeric(df_raw["风向"], errors="coerce").replace([-9999, 999, 9999], np.nan)

    obs["T2"] = (temp_raw / 10.0).reindex(target_time_index)
    obs["DPT"] = (dew_raw / 10.0).reindex(target_time_index)
    obs["WS"] = (ws_raw / 10.0).reindex(target_time_index)
    obs["WD"] = wd_raw.reindex(target_time_index)

    obs.loc[(obs["T2"] < -100) | (obs["T2"] > 70), "T2"] = np.nan
    obs.loc[(obs["DPT"] < -120) | (obs["DPT"] > 70), "DPT"] = np.nan
    obs.loc[(obs["WS"] < 0) | (obs["WS"] > 100), "WS"] = np.nan
    obs.loc[(obs["WD"] < 0) | (obs["WD"] > 360), "WD"] = np.nan

    # 根据温度和露点温度计算相对湿度。
    T2 = obs["T2"].to_numpy(dtype=float)
    DPT = obs["DPT"].to_numpy(dtype=float)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        E = 6.112 * np.exp((17.67 * T2) / (T2 + 243.5))
        e = 6.112 * np.exp((17.67 * DPT) / (DPT + 243.5))
        RH = e / E * 100.0

    RH[(RH < 0) | (RH > 150)] = np.nan
    obs["RH"] = np.clip(RH, 0, 100)

    if save_converted_csv_path is not None:
        if year is None:
            year = str(target_time_index[0].year)
        metstationFilesTo2CSV(station_file, save_converted_csv_path, year)

    return {
        "T2": obs["T2"].to_numpy(dtype=float),
        "RH": obs["RH"].to_numpy(dtype=float),
        "WS": obs["WS"].to_numpy(dtype=float),
        "WD": obs["WD"].to_numpy(dtype=float),
    }


# =============================================================================
# 绘图
# =============================================================================

def _plot_line(
    time_use,
    obs_use,
    sim_use,
    stname,
    var_name,
    metrics,
    unit_label,
    validation_label,
    out_dir,
    suffix="",
    close_fig=True,
):
    pics_dir = os.path.join(out_dir, "pics_line", var_name)
    _ensure_dir(pics_dir)

    fig, ax = plt.subplots(figsize=(5.8, 2.4), dpi=220)
    ax.plot(time_use, obs_use, linewidth=1.0, label="Obs", color="#258080")
    ax.plot(time_use, sim_use, linewidth=1.0, label="Sim", color="red", alpha=0.85)

    ax.set_ylabel(unit_label, fontsize=9)
    ax.set_xlabel("Date", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.tick_params(axis="x", labelsize=7, rotation=0)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, which="major", linewidth=0.3, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=7, handlelength=2.2, borderaxespad=0.2)

    metric_txt = []
    for k in ["IOA", "MB", "RMSE", "R", "NMB"]:
        if k in metrics and np.isfinite(metrics[k]):
            u = f" {unit_label}" if k in ("MB", "RMSE", "MAE") else "%"
            if k in ("R", "IOA"):
                u = ""
            metric_txt.append(f"{k}={metrics[k]:.2f}{u}")

    if metric_txt:
        ax.text(
            0.0,
            1.08,
            "   ".join(metric_txt),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8),
        )

    ax.set_title(f"{stname} | {var_name} | {validation_label}", loc="right", fontsize=7)
    fig.tight_layout()

    fname = f"{_safe_station_filename(stname)}_{var_name}_{suffix}.png" if suffix else f"{_safe_station_filename(stname)}_{var_name}.png"
    fig.savefig(os.path.join(pics_dir, fname), bbox_inches="tight", dpi=600)

    if close_fig:
        plt.close(fig)


def _scatter_core(
    ax,
    obs_use,
    sim_use,
    stname,
    var_name,
    unit_label,
    metrics,
    scatter_density=True,
    scatter_xlim=None,
    scatter_ylim=None,
):
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
        ax.scatter(obs_clean, sim_clean, c=z, cmap="viridis", alpha=0.65, s=16)
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
        ax.plot(
            x_line,
            y_fit,
            color="red",
            linewidth=1.5,
            label=f"y={model.coef_[0]:.2f}x+{model.intercept_:.2f}",
        )

    ax.set_xlabel(f"Obs ({unit_label})")
    ax.set_ylabel(f"Sim ({unit_label})")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    stats = []
    for k in ["R", "RMSE", "NMB"]:
        if k in metrics and np.isfinite(metrics[k]):
            unit = "%" if k == "NMB" else ""
            stats.append(f"{k}={metrics[k]:.2f}{unit}")

    ax.set_title(f"{stname} | {var_name}\n" + "  ".join(stats), fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc="lower right")


def _plot_scatter(
    obs_use,
    sim_use,
    stname,
    var_name,
    metrics,
    unit_label,
    out_dir,
    suffix="",
    scatter_density=True,
    scatter_xlim=None,
    scatter_ylim=None,
    close_fig=True,
):
    pics_dir = os.path.join(out_dir, "pics_scatter", var_name)
    _ensure_dir(pics_dir)

    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=220)
    _scatter_core(
        ax,
        obs_use,
        sim_use,
        stname,
        var_name,
        unit_label,
        metrics,
        scatter_density=scatter_density,
        scatter_xlim=scatter_xlim,
        scatter_ylim=scatter_ylim,
    )
    fig.tight_layout()

    fname = f"{_safe_station_filename(stname)}_{var_name}_{suffix}.png" if suffix else f"{_safe_station_filename(stname)}_{var_name}.png"
    fig.savefig(os.path.join(pics_dir, fname), bbox_inches="tight", dpi=500)

    if close_fig:
        plt.close(fig)


def calculate_figRowCol_size(x):
    import math
    n = int(math.ceil(math.sqrt(x)))
    m = int(math.ceil(x / n))
    return n, m


# =============================================================================
# 主函数
# =============================================================================

def WRF_site_validation(
    start_date="YYYY-MM-DD",
    daycount=5,
    wrfout_files_dir="",
    target_domain="",
    metstation_files_dir="",
    metstation_infofile_dir="",
    result_pic_types=None,
    out_dir="",
    suffix="",
    result_csv_name="validationPara",
    validation_freq="hourly",
    metric_method="vectorized",
    metric_names=None,
    metstation_selected=None,
    manual_stations=None,
    add_default_manual_stations=True,
    save_converted_met_csv=False,
    save_site_timeseries=True,
    timeseries_output_mode="per_station",
    result_pic_combine=None,
    max_combine_scatter_sites=60,
    scatter_xlim=None,
    scatter_ylim=None,
    scatter_density=True,
    wind_direction_circular=True,
    save_metrics_xlsx=False,
    close_fig=True,
):
    """
    使用全球 ISD 逐小时气象站数据验证 WRF 模拟气象场。

    参数：
        start_date:
            验证开始日期，格式 'YYYY-MM-DD'。
        daycount:
            验证天数。
        wrfout_files_dir:
            wrfout 文件目录。
        target_domain:
            WRF domain，例如 'd01'、'd02'、'd03'。
        metstation_files_dir:
            ISD 气象站逐小时数据文件目录。
        metstation_infofile_dir:
            ISD 气象站位置 CSV 文件路径，需包含 Lon/Lat/name/stationid。
        result_pic_types:
            需要输出的单站图类型，可包含 'line'、'scatter'、'csv'。
        out_dir:
            输出目录。
        suffix:
            输出文件后缀。
        result_csv_name:
            指标 CSV 文件名前缀。
        validation_freq:
            'hourly' 或 'daily'。
        metric_method:
            'vectorized' / 'manual' / 'sklearn'。
        metric_names:
            输出指标列表，默认与 CMAQ 升级版一致。
        metstation_selected:
            指定验证站点名列表；None 或 [] 表示验证所有筛选站点。
        manual_stations:
            手动加入站点字典，格式 {'成都': [104.02, 30.67, '562940']}。
        add_default_manual_stations:
            True 时保留原脚本中手动加入的成都、南充站。
        save_converted_met_csv:
            True 时保存转换后的全年站点 CSV。
        save_site_timeseries:
            True 时保存逐站逐变量 Obs/Sim 时间序列 CSV。
        timeseries_output_mode:
            'per_station'：每个站点一个 CSV；
            'single'：所有站点合并为一个 CSV；
            'none'：不保存时间序列。
        result_pic_combine:
            可包含 'scatter'，输出每个变量的多站合并散点图。
        wind_direction_circular:
            True 时 WD 的误差类指标使用圆周差值。
    """
    if result_pic_types is None:
        result_pic_types = []
    if result_pic_combine is None:
        result_pic_combine = []
    if metstation_selected is None:
        metstation_selected = []
    if metric_names is None:
        metric_names = ["MAE", "RMSE", "R", "R2", "IOA", "MB", "NMB", "NME", "MFE", "MFB", "FE", "FB", "GE"]

    validation_freq = str(validation_freq).lower()
    if validation_freq not in ("hourly", "daily"):
        raise ValueError("validation_freq must be 'hourly' or 'daily'")

    timeseries_output_mode = str(timeseries_output_mode).lower()
    if timeseries_output_mode not in ("per_station", "single", "none"):
        raise ValueError("timeseries_output_mode must be 'per_station', 'single', or 'none'")
    if timeseries_output_mode == "none":
        save_site_timeseries = False

    if not start_date or start_date == "YYYY-MM-DD":
        raise ValueError("请提供 start_date='YYYY-MM-DD'")
    if not wrfout_files_dir:
        raise ValueError("wrfout_files_dir 不能为空")
    if not target_domain:
        raise ValueError("target_domain 不能为空，例如 'd03'")
    if not metstation_files_dir:
        raise ValueError("metstation_files_dir 不能为空")
    if not metstation_infofile_dir:
        raise ValueError("metstation_infofile_dir 不能为空")
    if not out_dir:
        raise ValueError("out_dir 不能为空")

    _ensure_dir(out_dir)
    matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    input_year = str(pd.Timestamp(start_date).year)
    target_time_hourly = _validation_time_index(start_date, daycount, validation_freq="hourly")

    wrf_files = _list_wrfout_files(wrfout_files_dir, target_domain)

    with nc.Dataset(wrf_files[0], "r") as wrfout_init:
        XLONG = _first_2d(wrfout_init.variables["XLONG"][:])
        XLAT = _first_2d(wrfout_init.variables["XLAT"][:])

    lonmin, latmax, lonmax, latmin = float(np.nanmin(XLONG)), float(np.nanmax(XLAT)), float(np.nanmax(XLONG)), float(np.nanmin(XLAT))

    metstations = isdsite_metdata_select(
        latmin,
        latmax,
        lonmin,
        lonmax,
        year=input_year,
        metstation_files_dir=metstation_files_dir,
        metstation_infofile_dir=metstation_infofile_dir,
    )

    if add_default_manual_stations:
        # 保留原脚本的手动站点；只有存在数据文件时才加入。
        default_manual = {
            "成都": [104.02, 30.67, "562940"],
            "南充": [106.08, 30.78, "574110"],
        }
        if manual_stations is None:
            manual_stations = default_manual
        else:
            merged = dict(default_manual)
            merged.update(manual_stations)
            manual_stations = merged

    if manual_stations:
        for name, info in manual_stations.items():
            lon, lat, stid = info[0], info[1], str(info[2])
            fp = os.path.join(metstation_files_dir, f"{stid}-99999-{input_year}")
            if os.path.exists(fp):
                metstations[name] = [float(lon), float(lat), stid]
            else:
                warnings.warn(f"手动站点 {name} 数据文件不存在，已跳过：{fp}")

    if metstation_selected:
        metstations = {k: v for k, v in metstations.items() if k in set(metstation_selected)}

    print(f"[INFO] selected WRF files: {len(wrf_files)}")
    print(f"[INFO] selected met stations: {len(metstations)}")
    if len(metstations) == 0:
        print("[WARN] 没有筛选到可验证气象站，请检查 WRF 范围、站点信息文件或 ISD 数据目录。")

    units = {
        "T2": "°C",
        "RH": "%",
        "WS": "m s$^{-1}$",
        "WD": "degree",
    }
    var_full_names = {
        "T2": "2m temperature",
        "RH": "Relative humidity",
        "WS": "10m wind speed",
        "WD": "10m wind direction",
    }

    result_rows = []
    combine_scatter_data = {v: [] for v in ["T2", "RH", "WS", "WD"]}

    ts_dir = os.path.join(out_dir, "csv_timeseries")
    if save_site_timeseries or "csv" in result_pic_types:
        _ensure_dir(ts_dir)
    single_ts_path = os.path.join(ts_dir, f"site_timeseries_{suffix}.csv" if suffix else "site_timeseries.csv")
    single_ts_header_written = False

    converted_dir = os.path.join(out_dir, "converted_met_csv")
    if save_converted_met_csv:
        _ensure_dir(converted_dir)

    for stname, metinfo in tqdm(metstations.items(), desc="处理所有有效气象站..."):
        station_lon = float(metinfo[0])
        station_lat = float(metinfo[1])
        station_id = str(metinfo[2])

        station_file = os.path.join(metstation_files_dir, f"{station_id}-99999-{input_year}")
        if not os.path.exists(station_file):
            print(f"[WARN] obs file not found, skip {stname}: {station_file}")
            continue

        row, col = getNearestPos(station_lat, station_lon, XLAT, XLONG)

        sim_hour = _extract_wrf_timeseries_at_station(wrf_files, row, col, target_time_hourly)

        save_converted_csv_path = None
        if save_converted_met_csv:
            save_converted_csv_path = os.path.join(converted_dir, f"{_safe_station_filename(stname)}-{input_year}.csv")

        obs_hour = _read_station_obs_for_period(
            station_file,
            target_time_hourly,
            save_converted_csv_path=save_converted_csv_path,
            year=input_year,
        )

        site_ts_rows = []

        for var_name in ["T2", "RH", "WS", "WD"]:
            obs_use, sim_use, time_use, validation_label = _aggregate_series(
                obs_hour[var_name],
                sim_hour[var_name],
                start_date,
                daycount,
                validation_freq,
            )

            if len(obs_use) == 0 or np.all(~np.isfinite(obs_use)):
                continue

            angular = bool(wind_direction_circular and var_name == "WD")
            metrics = calc_metrics(
                sim_use,
                obs_use,
                metric_method=metric_method,
                metric_names=metric_names,
                angular=angular,
            )

            row_out = {
                "变量": var_name,
                "变量名称": var_full_names.get(var_name, var_name),
                "站点": stname,
                "站点经度": station_lon,
                "站点纬度": station_lat,
                "站点编码": station_id,
                "最近网格ROW": row,
                "最近网格COL": col,
                "最近网格经度": float(XLONG[row, col]),
                "最近网格纬度": float(XLAT[row, col]),
                "验证频率": validation_label,
                "单位": units[var_name],
                "指标算法": metric_method,
                "风向是否使用圆周差值": bool(angular),
            }
            for name in ["N"] + metric_names:
                if name in metrics:
                    row_out[name] = metrics[name]
            result_rows.append(row_out)

            if save_site_timeseries or "csv" in result_pic_types:
                for t, obs_v, sim_v in zip(time_use, obs_use, sim_use):
                    site_ts_rows.append({
                        "time": pd.Timestamp(t),
                        "站点": stname,
                        "站点编码": station_id,
                        "站点经度": station_lon,
                        "站点纬度": station_lat,
                        "变量": var_name,
                        "单位": units[var_name],
                        "Obs": obs_v,
                        "Sim": sim_v,
                    })

            if "line" in result_pic_types:
                _plot_line(
                    time_use,
                    obs_use,
                    sim_use,
                    stname,
                    var_name,
                    metrics,
                    units[var_name],
                    validation_label,
                    out_dir,
                    suffix=suffix,
                    close_fig=close_fig,
                )

            if "scatter" in result_pic_types:
                _plot_scatter(
                    obs_use,
                    sim_use,
                    stname,
                    var_name,
                    metrics,
                    units[var_name],
                    out_dir,
                    suffix=suffix,
                    scatter_density=scatter_density,
                    scatter_xlim=scatter_xlim,
                    scatter_ylim=scatter_ylim,
                    close_fig=close_fig,
                )

            if "scatter" in result_pic_combine and len(combine_scatter_data[var_name]) < max_combine_scatter_sites:
                combine_scatter_data[var_name].append((obs_use, sim_use, stname, var_name, units[var_name], metrics))

        if (save_site_timeseries or "csv" in result_pic_types) and site_ts_rows:
            site_ts_df = pd.DataFrame(site_ts_rows)
            if timeseries_output_mode == "per_station":
                ts_path = os.path.join(
                    ts_dir,
                    f"{_safe_station_filename(stname)}_{suffix}.csv" if suffix else f"{_safe_station_filename(stname)}.csv",
                )
                site_ts_df.to_csv(ts_path, index=False, encoding="utf-8-sig")
            elif timeseries_output_mode == "single":
                site_ts_df.to_csv(
                    single_ts_path,
                    mode="a",
                    header=not single_ts_header_written,
                    index=False,
                    encoding="utf-8-sig",
                )
                single_ts_header_written = True

    result_csv_data = pd.DataFrame(result_rows)

    result_path = os.path.join(
        out_dir,
        f"{result_csv_name}_{suffix}.csv" if suffix else f"{result_csv_name}.csv",
    )
    result_csv_data.to_csv(result_path, index=False, encoding="utf-8-sig")
    print(f"[OK] metrics csv -> {result_path}")

    if save_metrics_xlsx:
        xlsx_path = os.path.join(
            out_dir,
            f"{result_csv_name}_{suffix}.xlsx" if suffix else f"{result_csv_name}.xlsx",
        )
        with pd.ExcelWriter(xlsx_path) as writer:
            result_csv_data.to_excel(writer, sheet_name="all", index=False)
            for var_name in ["T2", "RH", "WS", "WD"]:
                sub = result_csv_data[result_csv_data["变量"] == var_name]
                if not sub.empty:
                    sub.to_excel(writer, sheet_name=var_name, index=False)
        print(f"[OK] metrics xlsx -> {xlsx_path}")

    if save_site_timeseries or "csv" in result_pic_types:
        if timeseries_output_mode == "single":
            print(f"[OK] site time series csv -> {single_ts_path}")
        elif timeseries_output_mode == "per_station":
            print(f"[OK] per-station time series csv -> {ts_dir}")
        else:
            print("[INFO] time series csv skipped by timeseries_output_mode='none'")

    if "scatter" in result_pic_combine:
        for var_name, data_list in combine_scatter_data.items():
            if not data_list:
                continue
            nn, mm = calculate_figRowCol_size(len(data_list))
            fig_c, axs_c = plt.subplots(nn, mm, figsize=(4.2 * mm, 4.2 * nn), dpi=220, squeeze=False)

            for i, (obs_use, sim_use, stname, var_nm, unit_label, metrics) in enumerate(data_list):
                rr, cc = i // mm, i % mm
                _scatter_core(
                    axs_c[rr, cc],
                    obs_use,
                    sim_use,
                    stname,
                    var_nm,
                    unit_label,
                    metrics,
                    scatter_density=scatter_density,
                    scatter_xlim=scatter_xlim,
                    scatter_ylim=scatter_ylim,
                )

            for j in range(len(data_list), nn * mm):
                rr, cc = j // mm, j % mm
                fig_c.delaxes(axs_c[rr, cc])

            fig_c.supxlabel(f"Observed ({units[var_name]})", fontsize=14)
            fig_c.supylabel(f"Simulated ({units[var_name]})", fontsize=14)
            fig_c.tight_layout()

            combine_path = os.path.join(
                out_dir,
                f"scatter_combine_{var_name}_{suffix}.png" if suffix else f"scatter_combine_{var_name}.png",
            )
            fig_c.savefig(combine_path, bbox_inches="tight", dpi=500)

            if close_fig:
                plt.close(fig_c)

            print(f"[OK] combined scatter {var_name} -> {combine_path}")

    return result_csv_data


if __name__ == "__main__":
    # 示例：
    WRF_site_validation(
        start_date="2022-08-01",
        daycount=31,
        wrfout_files_dir=r"E:\CMAQdata_chengdu202208\WRFd03_2\\",
        target_domain="d03",
        metstation_files_dir=r"E:\气象站数据\china_isdsite_metdata\\",
        metstation_infofile_dir=r"E:\气象站数据\全国气象站位置信息\站点列表_原始数据.csv",
        result_pic_types=["line", "scatter", "csv"],
        result_pic_combine=[],
        out_dir=r"E:\Emission_update\\test_wrf_validation\\",
        result_csv_name="validationPara_WRF",
        suffix="CD202208",
        validation_freq="hourly",
        metric_method="vectorized",
        save_site_timeseries=True,
        timeseries_output_mode="per_station",
        save_converted_met_csv=False,
        save_metrics_xlsx=False,
    )
