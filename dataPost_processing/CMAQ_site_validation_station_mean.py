
"""
CMAQ_site_validation_station_mean.py

新的总体站点平均验证方式：
不再逐站分别输出验证图和验证指标，而是先提取所有有效站点的 Obs/Sim 时间序列，
在每个时刻对所有有效站点求平均和标准差，再用总体平均 Obs/Sim 序列计算验证参数，
并绘制带标准差误差棒的总体观测-模拟时间序列图。
"""

import datetime
import os
import re
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import netCDF4 as nc
from tqdm import tqdm


def generate_date_list(start_date_str, n):
    start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
    return [start_date + datetime.timedelta(days=i) for i in range(n)]


def getAirStationsFromLatLon(uplat, downlat, leftlon, rightlon, station_info_csv=""):
    """
    从站点信息 csv 文件中，返回位于指定矩形经纬度范围内的站点信息。

    返回格式：
    {
        "站点名": [经度, 纬度, 站点编码, 城市]
    }
    """
    airStation_infofile = pd.read_csv(station_info_csv)

    airStation_infofile["经度"] = pd.to_numeric(airStation_infofile["经度"], errors="coerce")
    airStation_infofile["纬度"] = pd.to_numeric(airStation_infofile["纬度"], errors="coerce")

    filtered_rows = airStation_infofile[
        airStation_infofile["经度"].notna()
        & airStation_infofile["纬度"].notna()
    ]

    final_filtered_rows = filtered_rows[
        (filtered_rows["经度"] > leftlon)
        & (filtered_rows["经度"] < rightlon)
        & (filtered_rows["纬度"] > downlat)
        & (filtered_rows["纬度"] < uplat)
    ]

    airStations = {}
    for row in final_filtered_rows.index.tolist():
        airStations.update({
            airStation_infofile.at[row, "监测点名称"]: [
                float(airStation_infofile.at[row, "经度"]),
                float(airStation_infofile.at[row, "纬度"]),
                airStation_infofile.at[row, "监测点编码"],
                airStation_infofile.at[row, "城市"],
            ]
        })

    return airStations


def _norm_key(name):
    key = re.sub(r"[^A-Za-z0-9.]+", "", str(name).upper())
    aliases = {
        "OZONE": "O3",
        "PM25": "PM25",
        "PM2.5": "PM2.5",
        "PM2_5": "PM2.5",
    }
    return aliases.get(key, key)


def _as_2d_grid(arr, name="grid"):
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"{name} squeeze 后仍不是二维，当前 shape={arr.shape}")
    return arr


def _get_nearest_row_col(stlat, stlon, lat2d, lon2d):
    lat2d = _as_2d_grid(lat2d, "LAT")
    lon2d = _as_2d_grid(lon2d, "LON")

    dist2 = (lat2d - stlat) ** 2 + (lon2d - stlon) ** 2
    row, col = np.unravel_index(np.nanargmin(dist2), dist2.shape)
    return int(row), int(col)


def _slice_3x3_mean(field4d, row, col):
    nrow, ncol = field4d.shape[2], field4d.shape[3]
    r0, r1 = max(row - 1, 0), min(row + 2, nrow)
    c0, c1 = max(col - 1, 0), min(col + 2, ncol)
    return np.nanmean(field4d[:, 0, r0:r1, c0:c1], axis=(1, 2))


def _idw_timeseries(field4d, stlat, stlon, lat2d, lon2d, k=4, power=2.0):
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
                candidates.append((abs(r - row) + abs(c - col), r, c))

    if candidates:
        _, r, c = sorted(candidates, key=lambda x: x[0])[0]

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


def _extract_sim_hour(
    field4d,
    stlat,
    stlon,
    lat2d,
    lon2d,
    simdata_inithour,
    daycount,
    grid_match_method="nearest",
):
    row, col = _get_nearest_row_col(stlat, stlon, lat2d, lon2d)

    if grid_match_method == "nearest":
        series = field4d[:, 0, row, col]
        actual_method = "nearest"
    elif grid_match_method == "mean3x3":
        series = _slice_3x3_mean(field4d, row, col)
        actual_method = "mean3x3"
    elif grid_match_method == "bilinear":
        series, actual_method = _bilinear_or_idw_timeseries(
            field4d, stlat, stlon, lat2d, lon2d, row, col
        )
    else:
        raise ValueError("grid_match_method must be 'nearest', 'mean3x3', or 'bilinear'")

    sim_hour = np.asarray(
        series[simdata_inithour:simdata_inithour + 24 * daycount],
        dtype=float,
    )

    return sim_hour, row, col, actual_method


def _read_obs_for_station(
    air_csv_data,
    station_col,
    target_substance_obs,
    stream_obs_csv=True,
):
    """
    读取一个站点的逐小时观测序列。

    stream_obs_csv=True 时，air_csv_data 是逐日 CSV 路径列表；
    每次只读 type/hour/本站点 三列，适合大文件。
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

        subdf = df[df["type"] == target_substance_obs] if "type" in df.columns else pd.DataFrame()

        if subdf.empty or station_col not in subdf.columns:
            vals.extend([np.nan] * 24)
            continue

        subdf = subdf[["hour", station_col]].copy()
        subdf["hour"] = pd.to_numeric(subdf["hour"], errors="coerce")
        subdf[station_col] = pd.to_numeric(subdf[station_col], errors="coerce")
        subdf = subdf.dropna(subset=["hour"])
        subdf["hour"] = subdf["hour"].astype(int)

        hour_map = dict(zip(subdf["hour"], subdf[station_col]))
        vals.extend([hour_map.get(h, np.nan) for h in range(24)])

        if stream_obs_csv:
            del df, subdf

    return np.asarray(vals, dtype=float)


def _daily_mean(values, daycount):
    values = np.asarray(values, dtype=float)[:daycount * 24]
    n_day = len(values) // 24
    if n_day == 0:
        return np.array([], dtype=float)

    arr2d = values[:n_day * 24].reshape(n_day, 24)

    with np.errstate(all="ignore"):
        return np.nanmean(arr2d, axis=1)


def _daily_mda8(values, daycount, mda8_min_hours=6):
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


def _aggregate_time_scale(
    obs_hour,
    sim_hour,
    start_date,
    daycount,
    validation_freq="hourly",
    validate_mda8=False,
    mda8_min_hours=6,
):
    obs_hour = np.asarray(obs_hour, dtype=float)
    sim_hour = np.asarray(sim_hour, dtype=float)

    t_hour = pd.date_range(
        start=pd.Timestamp(start_date),
        periods=daycount * 24,
        freq="h",
    )

    n = min(len(obs_hour), len(sim_hour), len(t_hour))
    obs_hour = obs_hour[:n]
    sim_hour = sim_hour[:n]
    t_hour = t_hour[:n]

    if validate_mda8:
        obs_use = _daily_mda8(obs_hour, daycount, mda8_min_hours=mda8_min_hours)
        sim_use = _daily_mda8(sim_hour, daycount, mda8_min_hours=mda8_min_hours)
        time_use = pd.date_range(start=pd.Timestamp(start_date), periods=len(obs_use), freq="D")
        label = "Daily MDA8"
    elif validation_freq == "daily":
        obs_use = _daily_mean(obs_hour, daycount)
        sim_use = _daily_mean(sim_hour, daycount)
        time_use = pd.date_range(start=pd.Timestamp(start_date), periods=len(obs_use), freq="D")
        label = "Daily mean"
    else:
        obs_use = obs_hour
        sim_use = sim_hour
        time_use = t_hour
        label = "Hourly"

    n2 = min(len(obs_use), len(sim_use), len(time_use))
    return (
        np.asarray(obs_use[:n2], dtype=float),
        np.asarray(sim_use[:n2], dtype=float),
        pd.DatetimeIndex(time_use[:n2]),
        label,
    )


def _paired_clean(sim, obs):
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)
    n = min(len(sim), len(obs))
    sim = sim[:n]
    obs = obs[:n]

    mask = np.isfinite(sim) & np.isfinite(obs)
    return sim[mask], obs[mask], mask


def _calc_metrics_vectorized(sim, obs, metric_names=None):
    if metric_names is None:
        metric_names = [
            "MAE", "RMSE", "R", "R2", "IOA", "MB",
            "NMB", "NME", "MFE", "MFB", "FE", "FB", "GE"
        ]

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

    if len(obs) >= 2 and np.std(obs) > 0 and np.std(sim) > 0:
        r = float(np.corrcoef(sim, obs)[0, 1])
        if "R" in out:
            out["R"] = r
        if "R2" in out:
            out["R2"] = r ** 2

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


def _calc_metrics(sim, obs, metric_method="vectorized", metric_names=None):
    """
    对总体平均后的 Obs/Sim 序列计算验证参数。
    默认使用 numpy 向量化计算。
    metric_method='sklearn' 时，MAE/RMSE/R2 尝试使用 sklearn，其余仍用 numpy。
    """
    method = str(metric_method).lower()

    out = _calc_metrics_vectorized(sim, obs, metric_names=metric_names)

    if method in ("vectorized", "numpy", "np", "manual"):
        return out

    if method in ("sklearn", "scikit-learn"):
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

    raise ValueError("metric_method must be 'vectorized', 'manual', or 'sklearn'")


def _nanstd_1d(arr, axis=0):
    """
    计算标准差。
    有效样本数 < 2 的位置返回 0，避免全 NaN 或单样本导致 warning。
    """
    arr = np.asarray(arr, dtype=float)
    n = np.sum(np.isfinite(arr), axis=axis)

    with np.errstate(all="ignore"):
        std = np.nanstd(arr, axis=axis, ddof=1)

    std = np.where(n >= 2, std, 0.0)
    return std


def _convert_units(
    obs_arr,
    sim_arr,
    target_substances,
    target_substance_obs,
    Molar_mass=0,
    convert_to_ppb=False,
    obs_unit="ug/m3",
    sim_unit="ppm",
    temperature_K=298.15,
    pressure_atm=1.0,
):
    obs = np.asarray(obs_arr, dtype=float)
    sim = np.asarray(sim_arr, dtype=float)

    gas_mw_map = {
        "O3": 48.0,
        "NO2": 46.0055,
        "NO": 30.0061,
        "NOX": 46.0055,
        "SO2": 64.066,
        "CO": 28.0101,
        "NH3": 17.031,
        "HCHO": 30.026,
        "FORM": 30.026,
    }
    particle_keys = {"PM25", "PM2.5", "PM10", "PM1", "PM25_TOT", "PM2_5"}

    key_obs = _norm_key(target_substance_obs)
    keys_mod = {_norm_key(x) for x in target_substances}
    is_particle = key_obs in particle_keys or bool(keys_mod & particle_keys)

    if Molar_mass not in (None, 0):
        mw = float(Molar_mass)
    else:
        mw = gas_mw_map.get(key_obs or _norm_key(target_substances[0]), None)

    can_convert = (mw is not None) and (not is_particle)

    def ppb_factor_from_ugm3(mw_value):
        # ppb = ug/m3 * R*T/P/MW
        r_atm = 0.082057
        molar_volume_l = r_atm * float(temperature_K) / float(pressure_atm)
        return molar_volume_l / float(mw_value)

    if can_convert:
        sim_unit_lower = str(sim_unit).lower()
        obs_unit_lower = str(obs_unit).lower()

        if sim_unit_lower in ("ppm", "ppmv"):
            sim_ugm3 = sim * mw / 22.4 * 1000.0
        elif sim_unit_lower in ("ppb", "ppbv"):
            sim_ugm3 = sim / ppb_factor_from_ugm3(mw)
        else:
            sim_ugm3 = sim

        if obs_unit_lower in ("ppb", "ppbv"):
            obs_ugm3 = obs / ppb_factor_from_ugm3(mw)
        else:
            obs_ugm3 = obs

        if convert_to_ppb:
            fac = ppb_factor_from_ugm3(mw)
            return obs_ugm3 * fac, sim_ugm3 * fac, "ppb"

        return obs_ugm3, sim_ugm3, "ug/m3"

    return obs, sim, "ug/m3"


def _apply_grid_match_alias(grid_match_method):
    grid_match_method = str(grid_match_method).strip().lower()

    alias = {
        "a": "nearest",
        "nearest": "nearest",
        "closest": "nearest",
        "最近格点": "nearest",

        "b": "mean3x3",
        "3x3": "mean3x3",
        "3×3": "mean3x3",
        "3*3": "mean3x3",
        "mean3x3": "mean3x3",

        "c": "bilinear",
        "bilinear": "bilinear",
        "idw": "bilinear",
        "distance_weighted": "bilinear",
        "双线性": "bilinear",
        "距离加权": "bilinear",
    }

    if grid_match_method not in alias:
        raise ValueError(
            "grid_match_method must be one of: "
            "'nearest'/'A', 'mean3x3'/'3x3'/'B', 'bilinear'/'idw'/'C'"
        )

    return alias[grid_match_method]


def _plot_mean_validation(
    time_use,
    obs_mean,
    obs_std,
    sim_mean,
    sim_std,
    metrics,
    out_png,
    unit_label,
    validation_label,
    title="Station-mean CMAQ validation",
    errorbar_every=1,
    obs_color="#258080",
    sim_color="red",
    close_fig=True,
):
    fig, ax = plt.subplots(figsize=(8.5, 3.2), dpi=260)

    time_use = pd.DatetimeIndex(time_use)
    obs_mean = np.asarray(obs_mean, dtype=float)
    obs_std = np.asarray(obs_std, dtype=float)
    sim_mean = np.asarray(sim_mean, dtype=float)
    sim_std = np.asarray(sim_std, dtype=float)

    if errorbar_every is None or int(errorbar_every) < 1:
        errorbar_every = 1
    errorbar_every = int(errorbar_every)

    idx = np.arange(len(time_use))
    err_idx = idx[::errorbar_every]

    ax.errorbar(
        time_use[err_idx],
        obs_mean[err_idx],
        yerr=obs_std[err_idx],
        fmt="o",
        markersize=3.2,
        linewidth=0.0,
        elinewidth=0.8,
        capsize=2,
        color=obs_color,
        ecolor=obs_color,
        alpha=0.9,
        label="Obs mean ± SD",
        zorder=3,
    )

    # 模拟值主线
    ax.plot(
        time_use,
        sim_mean,
        "-",
        linewidth=1.2,
        color=sim_color,
        alpha=0.9,
        label="Sim mean",
        zorder=2,
    )

    # 模拟值每个点的标准差误差棒
    ax.errorbar(
        time_use[err_idx],
        sim_mean[err_idx],
        yerr=sim_std[err_idx],
        fmt="none",
        elinewidth=0.8,
        capsize=2,
        ecolor=sim_color,
        alpha=0.65,
        label="Sim ± SD",
        zorder=1,
    )

    ax.set_ylabel(unit_label)
    ax.set_xlabel("Date")
    ax.set_title(f"{title} | {validation_label}", fontsize=10)
    ax.grid(True, linewidth=0.35, alpha=0.35)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if len(time_use) <= 40:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    else:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    ax.tick_params(axis="x", labelsize=8, rotation=0)
    ax.tick_params(axis="y", labelsize=8)

    metric_txt = []
    for k in ["IOA", "R", "RMSE", "MB", "NMB"]:
        if k in metrics and np.isfinite(metrics[k]):
            if k in ("RMSE", "MB", "MAE"):
                metric_txt.append(f"{k}={metrics[k]:.2f} {unit_label}")
            else:
                metric_txt.append(f"{k}={metrics[k]:.2f}")

    if metric_txt:
        ax.text(
            0.0,
            1.04,
            "   ".join(metric_txt),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.5,
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor="white",
                edgecolor="none",
                alpha=0.85,
            ),
        )

    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=3)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=600)

    if close_fig:
        plt.close(fig)

    return fig


def CMAQ_site_validation_station_mean(
    start_date="YYYY-MM-DD",
    daycount=5,
    simdata_inithour=16,
    GRIDCRO2D_file_dir="",
    Combine_file_dir="",
    target_substances=None,
    target_substance_obs="",
    Molar_mass=0,
    ifdaily=False,
    validation_freq=None,
    convert_to_ppb=False,
    validate_mda8=False,
    mda8_min_hours=6,
    grid_match_method="nearest",
    metric_method="vectorized",
    metric_names=None,
    obs_unit="ug/m3",
    sim_unit="ppm",
    temperature_K=298.15,
    pressure_atm=1.0,
    airstation_files_dir="",
    airstation_infofile_dir="",
    airstation_selected=None,
    out_dir="",
    result_csv_name="station_mean_validation",
    suffix="",
    stream_obs_csv=True,
    paired_mean=True,
    min_valid_station_count=1,
    save_station_series=False,
    save_mean_timeseries=True,
    plot_result=True,
    errorbar_every=1,
    close_fig=True,
):
    """
    所有站点总体平均验证函数。

    :param start_date: 验证开始日期，格式 "YYYY-MM-DD"。
    :param daycount: 验证天数。
    :param simdata_inithour: CMAQ 模拟值截取起始小时。
    :param GRIDCRO2D_file_dir: CMAQ GRIDCRO2D 文件路径。
    :param Combine_file_dir: CMAQ combine 后浓度 nc 文件路径。
    :param target_substances: CMAQ 中参与验证的变量名列表，多个变量会相加。
    :param target_substance_obs: 观测 CSV 中对应污染物名称。
    :param Molar_mass: 气态污染物分子量，0 表示自动识别或不转换。
    :param ifdaily: 旧参数兼容，validation_freq 未指定时 True 表示日均。
    :param validation_freq: "hourly" 或 "daily"。
    :param convert_to_ppb: 是否将可转换气态物质转为 ppb。
    :param validate_mda8: 是否验证每日 MDA8，通常用于 O3。
    :param mda8_min_hours: MDA8 每个 8h 窗口最少有效小时数。
    :param grid_match_method: 站点-格点匹配方法，"nearest"、"mean3x3" 或 "bilinear"。
    :param metric_method: 验证指标算法，"vectorized"、"manual" 或 "sklearn"。
    :param metric_names: 需要输出的验证指标名称列表，None 表示默认全部。
    :param obs_unit: 观测原始单位，常用 "ug/m3" 或 "ppb"。
    :param sim_unit: 模拟原始单位，气态 CMAQ 常用 "ppm"。
    :param temperature_K: 单位转换温度。
    :param pressure_atm: 单位转换气压。
    :param airstation_files_dir: 逐日站点观测 CSV 文件夹。
    :param airstation_infofile_dir: 站点信息 CSV 文件路径。
    :param airstation_selected: 指定站点名称列表，空列表表示使用模拟范围内全部站点。
    :param out_dir: 输出文件夹。
    :param result_csv_name: 输出文件名前缀。
    :param suffix: 输出文件名后缀。
    :param stream_obs_csv: 是否逐站逐日只读取必要列，适合大文件。
    :param paired_mean: True 表示同一时刻只用 Obs/Sim 都有效的站点参与平均。
    :param min_valid_station_count: 同一时刻至少多少个有效站点才保留总体平均值。
    :param save_station_series: 是否额外保存每个站点聚合后的 Obs/Sim 序列。
    :param save_mean_timeseries: 是否保存总体平均时间序列表。
    :param plot_result: 是否绘制总体平均验证曲线。
    :param errorbar_every: 误差棒绘制间隔，逐小时时可设为 3、6 或 24 减少遮挡。
    :param close_fig: 保存图后是否关闭 figure。
    :return: metrics_df, mean_ts_df。
    """

    if target_substances is None:
        target_substances = []
    if airstation_selected is None:
        airstation_selected = []
    if metric_names is None:
        metric_names = [
            "MAE", "RMSE", "R", "R2", "IOA", "MB",
            "NMB", "NME", "MFE", "MFB", "FE", "FB", "GE"
        ]

    if validation_freq is None:
        validation_freq = "daily" if ifdaily else "hourly"
    validation_freq = str(validation_freq).lower()

    if validation_freq not in ("hourly", "daily"):
        raise ValueError("validation_freq must be 'hourly' or 'daily'")
    if validate_mda8:
        validation_freq = "daily"

    grid_match_method = _apply_grid_match_alias(grid_match_method)

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

    matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    GRIDCRO2D = nc.Dataset(GRIDCRO2D_file_dir)
    CMAQoutf = nc.Dataset(Combine_file_dir, "r")

    try:
        var_lon = np.array(GRIDCRO2D.variables["LON"][:][0])
        var_lat = np.array(GRIDCRO2D.variables["LAT"][:][0])

        lonmin = float(np.nanmin(var_lon))
        lonmax = float(np.nanmax(var_lon))
        latmin = float(np.nanmin(var_lat))
        latmax = float(np.nanmax(var_lat))

        # 与原函数一致：先按 CMAQ 经纬度范围筛选站点，再按 airstation_selected 二次筛选。
        airStation_locations_all = getAirStationsFromLatLon(
            latmax,
            latmin,
            lonmin,
            lonmax,
            airstation_infofile_dir,
        )

        if airstation_selected != []:
            airStation_locations = {
                name: info
                for name, info in airStation_locations_all.items()
                if name in airstation_selected
            }
        else:
            airStation_locations = airStation_locations_all

        print(f"[INFO] selected stations: {len(airStation_locations)}")
        if len(airStation_locations) == 0:
            raise ValueError("没有筛选到站点，请检查 GRIDCRO2D 范围、站点信息文件或 airstation_selected。")

        validation_dates = generate_date_list(start_date, daycount)
        airStation_csv_list = []
        for date in validation_dates:
            date2 = str(date).split(" ")[0]
            filedate = "".join(date2.split("-"))
            airStation_csv_list.append(
                os.path.join(airstation_files_dir, f"china_sites_{filedate}.csv")
            )

        if stream_obs_csv:
            air_csv_data = airStation_csv_list
        else:
            air_csv_data = []
            for fp in airStation_csv_list:
                if os.path.exists(fp):
                    air_csv_data.append(pd.read_csv(fp))
                else:
                    warnings.warn(f"obs csv not found: {fp}")
                    air_csv_data.append(pd.DataFrame(columns=["type", "hour"]))

        substance = None
        for sub in target_substances:
            if sub not in CMAQoutf.variables:
                raise KeyError(f"{sub} 不在 Combine_file_dir 的变量中")
            arr = np.asarray(CMAQoutf.variables[sub][:], dtype=np.float32)
            substance = arr if substance is None else substance + arr

        CMAQXLAT = _as_2d_grid(GRIDCRO2D.variables["LAT"][:], "LAT")
        CMAQXLONG = _as_2d_grid(GRIDCRO2D.variables["LON"][:], "LON")

        obs_site_series = []
        sim_site_series = []
        station_meta_rows = []
        station_series_rows = []
        time_use_ref = None
        unit_label_ref = None
        validation_label_ref = None

        for stname, airstation in tqdm(airStation_locations.items(), desc="处理所有有效站点..."):
            stlon = float(airstation[0])
            stlat = float(airstation[1])
            station_col = airstation[2]
            stcity = airstation[3]

            sim_hour_raw, row, col, actual_grid_method = _extract_sim_hour(
                substance,
                stlat,
                stlon,
                CMAQXLAT,
                CMAQXLONG,
                simdata_inithour=simdata_inithour,
                daycount=daycount,
                grid_match_method=grid_match_method,
            )

            obs_hour_raw = _read_obs_for_station(
                air_csv_data,
                station_col=station_col,
                target_substance_obs=target_substance_obs,
                stream_obs_csv=stream_obs_csv,
            )

            if np.all(~np.isfinite(obs_hour_raw)):
                continue

            obs_hour, sim_hour, unit_label = _convert_units(
                obs_hour_raw,
                sim_hour_raw,
                target_substances=target_substances,
                target_substance_obs=target_substance_obs,
                Molar_mass=Molar_mass,
                convert_to_ppb=convert_to_ppb,
                obs_unit=obs_unit,
                sim_unit=sim_unit,
                temperature_K=temperature_K,
                pressure_atm=pressure_atm,
            )

            obs_use, sim_use, time_use, validation_label = _aggregate_time_scale(
                obs_hour,
                sim_hour,
                start_date=start_date,
                daycount=daycount,
                validation_freq=validation_freq,
                validate_mda8=validate_mda8,
                mda8_min_hours=mda8_min_hours,
            )

            if len(obs_use) == 0 or np.all(~np.isfinite(obs_use)):
                continue

            if paired_mean:
                pair_mask = np.isfinite(obs_use) & np.isfinite(sim_use)
                obs_for_mean = np.where(pair_mask, obs_use, np.nan)
                sim_for_mean = np.where(pair_mask, sim_use, np.nan)
            else:
                obs_for_mean = obs_use
                sim_for_mean = sim_use

            if time_use_ref is None:
                time_use_ref = time_use
                unit_label_ref = unit_label
                validation_label_ref = validation_label
            else:
                if len(time_use) != len(time_use_ref):
                    warnings.warn(f"{stname} 时间长度与首个有效站点不一致，跳过。")
                    continue

            obs_site_series.append(obs_for_mean)
            sim_site_series.append(sim_for_mean)

            station_meta_rows.append({
                "站点": stname,
                "城市": stcity,
                "站点编码": station_col,
                "站点经度": stlon,
                "站点纬度": stlat,
                "最近网格ROW": row,
                "最近网格COL": col,
                "格点匹配方法": grid_match_method,
                "实际格点匹配方法": actual_grid_method,
                "有效配对数": int(np.sum(np.isfinite(obs_for_mean) & np.isfinite(sim_for_mean))),
            })

            if save_station_series:
                for t, o, s in zip(time_use, obs_for_mean, sim_for_mean):
                    station_series_rows.append({
                        "datetime": t,
                        "站点": stname,
                        "城市": stcity,
                        "站点编码": station_col,
                        "obs": o,
                        "sim": s,
                        "unit": unit_label,
                        "validation_freq": validation_label,
                    })

            del obs_hour_raw, sim_hour_raw, obs_hour, sim_hour, obs_use, sim_use

        if len(obs_site_series) == 0:
            raise ValueError("没有获得任何有效站点序列，无法进行总体平均验证。")

        obs_arr = np.vstack(obs_site_series)
        sim_arr = np.vstack(sim_site_series)

        valid_pair = np.isfinite(obs_arr) & np.isfinite(sim_arr)
        valid_station_count = np.sum(valid_pair, axis=0)

        obs_arr_mean_input = np.where(valid_pair, obs_arr, np.nan) if paired_mean else obs_arr
        sim_arr_mean_input = np.where(valid_pair, sim_arr, np.nan) if paired_mean else sim_arr

        with np.errstate(all="ignore"):
            obs_mean = np.nanmean(obs_arr_mean_input, axis=0)
            sim_mean = np.nanmean(sim_arr_mean_input, axis=0)

        obs_std = _nanstd_1d(obs_arr_mean_input, axis=0)
        sim_std = _nanstd_1d(sim_arr_mean_input, axis=0)

        low_count_mask = valid_station_count < int(min_valid_station_count)
        obs_mean[low_count_mask] = np.nan
        sim_mean[low_count_mask] = np.nan
        obs_std[low_count_mask] = np.nan
        sim_std[low_count_mask] = np.nan

        metrics = _calc_metrics(
            sim_mean,
            obs_mean,
            metric_method=metric_method,
            metric_names=metric_names,
        )

        metrics_row = {
            "验证对象": "所有站点总体平均序列",
            "开始日期": start_date,
            "验证天数": daycount,
            "有效站点数": len(obs_site_series),
            "总筛选站点数": len(airStation_locations),
            "验证频率": validation_label_ref,
            "单位": unit_label_ref,
            "目标模拟物质": "+".join(target_substances),
            "目标观测物质": target_substance_obs,
            "指标算法": metric_method,
            "格点匹配方法": grid_match_method,
            "是否ppb转换": bool(convert_to_ppb),
            "是否MDA8": bool(validate_mda8),
            "paired_mean": bool(paired_mean),
            "min_valid_station_count": int(min_valid_station_count),
        }

        for name in ["N"] + metric_names:
            if name in metrics:
                metrics_row[name] = metrics[name]

        metrics_df = pd.DataFrame([metrics_row])

        mean_ts_df = pd.DataFrame({
            "datetime": time_use_ref,
            "obs_mean": obs_mean,
            "obs_std": obs_std,
            "sim_mean": sim_mean,
            "sim_std": sim_std,
            "valid_station_count": valid_station_count,
            "unit": unit_label_ref,
            "validation_freq": validation_label_ref,
        })

        suffix_part = f"_{suffix}" if suffix else ""

        metrics_path = os.path.join(out_dir, f"{result_csv_name}{suffix_part}_metrics.csv")
        mean_ts_path = os.path.join(out_dir, f"{result_csv_name}{suffix_part}_mean_timeseries.csv")
        station_meta_path = os.path.join(out_dir, f"{result_csv_name}{suffix_part}_stations_used.csv")
        station_series_path = os.path.join(out_dir, f"{result_csv_name}{suffix_part}_station_series.csv")
        png_path = os.path.join(out_dir, f"{result_csv_name}{suffix_part}_mean_curve.png")

        metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(station_meta_rows).to_csv(station_meta_path, index=False, encoding="utf-8-sig")

        if save_mean_timeseries:
            mean_ts_df.to_csv(mean_ts_path, index=False, encoding="utf-8-sig")

        if save_station_series:
            pd.DataFrame(station_series_rows).to_csv(station_series_path, index=False, encoding="utf-8-sig")

        if plot_result:
            _plot_mean_validation(
                time_use=time_use_ref,
                obs_mean=obs_mean,
                obs_std=obs_std,
                sim_mean=sim_mean,
                sim_std=sim_std,
                metrics=metrics,
                out_png=png_path,
                unit_label=unit_label_ref,
                validation_label=validation_label_ref,
                title=f"{target_substance_obs} station-mean validation",
                errorbar_every=errorbar_every,
                close_fig=close_fig,
            )
            print(f"[OK] mean validation figure -> {png_path}")

        print(f"[OK] metrics csv -> {metrics_path}")
        if save_mean_timeseries:
            print(f"[OK] mean time series csv -> {mean_ts_path}")
        print(f"[OK] station list csv -> {station_meta_path}")

        return metrics_df, mean_ts_df

    finally:
        try:
            GRIDCRO2D.close()
        except Exception:
            pass
        try:
            CMAQoutf.close()
        except Exception:
            pass


if __name__ == "__main__":
    # 示例：
    # metrics_df, mean_ts_df = CMAQ_site_validation_station_mean(
    #     start_date="2020-08-01",
    #     daycount=30,
    #     simdata_inithour=16,
    #     GRIDCRO2D_file_dir=r"E:\project\GRIDCRO2D_d03.nc",
    #     Combine_file_dir=r"E:\project\COMBINE_ACONC.nc",
    #     target_substances=["O3"],
    #     target_substance_obs="O3",
    #     Molar_mass=48,
    #     validation_freq="daily",
    #     validate_mda8=True,
    #     convert_to_ppb=True,
    #     metric_method="vectorized",
    #     grid_match_method="nearest",
    #     airstation_files_dir=r"E:\全国空气质量\全国站点小时浓度csv_files\\",
    #     airstation_infofile_dir=r"E:\全国空气质量\_站点列表\站点列表-2022.02.13起.csv",
    #     out_dir=r"E:\project\validation_station_mean\\",
    #     result_csv_name="O3_MDA8_station_mean",
    #     suffix="case01",
    #     errorbar_every=1,
    # )
    pass
