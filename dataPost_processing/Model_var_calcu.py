from pathlib import Path
from typing import Optional, Union, Sequence, Tuple
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


def infer_main_dims(
    da: xr.DataArray,
    layer_candidates: Sequence[str] = (
        "LAY", "lay", "Layer", "layer",
        "bottom_top", "level", "lev", "z", "Z"
    ),
) -> dict:
    """
    根据变量维度自动推断：
    - 时间维：默认第一个维度
    - 行列维：优先 ROW / COL，否则默认最后两个维度
    - 垂直维：LAY / bottom_top / level 等
    """
    if len(da.dims) < 3:
        raise ValueError(f"变量维度过少，当前维度为：{da.dims}")

    time_dim = da.dims[0]

    if "ROW" in da.dims:
        row_dim = "ROW"
    else:
        row_dim = da.dims[-2]

    if "COL" in da.dims:
        col_dim = "COL"
    else:
        col_dim = da.dims[-1]

    layer_dim = None
    for dim in layer_candidates:
        if dim in da.dims and dim not in (time_dim, row_dim, col_dim):
            layer_dim = dim
            break

    return {
        "time_dim": time_dim,
        "layer_dim": layer_dim,
        "row_dim": row_dim,
        "col_dim": col_dim,
    }


def get_lat_lon(
    ds: xr.Dataset,
    lat_candidates: Sequence[str] = ("LAT", "lat", "latitude", "XLAT"),
    lon_candidates: Sequence[str] = ("LONG", "LON", "lon", "longitude", "XLONG"),
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    从 Dataset 中获取二维经纬度坐标。
    默认优先识别 LAT / LONG。
    """
    lat_name = None
    lon_name = None

    for name in lat_candidates:
        if name in ds:
            lat_name = name
            break

    for name in lon_candidates:
        if name in ds:
            lon_name = name
            break

    if lat_name is None:
        raise KeyError(f"未找到纬度变量，候选名称为：{lat_candidates}")

    if lon_name is None:
        raise KeyError(f"未找到经度变量，候选名称为：{lon_candidates}")

    return ds[lat_name], ds[lon_name]


def subset_time(
    da: xr.DataArray,
    time_dim: str,
    start_time=None,
    end_time=None,
) -> xr.DataArray:
    """
    按时间维切片。

    start_time / end_time 可以是：
    - datetime 字符串，例如 "2024-01-01 00:00:00"
    - pandas Timestamp
    - 如果时间维是整数，也可以传整数
    """
    if start_time is None and end_time is None:
        return da

    return da.sel({time_dim: slice(start_time, end_time)})


def reduce_time(
    da: xr.DataArray,
    time_dim: str,
    method: str = "mean",
    skipna: bool = True,
) -> xr.DataArray:
    """
    对时间维做统计。
    """
    method = method.lower()

    if method == "mean":
        return da.mean(dim=time_dim, skipna=skipna)

    if method == "max":
        return da.max(dim=time_dim, skipna=skipna)

    if method == "min":
        return da.min(dim=time_dim, skipna=skipna)

    if method == "sum":
        return da.sum(dim=time_dim, skipna=skipna)

    if method == "median":
        return da.median(dim=time_dim, skipna=skipna)

    if method == "std":
        return da.std(dim=time_dim, skipna=skipna)

    raise ValueError(
        "method 目前支持：mean, max, min, sum, median, std"
    )


def calc_time_statistics(
    ds: xr.Dataset,
    var_name: str,
    start_time=None,
    end_time=None,
    methods: Sequence[str] = ("mean", "max", "max8h"),
    max8h_window: int = 8,
    output_path: Optional[Union[str, Path]] = None,
    zlib: bool = True,
    complevel: int = 4,
) -> xr.Dataset:
    """
    对 combine 后的连续变量做时间统计。

    支持 methods:
    - "mean"       : 时间平均
    - "max"        : 时间最大
    - "min"        : 时间最小
    - "sum"        : 时间累计
    - "median"     : 时间中位数
    - "std"        : 时间标准差
    - "max8h"      : 时间切片内最大 8 步滑动平均值
    - "daily_max8h": 每日最大 8 步滑动平均值
    - "p95"        : 95 分位数
    - "p99"        : 99 分位数

    注意：
    max8h_window=8 默认表示 8 个时间步。
    如果你的数据是小时数据，则表示 8 小时。
    如果是 3 小时数据，需要自己调整 window。
    """
    if var_name not in ds:
        raise KeyError(f"Dataset 中不存在变量：{var_name}")

    da = ds[var_name]
    dims = infer_main_dims(da)
    time_dim = dims["time_dim"]

    da = subset_time(
        da=da,
        time_dim=time_dim,
        start_time=start_time,
        end_time=end_time,
    )

    result_vars = {}

    for method in methods:
        m = method.lower()

        if m in ("mean", "max", "min", "sum", "median", "std"):
            out = reduce_time(
                da=da,
                time_dim=time_dim,
                method=m,
            )
            result_vars[f"{var_name}_{m}"] = out

        elif m == "max8h":
            rolling_mean = da.rolling(
                {time_dim: max8h_window},
                min_periods=max8h_window,
            ).mean()

            out = rolling_mean.max(dim=time_dim, skipna=True)
            result_vars[f"{var_name}_max{max8h_window}step_mean"] = out

        elif m == "daily_max8h":
            if not np.issubdtype(ds[time_dim].dtype, np.datetime64):
                raise TypeError(
                    f"{time_dim} 不是 datetime64 类型，无法计算 daily_max8h。"
                    "请确认读取 nc 时 decode_times=True，或时间坐标已正确解码。"
                )

            rolling_mean = da.rolling(
                {time_dim: max8h_window},
                min_periods=max8h_window,
            ).mean()

            out = rolling_mean.groupby(f"{time_dim}.date").max(
                dim=time_dim,
                skipna=True,
            )

            if "date" in out.dims:
                out = out.rename({"date": "DATE"})

            result_vars[f"{var_name}_daily_max{max8h_window}step_mean"] = out

        elif m.startswith("p"):
            q = float(m[1:]) / 100.0

            out = da.quantile(
                q=q,
                dim=time_dim,
                skipna=True,
            )

            if "quantile" in out.dims:
                out = out.squeeze("quantile", drop=True)

            if "quantile" in out.coords:
                out = out.drop_vars("quantile")

            result_vars[f"{var_name}_{m}"] = out

        else:
            raise ValueError(
                f"不支持的统计方法：{method}。"
                "支持 mean, max, min, sum, median, std, max8h, daily_max8h, p95, p99 等。"
            )

    out_ds = xr.Dataset(result_vars)

    out_ds.attrs["source_variable"] = var_name
    out_ds.attrs["time_stat_start"] = str(start_time)
    out_ds.attrs["time_stat_end"] = str(end_time)
    out_ds.attrs["time_dimension"] = time_dim
    out_ds.attrs["methods"] = ", ".join(methods)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        encoding = {
            name: {
                "zlib": zlib,
                "complevel": complevel,
            }
            for name in out_ds.data_vars
        }

        out_ds.to_netcdf(
            output_path,
            engine="netcdf4",
            format="NETCDF4",
            encoding=encoding,
        )

    return out_ds

"""
ds = xr.open_dataset("./O3_combined_with_latlong.nc")

stat_ds = calc_time_statistics(
    ds=ds,
    var_name="O3",
    start_time="2024-01-01 00:00:00",
    end_time="2024-01-31 23:00:00",
    methods=("mean", "max", "max8h", "p95"),
    output_path="./O3_time_statistics.nc",
)

"""

def find_nearest_grid_point(
    ds: xr.Dataset,
    target_lat: float,
    target_lon: float,
    lat_name: str = "LAT",
    lon_name: str = "LONG",
) -> dict:
    """
    根据目标经纬度，找到最近的网格点 ROW / COL。
    默认使用 LAT / LONG。
    如果你的文件里是 LAT / LON，也会自动兼容。
    """
    lat, lon = get_lat_lon(
        ds,
        lat_candidates=(lat_name, "LAT", "lat", "latitude", "XLAT"),
        lon_candidates=(lon_name, "LONG", "LON", "lon", "longitude", "XLONG"),
    )

    lat_values = lat.values
    lon_values = lon.values

    if lat_values.ndim != 2 or lon_values.ndim != 2:
        raise ValueError(
            f"经纬度变量必须是二维，当前 LAT 维度为 {lat.dims}, LONG 维度为 {lon.dims}"
        )

    # 近似距离，考虑经度方向随纬度缩放
    cos_lat = np.cos(np.deg2rad(target_lat))
    dist2 = (lat_values - target_lat) ** 2 + ((lon_values - target_lon) * cos_lat) ** 2

    flat_index = np.nanargmin(dist2)
    row_index, col_index = np.unravel_index(flat_index, dist2.shape)

    return {
        "target_lat": float(target_lat),
        "target_lon": float(target_lon),
        "row_index": int(row_index),
        "col_index": int(col_index),
        "nearest_lat": float(lat_values[row_index, col_index]),
        "nearest_lon": float(lon_values[row_index, col_index]),
        "distance_degree": float(np.sqrt(dist2[row_index, col_index])),
    }


def extract_nearest_point_timeseries(
    ds: xr.Dataset,
    var_name: str,
    target_lat: float,
    target_lon: float,
    start_time=None,
    end_time=None,
    layer_index: Optional[int] = None,
    as_dataframe: bool = True,
) -> Union[pd.DataFrame, xr.DataArray]:
    """
    提取目标经纬度最近网格点的时间序列。

    如果变量有垂直层：
    - layer_index=None : 返回所有垂直层的时间序列
    - layer_index=0    : 返回第 0 层时间序列
    - layer_index=1    : 返回第 1 层时间序列

    返回：
    - as_dataframe=True  : pandas.DataFrame
    - as_dataframe=False : xarray.DataArray
    """
    if var_name not in ds:
        raise KeyError(f"Dataset 中不存在变量：{var_name}")

    da = ds[var_name]
    dims = infer_main_dims(da)

    time_dim = dims["time_dim"]
    layer_dim = dims["layer_dim"]
    row_dim = dims["row_dim"]
    col_dim = dims["col_dim"]

    point_info = find_nearest_grid_point(
        ds=ds,
        target_lat=target_lat,
        target_lon=target_lon,
    )

    da = subset_time(
        da=da,
        time_dim=time_dim,
        start_time=start_time,
        end_time=end_time,
    )

    point_da = da.isel(
        {
            row_dim: point_info["row_index"],
            col_dim: point_info["col_index"],
        }
    )

    if layer_index is not None:
        if layer_dim is None:
            raise ValueError(f"变量 {var_name} 没有识别到垂直层维度，无法选择 layer_index。")

        point_da = point_da.isel({layer_dim: layer_index})

    point_da.attrs.update(point_info)

    if not as_dataframe:
        return point_da

    df = point_da.to_dataframe(name=var_name).reset_index()

    for key, value in point_info.items():
        df[key] = value

    if layer_index is not None:
        df["layer_index"] = layer_index

    return df

"""
# 提取某个经纬度最近点，所有垂直层时间序列
df_all_layer = extract_nearest_point_timeseries(
    ds=ds,
    var_name="O3",
    target_lat=39.90,
    target_lon=116.40,
    start_time="2024-01-01",
    end_time="2024-01-03",
    layer_index=None,
)

# 只提取第 0 层
df_layer0 = extract_nearest_point_timeseries(
    ds=ds,
    var_name="O3",
    target_lat=39.90,
    target_lon=116.40,
    layer_index=0,
)

df_layer0.to_csv("./O3_nearest_point_layer0_timeseries.csv", index=False)


"""



def calc_vertical_profile_area_mean(
    ds: xr.Dataset,
    var_name: str,
    start_time=None,
    end_time=None,
    time_method: str = "mean",
    row_slice: Optional[slice] = None,
    col_slice: Optional[slice] = None,
    lat_range: Optional[Tuple[float, float]] = None,
    lon_range: Optional[Tuple[float, float]] = None,
) -> xr.DataArray:
    """
    计算区域平均垂直廓线。

    支持两种区域选择方式：

    1. row / col 索引范围：
        row_slice=slice(10, 30)
        col_slice=slice(20, 50)

    2. 经纬度范围：
        lat_range=(39.0, 41.0)
        lon_range=(115.0, 117.0)

    输出结果通常为：
        profile(LAY)
    """
    if var_name not in ds:
        raise KeyError(f"Dataset 中不存在变量：{var_name}")

    da = ds[var_name]
    dims = infer_main_dims(da)

    time_dim = dims["time_dim"]
    layer_dim = dims["layer_dim"]
    row_dim = dims["row_dim"]
    col_dim = dims["col_dim"]

    if layer_dim is None:
        raise ValueError(f"变量 {var_name} 没有识别到垂直层维度，无法绘制垂直廓线。")

    da = subset_time(
        da=da,
        time_dim=time_dim,
        start_time=start_time,
        end_time=end_time,
    )

    if row_slice is not None:
        da = da.isel({row_dim: row_slice})

    if col_slice is not None:
        da = da.isel({col_dim: col_slice})

    if lat_range is not None or lon_range is not None:
        lat, lon = get_lat_lon(ds)

        if row_slice is not None:
            lat = lat.isel({row_dim: row_slice})

        if col_slice is not None:
            lat = lat.isel({col_dim: col_slice})

        if row_slice is not None:
            lon = lon.isel({row_dim: row_slice})

        if col_slice is not None:
            lon = lon.isel({col_dim: col_slice})

        mask = xr.ones_like(lat, dtype=bool)

        if lat_range is not None:
            lat_min, lat_max = lat_range
            mask = mask & (lat >= lat_min) & (lat <= lat_max)

        if lon_range is not None:
            lon_min, lon_max = lon_range
            mask = mask & (lon >= lon_min) & (lon <= lon_max)

        da = da.where(mask)

    da_time = reduce_time(
        da=da,
        time_dim=time_dim,
        method=time_method,
    )

    profile = da_time.mean(
        dim=[row_dim, col_dim],
        skipna=True,
    )

    profile.name = f"{var_name}_vertical_profile"
    profile.attrs["source_variable"] = var_name
    profile.attrs["time_method"] = time_method
    profile.attrs["profile_type"] = "area_mean"

    return profile

"""
profile = calc_vertical_profile_area_mean(
    ds=ds,
    var_name="O3",
    start_time="2024-01-01",
    end_time="2024-01-31",
    time_method="mean",
    row_slice=slice(20, 60),
    col_slice=slice(30, 80),
)

profile = calc_vertical_profile_area_mean(
    ds=ds,
    var_name="O3",
    start_time="2024-01-01",
    end_time="2024-01-31",
    time_method="mean",
    lat_range=(39.0, 41.0),
    lon_range=(115.0, 117.0),
)

"""


def plot_vertical_profile(
    profile: xr.DataArray,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    invert_yaxis: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
):
    """
    绘制一维垂直廓线。
    """
    if profile.ndim != 1:
        raise ValueError(f"profile 必须是一维 DataArray，当前维度为：{profile.dims}")

    layer_dim = profile.dims[0]

    if layer_dim in profile.coords:
        y = profile[layer_dim].values
    else:
        y = np.arange(profile.sizes[layer_dim])

    x = profile.values

    fig, ax = plt.subplots(figsize=(5, 6))

    ax.plot(x, y, marker="o")

    ax.set_xlabel(xlabel or profile.name or "Value")
    ax.set_ylabel(ylabel or layer_dim)
    ax.set_title(title or "Vertical profile")
    ax.grid(True, linestyle="--", alpha=0.4)

    if invert_yaxis:
        ax.invert_yaxis()

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, ax

"""
fig, ax = plot_vertical_profile(
    profile,
    xlabel="O3",
    ylabel="Layer",
    title="O3 vertical profile",
    save_path="./O3_vertical_profile.png",
)

"""
def haversine_distance_km(
    lat1,
    lon1,
    lat2,
    lon2,
) -> float:
    """
    计算两点间球面距离，单位 km。
    """
    r = 6371.0

    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return float(r * c)


def calc_vertical_transect(
    ds: xr.Dataset,
    var_name: str,
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    n_points: int = 50,
    start_time=None,
    end_time=None,
    time_method: str = "mean",
    drop_duplicate_grid: bool = True,
) -> xr.DataArray:
    """
    沿经纬度线段提取垂直剖面。

    输出结果通常为：
        transect(LAY, POINT)

    其中 POINT 是沿线段的采样点。
    """
    if var_name not in ds:
        raise KeyError(f"Dataset 中不存在变量：{var_name}")

    da = ds[var_name]
    dims = infer_main_dims(da)

    time_dim = dims["time_dim"]
    layer_dim = dims["layer_dim"]
    row_dim = dims["row_dim"]
    col_dim = dims["col_dim"]

    if layer_dim is None:
        raise ValueError(f"变量 {var_name} 没有识别到垂直层维度，无法计算垂直剖面。")

    da = subset_time(
        da=da,
        time_dim=time_dim,
        start_time=start_time,
        end_time=end_time,
    )

    da_time = reduce_time(
        da=da,
        time_dim=time_dim,
        method=time_method,
    )

    sample_lats = np.linspace(start_lat, end_lat, n_points)
    sample_lons = np.linspace(start_lon, end_lon, n_points)

    profiles = []
    used_grid = set()

    out_lats = []
    out_lons = []
    out_rows = []
    out_cols = []
    out_distances = []

    cumulative_distance = 0.0
    last_lat = None
    last_lon = None

    for lat_i, lon_i in zip(sample_lats, sample_lons):
        point_info = find_nearest_grid_point(
            ds=ds,
            target_lat=float(lat_i),
            target_lon=float(lon_i),
        )

        row_index = point_info["row_index"]
        col_index = point_info["col_index"]

        grid_key = (row_index, col_index)

        if drop_duplicate_grid and grid_key in used_grid:
            continue

        used_grid.add(grid_key)

        nearest_lat = point_info["nearest_lat"]
        nearest_lon = point_info["nearest_lon"]

        if last_lat is None:
            cumulative_distance = 0.0
        else:
            cumulative_distance += haversine_distance_km(
                last_lat,
                last_lon,
                nearest_lat,
                nearest_lon,
            )

        last_lat = nearest_lat
        last_lon = nearest_lon

        prof_i = da_time.isel(
            {
                row_dim: row_index,
                col_dim: col_index,
            }
        )

        profiles.append(prof_i)

        out_lats.append(nearest_lat)
        out_lons.append(nearest_lon)
        out_rows.append(row_index)
        out_cols.append(col_index)
        out_distances.append(cumulative_distance)

    if not profiles:
        raise ValueError("没有成功提取任何剖面点，请检查经纬度范围。")

    transect = xr.concat(
        profiles,
        dim="POINT",
    )

    transect = transect.transpose(layer_dim, "POINT")

    transect = transect.assign_coords(
        {
            "POINT": np.arange(transect.sizes["POINT"]),
            "transect_lat": ("POINT", np.array(out_lats)),
            "transect_lon": ("POINT", np.array(out_lons)),
            "grid_row": ("POINT", np.array(out_rows)),
            "grid_col": ("POINT", np.array(out_cols)),
            "distance_km": ("POINT", np.array(out_distances)),
        }
    )

    transect.name = f"{var_name}_vertical_transect"
    transect.attrs["source_variable"] = var_name
    transect.attrs["time_method"] = time_method
    transect.attrs["start_lat"] = start_lat
    transect.attrs["start_lon"] = start_lon
    transect.attrs["end_lat"] = end_lat
    transect.attrs["end_lon"] = end_lon

    return transect

"""
transect = calc_vertical_transect(
    ds=ds,
    var_name="O3",
    start_lat=39.0,
    start_lon=115.0,
    end_lat=41.0,
    end_lon=117.5,
    n_points=80,
    start_time="2024-01-01",
    end_time="2024-01-31",
    time_method="mean",
)

"""

def plot_vertical_transect(
    transect: xr.DataArray,
    xlabel: str = "Distance along transect (km)",
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    cmap: str = "viridis",
    levels: int = 20,
    invert_yaxis: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
):
    """
    绘制沿线段的垂直剖面图。
    """
    if "POINT" not in transect.dims:
        raise ValueError(f"transect 中必须包含 POINT 维度，当前维度为：{transect.dims}")

    layer_dim = [dim for dim in transect.dims if dim != "POINT"][0]

    transect_plot = transect.transpose(layer_dim, "POINT")

    if "distance_km" in transect_plot.coords:
        x = transect_plot["distance_km"].values
    else:
        x = np.arange(transect_plot.sizes["POINT"])

    if layer_dim in transect_plot.coords:
        y = transect_plot[layer_dim].values
    else:
        y = np.arange(transect_plot.sizes[layer_dim])

    z = transect_plot.values

    fig, ax = plt.subplots(figsize=(9, 5))

    cf = ax.contourf(
        x,
        y,
        z,
        levels=levels,
        cmap=cmap,
    )

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(transect.name or "Value")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or layer_dim)
    ax.set_title(title or "Vertical transect")

    if invert_yaxis:
        ax.invert_yaxis()

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, ax

"""
fig, ax = plot_vertical_transect(
    transect,
    title="O3 vertical transect",
    save_path="./O3_vertical_transect.png",
)

"""



"""
import xarray as xr

ds = xr.open_dataset("./O3_combined_with_latlong.nc")

# 1. 时间统计
stat_ds = calc_time_statistics(
    ds=ds,
    var_name="O3",
    start_time="2024-01-01",
    end_time="2024-01-31",
    methods=("mean", "max", "max8h", "p95"),
    output_path="./O3_stats.nc",
)

# 2. 最近点时间序列
df_ts = extract_nearest_point_timeseries(
    ds=ds,
    var_name="O3",
    target_lat=39.90,
    target_lon=116.40,
    layer_index=0,
)

df_ts.to_csv("./O3_point_timeseries.csv", index=False)

# 3. 区域平均垂直廓线
profile = calc_vertical_profile_area_mean(
    ds=ds,
    var_name="O3",
    start_time="2024-01-01",
    end_time="2024-01-31",
    time_method="mean",
    lat_range=(39.0, 41.0),
    lon_range=(115.0, 117.0),
)

plot_vertical_profile(
    profile,
    xlabel="O3",
    ylabel="LAY",
    title="O3 vertical profile",
    save_path="./O3_vertical_profile.png",
)

# 4. 经纬度线段垂直剖面
transect = calc_vertical_transect(
    ds=ds,
    var_name="O3",
    start_lat=39.0,
    start_lon=115.0,
    end_lat=41.0,
    end_lon=117.5,
    n_points=80,
    start_time="2024-01-01",
    end_time="2024-01-31",
    time_method="mean",
)

plot_vertical_transect(
    transect,
    title="O3 vertical transect",
    save_path="./O3_vertical_transect.png",
)


"""