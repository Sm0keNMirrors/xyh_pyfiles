from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Union
import re

import numpy as np
import pandas as pd
import xarray as xr

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def print_info(message: str, verbose: bool = True):
    """
    打印过程信息。
    """
    if verbose:
        print(f"[INFO] {message}", flush=True)


def find_file_sequence(
    input_dir: Union[str, Path],
    file_pattern: str,
    recursive: bool = False,
    sort_regex: Optional[str] = None,
    sort_datetime_format: Optional[str] = None,
) -> list[str]:
    """
    获取指定目录下符合命名规则的 NetCDF 文件序列。
    """
    input_dir = Path(input_dir)

    if recursive:
        files = list(input_dir.rglob(file_pattern))
    else:
        files = list(input_dir.glob(file_pattern))

    if not files:
        raise FileNotFoundError(
            f"未找到文件：input_dir={input_dir}, file_pattern={file_pattern}"
        )

    if sort_regex and sort_datetime_format:
        pattern = re.compile(sort_regex)

        def sort_key(path: Path):
            match = pattern.search(path.name)
            if not match:
                raise ValueError(f"文件名中未匹配到时间信息：{path.name}")
            return datetime.strptime(match.group(1), sort_datetime_format)

        files = sorted(files, key=sort_key)
    else:
        files = sorted(files, key=lambda p: p.name)

    return [str(p) for p in files]


def infer_time_dim(data_array: xr.DataArray) -> str:
    """
    默认认为变量的第一个维度是时间维。

    例如：
    CMAQ: O3(TSTEP, LAY, ROW, COL) -> TSTEP
    WRF : T2(Time, south_north, west_east) -> Time
    """
    if len(data_array.dims) == 0:
        raise ValueError("目标变量没有维度，无法判断时间维。")

    return data_array.dims[0]


def decode_wrf_times(ds: xr.Dataset) -> Optional[pd.DatetimeIndex]:
    """
    解码 WRF 的 Times 变量。

    WRF Times 通常格式：
    Times(Time, DateStrLen)
    """
    if "Times" not in ds:
        return None

    times = ds["Times"].values
    decoded = []

    for item in times:
        if isinstance(item, (bytes, np.bytes_)):
            s = item.decode("utf-8")
        else:
            if hasattr(item, "dtype") and item.dtype.kind in ("S", "a"):
                s = b"".join(item).decode("utf-8")
            else:
                s = "".join(item.astype(str).tolist())

        s = s.strip().replace("\x00", "")
        decoded.append(pd.to_datetime(s, format="%Y-%m-%d_%H:%M:%S"))

    return pd.DatetimeIndex(decoded)


def decode_cmaq_tflag(
    ds: xr.Dataset,
    var_name: str,
    tflag_name: str = "TFLAG",
    n_tsteps: Optional[int] = None,
) -> Optional[pd.DatetimeIndex]:
    """
    解码 CMAQ / IOAPI 的 TFLAG 时间变量。

    TFLAG 通常格式：
    TFLAG(TSTEP, VAR, DATE-TIME)

    DATE: YYYYDDD，例如 2024001
    TIME: HHMMSS，例如 130000
    """
    if tflag_name not in ds:
        return None

    tflag = ds[tflag_name].values

    if tflag.ndim != 3 or tflag.shape[-1] < 2:
        return None

    var_index = 0

    var_list = ds.attrs.get("VAR-LIST")
    if isinstance(var_list, str):
        names = var_list.split()
        if var_name in names:
            var_index = names.index(var_name)

    if var_index >= tflag.shape[1]:
        var_index = 0

    datetimes = []

    if n_tsteps is None:
        n_tsteps = tflag.shape[0]
    else:
        n_tsteps = min(int(n_tsteps), tflag.shape[0])

    for i in range(n_tsteps):
        yyyyddd = int(tflag[i, var_index, 0])
        hhmmss = int(tflag[i, var_index, 1])

        if yyyyddd <= 0:
            raise ValueError(
                f"TFLAG 第 {i} 个时间步日期无效：{yyyyddd}。"
                "如果 CMAQ 文件每天包含 25 个 TSTEP 且最后一个为空，"
                "请在 combine 时启用 cmaq_drop_last_tstep=True。"
            )

        year = yyyyddd // 1000
        day_of_year = yyyyddd % 1000

        hour = hhmmss // 10000
        minute = (hhmmss % 10000) // 100
        second = hhmmss % 100

        dt = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
        dt = dt.replace(hour=hour, minute=minute, second=second)

        datetimes.append(dt)

    return pd.DatetimeIndex(datetimes)


def get_time_coordinate(
    ds: xr.Dataset,
    data_array: xr.DataArray,
    var_name: str,
    time_dim: str,
    time_source: str = "auto",
) -> Optional[Union[pd.DatetimeIndex, np.ndarray]]:
    """
    获取时间坐标。

    time_source 可选：
    - "auto"  : 自动判断
    - "wrf"   : 使用 WRF Times
    - "cmaq"  : 使用 CMAQ TFLAG
    - "coord" : 使用已有时间坐标
    - "none"  : 使用连续整数时间索引
    """
    valid_sources = {"auto", "wrf", "cmaq", "coord", "none"}
    if time_source not in valid_sources:
        raise ValueError(f"time_source 必须是 {valid_sources} 之一，当前为 {time_source}")

    if time_source == "none":
        return None

    expected_len = data_array.sizes[time_dim]

    if time_source in ("coord", "auto"):
        if time_dim in ds.coords:
            values = ds[time_dim].values
            if len(values) >= expected_len:
                return values[:expected_len]

    if time_source in ("wrf", "auto"):
        wrf_time = decode_wrf_times(ds)
        if wrf_time is not None and len(wrf_time) >= expected_len:
            return wrf_time[:expected_len]

    if time_source in ("cmaq", "auto"):
        cmaq_time = decode_cmaq_tflag(ds, var_name, n_tsteps=expected_len)
        if cmaq_time is not None and len(cmaq_time) == expected_len:
            return cmaq_time

    return None


def extract_gridcro2d_2d_variable(
    grid_ds: xr.Dataset,
    var_name: str,
    row_dim: str,
    col_dim: str,
) -> xr.DataArray:
    """
    从 GRIDCRO2D 中提取二维经纬度变量。

    GRIDCRO2D 中 LAT / LON 常见维度为：
    LAT(TSTEP, LAY, ROW, COL)
    LON(TSTEP, LAY, ROW, COL)

    本函数会自动取非 ROW / COL 维度的第 0 个切片，
    最终得到：
    LAT(ROW, COL)
    LON(ROW, COL)
    """
    if var_name not in grid_ds:
        raise KeyError(f"GRIDCRO2D 文件中不存在变量 {var_name}")

    da = grid_ds[var_name]

    if row_dim not in da.dims or col_dim not in da.dims:
        raise ValueError(
            f"GRIDCRO2D 变量 {var_name} 中未找到 {row_dim}, {col_dim} 维度。"
            f"当前维度为 {da.dims}"
        )

    indexers = {
        dim: 0
        for dim in da.dims
        if dim not in (row_dim, col_dim)
    }

    da_2d = da.isel(indexers, drop=True)
    da_2d = da_2d.transpose(row_dim, col_dim)

    return da_2d


def add_gridcro2d_latlon(
    out_ds: xr.Dataset,
    data_var_name: str,
    gridcro2d_path: Union[str, Path],
    engine: str = "netcdf4",
    grid_lat_name: str = "LAT",
    grid_lon_name: str = "LON",
    output_lat_name: str = "LAT",
    output_lon_name: str = "LON",
    verbose: bool = True,
) -> xr.Dataset:
    """
    从 CMAQ GRIDCRO2D 文件中读取 LAT / LON，并加入输出 Dataset。

    默认加入为二维坐标：
    LAT(ROW, COL)
    LON(ROW, COL)
    """
    gridcro2d_path = Path(gridcro2d_path)

    if not gridcro2d_path.exists():
        raise FileNotFoundError(f"GRIDCRO2D 文件不存在：{gridcro2d_path}")

    if data_var_name not in out_ds:
        raise KeyError(f"输出 Dataset 中不存在变量 {data_var_name}")

    data_da = out_ds[data_var_name]

    if "ROW" in data_da.dims:
        row_dim = "ROW"
    else:
        row_dim = data_da.dims[-2]

    if "COL" in data_da.dims:
        col_dim = "COL"
    else:
        col_dim = data_da.dims[-1]

    print_info(f"读取 GRIDCRO2D 文件：{gridcro2d_path}", verbose)
    print_info(f"经纬度目标维度：{row_dim}, {col_dim}", verbose)

    grid_ds = xr.open_dataset(
        gridcro2d_path,
        engine=engine,
        decode_times=False,
        mask_and_scale=True,
    )

    try:
        lat_2d = extract_gridcro2d_2d_variable(
            grid_ds=grid_ds,
            var_name=grid_lat_name,
            row_dim=row_dim,
            col_dim=col_dim,
        )

        lon_2d = extract_gridcro2d_2d_variable(
            grid_ds=grid_ds,
            var_name=grid_lon_name,
            row_dim=row_dim,
            col_dim=col_dim,
        )

        if lat_2d.sizes[row_dim] != out_ds.sizes[row_dim]:
            raise ValueError(
                f"GRIDCRO2D 的 {row_dim} 长度与输出数据不一致："
                f"{lat_2d.sizes[row_dim]} != {out_ds.sizes[row_dim]}"
            )

        if lat_2d.sizes[col_dim] != out_ds.sizes[col_dim]:
            raise ValueError(
                f"GRIDCRO2D 的 {col_dim} 长度与输出数据不一致："
                f"{lat_2d.sizes[col_dim]} != {out_ds.sizes[col_dim]}"
            )

        if lon_2d.sizes[row_dim] != out_ds.sizes[row_dim]:
            raise ValueError(
                f"GRIDCRO2D 的 {row_dim} 长度与输出数据不一致："
                f"{lon_2d.sizes[row_dim]} != {out_ds.sizes[row_dim]}"
            )

        if lon_2d.sizes[col_dim] != out_ds.sizes[col_dim]:
            raise ValueError(
                f"GRIDCRO2D 的 {col_dim} 长度与输出数据不一致："
                f"{lon_2d.sizes[col_dim]} != {out_ds.sizes[col_dim]}"
            )

        out_ds = out_ds.assign_coords(
            {
                output_lat_name: (
                    (row_dim, col_dim),
                    lat_2d.values,
                    {
                        "long_name": "latitude from GRIDCRO2D",
                        "standard_name": "latitude",
                        "units": "degrees_north",
                    },
                ),
                output_lon_name: (
                    (row_dim, col_dim),
                    lon_2d.values,
                    {
                        "long_name": "longitude from GRIDCRO2D",
                        "standard_name": "longitude",
                        "units": "degrees_east",
                    },
                ),
            }
        )

        out_ds.attrs["gridcro2d_file"] = str(gridcro2d_path)
        out_ds.attrs["gridcro2d_lat_var"] = grid_lat_name
        out_ds.attrs["gridcro2d_lon_var"] = grid_lon_name

        print_info(
            f"已加入经纬度坐标：{output_lat_name}({row_dim}, {col_dim}), "
            f"{output_lon_name}({row_dim}, {col_dim})",
            verbose,
        )

    finally:
        grid_ds.close()

    return out_ds


def Model_var_combine_by_time(
    input_dir: Union[str, Path],
    file_pattern: str,
    var_name: str,
    output_path: Optional[Union[str, Path]] = None,
    recursive: bool = False,
    sort_regex: Optional[str] = None,
    sort_datetime_format: Optional[str] = None,
    engine: str = "netcdf4",
    decode_times: bool = False,
    mask_and_scale: bool = True,
    chunks: Optional[dict] = None,
    time_source: str = "auto",
    output_var_name: Optional[str] = None,
    zlib: bool = True,
    complevel: int = 4,
    gridcro2d_path: Optional[Union[str, Path]] = None,
    grid_lat_name: str = "LAT",
    grid_lon_name: str = "LON",
    output_lat_name: str = "LAT",
    output_lon_name: str = "LON",
    cmaq_drop_last_tstep: bool = True,
    cmaq_expected_tstep_per_file: int = 25,
    cmaq_keep_tstep_per_file: int = 24,
    verbose: bool = True,
    show_progress: bool = True,
) -> xr.Dataset:
    """
    将多个 WRF / CMAQ / NetCDF 文件中的指定变量按照时间维拼接。

    时间维不需要手动指定，默认使用目标变量的第一个维度。

    常见数据结构：
    CMAQ:
        var(TSTEP, LAY, ROW, COL)

    WRF:
        var(Time, bottom_top, south_north, west_east)
        var(Time, south_north, west_east)

    新增功能：
    1. verbose=True 时打印处理过程信息
    2. show_progress=True 时用 tqdm 显示文件处理进度
    3. gridcro2d_path 不为 None 时，从 CMAQ GRIDCRO2D 文件中读取 LAT / LON 加入输出结果

    Parameters
    ----------
    input_dir : str | Path
        输入文件目录。

    file_pattern : str
        文件匹配模式，例如：
        "ACONC_*.nc"
        "CCTM_ACONC_*.nc"
        "wrfout_d01_*"

    var_name : str
        目标变量名，例如：
        "O3", "PM25_TOT", "T2", "U", "V"

    output_path : str | Path | None
        输出 NetCDF 文件路径。
        如果为 None，则只返回 xr.Dataset，不写文件。

    recursive : bool
        是否递归搜索子目录。

    sort_regex : str | None
        从文件名中提取时间信息的正则表达式。
        例如 r"(\\d{8})"。

    sort_datetime_format : str | None
        文件名时间格式。
        例如 "%Y%m%d"。

    engine : str
        xarray 读取 NetCDF 的后端。
        常用 "netcdf4"。

    decode_times : bool
        是否让 xarray 自动解码时间。
        WRF / CMAQ 通常建议 False。

    mask_and_scale : bool
        是否应用 scale_factor / add_offset / missing_value。

    chunks : dict | None
        dask 分块参数。
        大文件可设置，例如 {"TSTEP": 24} 或 {"Time": 24}。

    time_source : str
        时间坐标来源：
        - "auto"  : 自动识别
        - "wrf"   : 使用 WRF Times
        - "cmaq"  : 使用 CMAQ TFLAG
        - "coord" : 使用已有坐标
        - "none"  : 使用连续整数时间索引

    output_var_name : str | None
        输出变量名。
        如果为 None，则沿用 var_name。

    zlib : bool
        输出 NetCDF 是否压缩。

    complevel : int
        压缩等级，1 到 9。

    gridcro2d_path : str | Path | None
        CMAQ GRIDCRO2D 文件路径。
        如果提供，则从该文件读取 LAT / LON 并加入最终输出。

    grid_lat_name : str
        GRIDCRO2D 中纬度变量名，默认 "LAT"。

    grid_lon_name : str
        GRIDCRO2D 中经度变量名，默认 "LON"。

    output_lat_name : str
        输出文件中纬度坐标名称，默认 "LAT"。

    output_lon_name : str
        输出文件中经度坐标名称，默认 "LON"。

    verbose : bool
        是否打印处理信息。

    cmaq_drop_last_tstep : bool
        是否对 CMAQ 日文件自动忽略最后一个多余 TSTEP。
        默认 True。仅当识别为 CMAQ 数据且单文件时间步数等于
        cmaq_expected_tstep_per_file 时生效。

    cmaq_expected_tstep_per_file : int
        CMAQ 单文件原始 TSTEP 数。默认 25。

    cmaq_keep_tstep_per_file : int
        CMAQ 单文件实际参与 combine 的 TSTEP 数。默认 24。

    show_progress : bool
        是否显示 tqdm 进度条。

    Returns
    -------
    xr.Dataset
        拼接后的单变量 Dataset。
    """

    print_info("开始查找输入文件", verbose)

    files = find_file_sequence(
        input_dir=input_dir,
        file_pattern=file_pattern,
        recursive=recursive,
        sort_regex=sort_regex,
        sort_datetime_format=sort_datetime_format,
    )

    print_info(f"输入目录：{input_dir}", verbose)
    print_info(f"文件匹配模式：{file_pattern}", verbose)
    print_info(f"找到文件数：{len(files)}", verbose)
    print_info(f"目标变量：{var_name}", verbose)

    if len(files) > 0:
        print_info(f"第一个文件：{files[0]}", verbose)
        print_info(f"最后一个文件：{files[-1]}", verbose)

    if show_progress and tqdm is None:
        print_info("未安装 tqdm，无法显示进度条；可运行 pip install tqdm", verbose)

    data_arrays = []

    final_time_dim = None
    non_time_signature = None
    time_offset = 0
    total_time_steps = 0

    if show_progress and tqdm is not None:
        file_iterator = tqdm(
            files,
            desc=f"Combining {var_name}",
            unit="file",
        )
    else:
        file_iterator = files

    for file_index, file_path in enumerate(file_iterator, start=1):
        ds = xr.open_dataset(
            file_path,
            engine=engine,
            decode_times=decode_times,
            mask_and_scale=mask_and_scale,
            chunks=chunks,
        )

        try:
            if var_name not in ds:
                raise KeyError(f"文件 {file_path} 中不存在变量 {var_name}")

            da = ds[var_name]

            current_time_dim = infer_time_dim(da)

            if final_time_dim is None:
                final_time_dim = current_time_dim
                print_info(f"自动识别时间维：{final_time_dim}", verbose)
                print_info(f"变量维度：{da.dims}", verbose)
                print_info(
                    f"变量维度长度：{dict(da.sizes)}",
                    verbose,
                )

            elif current_time_dim != final_time_dim:
                raise ValueError(
                    f"时间维名称不一致：基准为 {final_time_dim}，"
                    f"当前文件 {file_path} 为 {current_time_dim}"
                )

            current_signature = tuple(
                (dim, da.sizes[dim])
                for dim in da.dims[1:]
            )

            if non_time_signature is None:
                non_time_signature = current_signature
                print_info(
                    f"非时间维签名：{non_time_signature}",
                    verbose,
                )

            elif current_signature != non_time_signature:
                raise ValueError(
                    "非时间维不一致，无法拼接。\n"
                    f"基准维度: {non_time_signature}\n"
                    f"当前文件: {file_path}\n"
                    f"当前维度: {current_signature}"
                )

            nt_original = da.sizes[current_time_dim]

            is_cmaq_like = (
                time_source == "cmaq"
                or (
                    time_source == "auto"
                    and current_time_dim.upper() == "TSTEP"
                    and "TFLAG" in ds
                )
            )

            if (
                cmaq_drop_last_tstep
                and is_cmaq_like
                and nt_original == cmaq_expected_tstep_per_file
            ):
                da = da.isel(
                    {
                        current_time_dim: slice(
                            0,
                            cmaq_keep_tstep_per_file,
                        )
                    }
                )
                print_info(
                    f"CMAQ 文件 {Path(file_path).name} 原始 TSTEP={nt_original}，"
                    f"已忽略最后 {nt_original - cmaq_keep_tstep_per_file} 个，"
                    f"仅使用前 {cmaq_keep_tstep_per_file} 个时间步",
                    verbose,
                )
            elif cmaq_drop_last_tstep and is_cmaq_like:
                print_info(
                    f"CMAQ 文件 {Path(file_path).name} 当前 TSTEP={nt_original}，"
                    f"不等于 cmaq_expected_tstep_per_file={cmaq_expected_tstep_per_file}，"
                    "未自动裁剪",
                    verbose,
                )

            nt = da.sizes[current_time_dim]
            total_time_steps += nt

            time_coord = get_time_coordinate(
                ds=ds,
                data_array=da,
                var_name=var_name,
                time_dim=current_time_dim,
                time_source=time_source,
            )

            if time_coord is None:
                time_coord = np.arange(time_offset, time_offset + nt)

            time_offset += nt

            da = da.assign_coords({current_time_dim: time_coord})

            if output_var_name is not None:
                da = da.rename(output_var_name)

            # load 后可以安全关闭当前文件。
            # 如果数据极大，这一步会占用内存；但对文件句柄管理更安全。
            da = da.load()

            data_arrays.append(da)

            if verbose and not show_progress:
                print_info(
                    f"已处理 {file_index}/{len(files)}：{Path(file_path).name}, "
                    f"时间步数={nt}",
                    verbose,
                )

        finally:
            ds.close()

    print_info("开始沿时间维拼接数据", verbose)

    combined = xr.concat(
        data_arrays,
        dim=final_time_dim,
        coords="minimal",
        compat="override",
        combine_attrs="drop_conflicts",
    )

    out_name = output_var_name or var_name
    out_ds = combined.to_dataset(name=out_name)

    out_ds.attrs["combined_by"] = "combine_model_var_by_time"
    out_ds.attrs["time_dimension"] = final_time_dim
    out_ds.attrs["source_file_count"] = len(files)
    out_ds.attrs["total_time_steps"] = total_time_steps
    out_ds.attrs["cmaq_drop_last_tstep"] = str(cmaq_drop_last_tstep)
    out_ds.attrs["cmaq_expected_tstep_per_file"] = cmaq_expected_tstep_per_file
    out_ds.attrs["cmaq_keep_tstep_per_file"] = cmaq_keep_tstep_per_file
    out_ds.attrs["source_files"] = "\n".join(files)

    print_info("拼接完成", verbose)
    print_info(f"输出变量名：{out_name}", verbose)
    print_info(f"总时间步数：{total_time_steps}", verbose)
    print_info(f"输出数据维度：{dict(out_ds[out_name].sizes)}", verbose)

    if gridcro2d_path is not None:
        print_info("检测到 gridcro2d_path，开始加入 CMAQ 经纬度信息", verbose)

        out_ds = add_gridcro2d_latlon(
            out_ds=out_ds,
            data_var_name=out_name,
            gridcro2d_path=gridcro2d_path,
            engine=engine,
            grid_lat_name=grid_lat_name,
            grid_lon_name=grid_lon_name,
            output_lat_name=output_lat_name,
            output_lon_name=output_lon_name,
            verbose=verbose,
        )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print_info(f"开始写出 NetCDF：{output_path}", verbose)

        encoding = {
            out_name: {
                "zlib": zlib,
                "complevel": complevel,
            }
        }

        out_ds.to_netcdf(
            output_path,
            engine=engine,
            format="NETCDF4",
            encoding=encoding,
        )

        print_info(f"写出完成：{output_path}", verbose)

    return out_ds


"""

ds = Model_var_combine_by_time(
    input_dir="/data/CMAQ/ACONC",
    file_pattern="ACONC_*.nc",
    var_name="O3",
    output_path="./O3_combined.nc",
    time_source="cmaq",
    # CMAQ 日文件若为 25 个 TSTEP，默认自动忽略最后一个空 TSTEP，只合并前 24 个
    cmaq_drop_last_tstep=True,
)
"""


# ds = Model_var_combine_by_time(
#     input_dir=rf"E:\SichuanBasin_2024Ozone\WRFout_d03\2024\\",
#     file_pattern="wrfout*",
#     var_name="T2",
#     output_path="./T2_combined.nc",
#     time_source="wrf",
#     output_lat_name='LAT',
#     output_lon_name='LONG',
# )