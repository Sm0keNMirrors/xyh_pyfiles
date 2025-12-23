#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nc_inspect.py
功能：
  1) 列出 nc 文件所有变量名
  2) 计算目标变量的“逐月均值数组” + “逐月空间平均值”
支持：
  - 交互式运行（默认）
  - 命令行参数运行（--file/--mode/--var/--out）
"""

import os
import sys
import argparse
import numpy as np
from netCDF4 import Dataset, num2date
from typing import Optional



def find_time_var(ds: Dataset):
    """尽量智能地找到时间变量名。"""
    # 1) 常见名字优先
    for name in ("time", "Times", "datetime", "date"):
        if name in ds.variables:
            v = ds.variables[name]
            units = getattr(v, "units", "")
            if isinstance(units, str) and "since" in units:
                return name

    # 2) 根据属性/units猜测
    for name, v in ds.variables.items():
        units = getattr(v, "units", "")
        axis = getattr(v, "axis", "")
        stdn = getattr(v, "standard_name", "")
        if (isinstance(axis, str) and axis.upper() == "T") or (isinstance(stdn, str) and stdn.lower() == "time"):
            return name
        if isinstance(units, str) and "since" in units and v.ndim == 1:
            return name

    return None


def month_slices_from_datetimes(dts):
    """
    输入 datetime-like 序列（长度为 nt），返回按月分组的连续 slice 列表：
    [(YYYY-MM, slice(start, end)), ...]  其中 end 为开区间
    假设时间序列按时间递增，且每个月在时间维上是连续的（常见于模式/再分析逐小时数据）。
    """
    nt = len(dts)
    if nt == 0:
        return []

    def ym(dt):
        return f"{dt.year:04d}-{dt.month:02d}"

    groups = []
    cur = ym(dts[0])
    start = 0
    for i in range(1, nt):
        m = ym(dts[i])
        if m != cur:
            groups.append((cur, slice(start, i)))
            cur = m
            start = i
    groups.append((cur, slice(start, nt)))
    return groups


def compute_monthly_mean(ds: Dataset, varname: str, time_varname: str, chunk_t: int = 168):
    """
    返回：
      months: list[str]，例如 ["2024-08", ...]
      monthly_means: np.ndarray，shape = (n_month, ...)   # 去掉 time 维后剩余维度
      monthly_spatial_means: np.ndarray，shape = (n_month,)  # 对每个月的数组做空间均值
    """
    v = ds.variables[varname]
    tvar = ds.variables[time_varname]

    # 确保 time 是 1D
    if tvar.ndim != 1:
        raise ValueError(f"时间变量 {time_varname} 不是 1D（ndim={tvar.ndim}），请检查文件。")

    # 找到变量的 time 维位置（通常是第 0 维）
    time_dim_name = tvar.dimensions[0]
    if time_dim_name not in v.dimensions:
        raise ValueError(
            f"变量 {varname} 的维度 {v.dimensions} 中不包含 time 维 {time_dim_name}。"
        )
    time_axis = v.dimensions.index(time_dim_name)

    # 读取时间并转 datetime
    units = getattr(tvar, "units", None)
    calendar = getattr(tvar, "calendar", "standard")
    if units is None:
        raise ValueError(f"时间变量 {time_varname} 缺少 units，无法转换为日期。")

    time_vals = tvar[:]
    dts = num2date(time_vals, units=units, calendar=calendar)

    # 将 time 划分为按月的连续片段
    groups = month_slices_from_datetimes(dts)
    if not groups:
        raise ValueError("未能按月分组（time 长度为 0？）。")

    # monthly mean 的输出 shape（去掉 time 维）
    full_shape = v.shape
    out_shape = tuple(full_shape[i] for i in range(len(full_shape)) if i != time_axis)

    months = []
    monthly_means = []
    monthly_spatial_means = []

    # 按月计算：用 sum/count 累积，避免一次读完整个月造成内存压力
    for mon, sl in groups:
        months.append(mon)

        # 初始化累积
        sum_arr = np.zeros(out_shape, dtype=np.float64)
        cnt_arr = np.zeros(out_shape, dtype=np.int64)

        # 将 slice 映射到变量索引（支持 time 维不在第 0 维）
        # 通过构造切片列表实现 v[...]
        start, end = sl.start, sl.stop

        # 分块读取
        t0 = start
        while t0 < end:
            t1 = min(t0 + chunk_t, end)

            # 构造索引：除了 time_axis 用 slice(t0,t1)，其他用 :
            idx = [slice(None)] * v.ndim
            idx[time_axis] = slice(t0, t1)

            arr = v[tuple(idx)]  # 可能是 masked array（推荐）
            marr = np.ma.array(arr)

            # 沿 time_axis 做 sum/count（注意：此处 time_axis 在当前 arr 里仍保持原位置）
            s = np.ma.sum(marr, axis=time_axis)
            c = np.ma.count(marr, axis=time_axis)

            sum_arr += np.ma.filled(s, 0.0)
            cnt_arr += c.astype(np.int64)

            t0 = t1

        mean_arr = np.full(out_shape, np.nan, dtype=np.float64)
        valid = cnt_arr > 0
        mean_arr[valid] = sum_arr[valid] / cnt_arr[valid]

        monthly_means.append(mean_arr)

        # 空间均值：对所有非时间维求均值
        monthly_spatial_means.append(np.nanmean(mean_arr))

    monthly_means = np.stack(monthly_means, axis=0)
    monthly_spatial_means = np.array(monthly_spatial_means, dtype=np.float64)

    return months, monthly_means, monthly_spatial_means


def list_variables(ds: Dataset):
    print("\n========== Variables ==========")
    for i, name in enumerate(ds.variables.keys()):
        v = ds.variables[name]
        dims = ",".join(v.dimensions)
        print(f"[{i:03d}] {name:30s}  dims=({dims})  shape={v.shape}")
    print("================================\n")


def run_list_mode(nc_path: str):
    with Dataset(nc_path, "r") as ds:
        list_variables(ds)


def run_mean_mode(nc_path: str, varname: str, out_path: Optional[str] = None):
    with Dataset(nc_path, "r") as ds:
        ds.set_auto_maskandscale(True)

        time_name = find_time_var(ds)
        if time_name is None:
            raise RuntimeError("未找到时间变量（units 含 'since' 的 1D 变量），请手动检查。")

        if varname not in ds.variables:
            raise KeyError(f"变量 {varname} 不在文件中。")

        months, mm, sp = compute_monthly_mean(ds, varname, time_name)

    # 打印逐月空间均值
    print("\n========== Monthly spatial mean ==========")
    for m, val in zip(months, sp):
        print(f"{m}  spatial_mean = {val:.6g}")
    print("=========================================\n")

    # 保存逐月均值数组（推荐保存，避免终端刷屏/超大输出）
    if out_path is None:
        base = os.path.splitext(os.path.basename(nc_path))[0]
        out_path = f"{base}.{varname}.monthly_mean.npz"

    np.savez_compressed(out_path, months=np.array(months), monthly_mean=mm, monthly_spatial_mean=sp)
    print(f"已保存：{out_path}")
    print(f"monthly_mean shape = {mm.shape}")


def interactive():
    print("xyhTools_Cli (v0.1) by GPT5.2/xianyh_SCU")
    nc_path = input("请输入 nc 文件路径：").strip().strip('"').strip("'")
    if not nc_path:
        print("未输入路径，退出。")
        sys.exit(1)
    if not os.path.exists(nc_path):
        print(f"文件不存在：{nc_path}")
        sys.exit(1)

    print("\n请选择功能（输入代号）：")
    print("  a) 输出 nc 文件的所有变量名")
    print("  b) 输出目标变量的月均数组 + 月均空间平均值（并保存 npz）")
    choice = input("请输入代号 a/b：").strip().lower()

    if choice == "a":
        run_list_mode(nc_path)
    elif choice == "b":
        varname = input("请输入目标变量名（如 O3 / ACONC_O3 / T2 等）：").strip()
        if not varname:
            print("未输入变量名，退出。")
            sys.exit(1)
        out_path = input("可选：输出文件名（直接回车则自动命名为 *.npz）：").strip()
        out_path = out_path if out_path else None
        run_mean_mode(nc_path, varname, out_path)
    else:
        print("无效代号，退出。")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Inspect netCDF variables and compute monthly mean.")
    parser.add_argument("--file", "-f", help="nc 文件路径（不提供则进入交互式）")
    parser.add_argument("--mode", "-m", choices=["list", "mean"], help="list: 列变量；mean: 算月均")
    parser.add_argument("--var", "-v", help="目标变量名（mode=mean 时需要）")
    parser.add_argument("--out", "-o", help="输出 npz 路径（mode=mean 可选）")
    args = parser.parse_args()

    if not args.file:
        interactive()
        return

    if not os.path.exists(args.file):
        print(f"文件不存在：{args.file}")
        sys.exit(1)

    if not args.mode:
        # 有 --file 但没 --mode，也走交互（少打参数也能用）
        interactive()
        return

    if args.mode == "list":
        run_list_mode(args.file)
    else:
        if not args.var:
            print("mode=mean 时必须提供 --var 变量名（或不带参数走交互式）。")
            sys.exit(1)
        run_mean_mode(args.file, args.var, args.out)


if __name__ == "__main__":
    main()
