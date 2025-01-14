

import os
from multiprocessing import Pool
import glob
import os.path
import tqdm
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import re
from osgeo import gdal, osr
import pandas as pd
import PseudoNetCDF as pnc
import datetime
import netCDF4 as nc
import numpy as np
from tqdm import tqdm



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

def getMultiNearestPos(station_coords, XLAT, XLONG,num_processes):
    """
    对多个站点并行找到最近的经纬度索引值
    :param station_coords: 包含多个站点的经纬度的列表 [(lat1, lon1), (lat2, lon2), ...]
    :param XLAT: 纬度数组
    :param XLONG: 经度数组
    :return: 最近经纬度的索引列表
    """
    # 创建参数列表
    args = [(lat, lon, XLAT, XLONG) for lat, lon in station_coords]

    with Pool(processes=num_processes) as pool:
        results = pool.map(getNearestPos, args)

    return results


def data_distributeEvenly(data, x_target, y_target):
    """
    将低分辨率的数据平均重采样到高分辨率数据，用于MEIC
    :param data:
    :param x_target:
    :param y_target:
    :return:
    """
    target_data = np.zeros((x_target, y_target))
    # 计算每个原始数据块在目标网格中对应的大小
    row_ratio = x_target // data.shape[0]
    col_ratio = y_target // data.shape[1]
    # 将原始数据分配到目标网格
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            target_data[i * row_ratio:(i + 1) * row_ratio, j * col_ratio:(j + 1) * col_ratio] = data[i, j] / (
                        row_ratio * col_ratio)

    return target_data

def getLatLonAreaFormTiff(
        input_tif,
        latmax, lonmin, latmin, lonmax,
        disable_tqdm=False
):
    """
    根据tiff的信息，先获取其对应行列号的经纬度及其数据，
    然后计算对应纬度范围内的数据所在的array行列范围，同时范围对应的经纬度数组
    :return:
    """
    with rasterio.open(input_tif) as src:
        # 读取数据数组
        data_array = src.read(1)  # 读取第一个波段的数据
        # 获取仿射变换参数
        transform = src.transform

        # 获取经纬度网格点
        height, width = data_array.shape
        LON, LAT = np.meshgrid(
            np.arange(width) * transform[0] + transform[2],  # 经度
            np.arange(height) * transform[4] + transform[5]  # 纬度
        )
    # print(LAT,LON)
    # 根据geoem四川盆地左上和右下点，筛选出四川盆地内的部分
    leftup = getNearestPos(latmax, lonmin, LAT, LON)  # 从WRF得到站点格子
    rightdown = getNearestPos(latmin, lonmax, LAT, LON)  # 从WRF得到站点格子

    data = data_array[leftup[0]:rightdown[0], leftup[1]:rightdown[1]]
    LAT = LAT[leftup[0]:rightdown[0], leftup[1]:rightdown[1]]
    LON = LON[leftup[0]:rightdown[0], leftup[1]:rightdown[1]]

    return [data, LAT, LON]


def save_2tiff(out_path, tif_data, ROW, COL, lonmin, lon_res, latmax, lat_res):
    # 创建tif文件
    driver = gdal.GetDriverByName('GTiff')

    # 创建框架
    out_tif = driver.Create(out_path, COL, ROW, 1, gdal.GDT_Float32)
    # 设置影像显示范围
    geotransfor = (lonmin, lon_res, 0, latmax, 0, -lat_res)  # -latres < 0
    out_tif.SetGeoTransform(geotransfor)

    # 获取地理坐标系统信息
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)  # 定义输出的坐标系为"WGS 84"，AUTHORITY["EPSG","4326"]
    out_tif.SetProjection(srs.ExportToWkt())  # 赋予投影信息

    # 数据写出
    out_tif.GetRasterBand(1).WriteArray(tif_data)
    out_tif.FlushCache()
    out_tif = None


def tiff2ArrayWithLatLon(tif_dir):
    """
    读取一个tiff，得到它的数据以及经纬度数组，并返回像素行列长宽(分辨率)
    :param tif_dir:
    :return:
    """
    with rasterio.open(tif_dir) as src:
        # 读取数据数组
        data_array = src.read(1)  # 读取第一个波段的数据
        # 获取仿射变换参数
        transform = src.transform

        p_x = transform[0]  #
        p_y = -transform[4]

        # 获取经纬度网格点
        height, width = data_array.shape
        lon, lat = np.meshgrid(
            np.arange(width) * transform[0] + transform[2],  # 经度
            np.arange(height) * transform[4] + transform[5]  # 纬度
        )

    return [data_array, lat, lon, p_x, p_y]

def MEIC2Geotiff(input_dir,output_dir):
    # @Time      :2023/4/29 10:48
    # @Author    :Jiaxin Qiu
    print("MEIC2Geotiff: This script is written by Jiaxin Qiu.")
    # ------------------------------------------
    # input_dir = r"D:\MEIC\2020_Lambert"
    # output_dir = r"D:\MEIC\2020_Lambert_tiff"
    # ------------------------------------------
    if os.path.exists(output_dir) is False:
        os.mkdir(output_dir)

    files = glob.glob(f"{input_dir}/*.asc")

    for file in tqdm(files):
        sub_name = os.path.basename(file)
        condition = f"(.*?)_(.*?)_(.*?)_(.*?).asc"
        # condition = f"(.*?)_(.*?)__(.*?)__(.*?).asc"  # For 2019 and 2020.
        encode_name = re.findall(condition, sub_name)[0]
        year = r"%.4d" % int(encode_name[0])
        mm = r"%.2d" % int(encode_name[1])
        sector = encode_name[2]
        pollutant = encode_name[3].replace(".", "")
        output_name = f"{output_dir}/MEIC_{year}_{mm}__{sector}__{pollutant}.tiff"

        # 以只读模式打开文件
        with open(file, 'r', encoding='utf-8') as file:
            # 逐行读取文件内容，并以空格为分隔符分割每行，最后将其添加到数组（列表）中
            lines = [line.strip().split() for index, line in enumerate(file) if index >= 6]

        # 打印读取到的数组（列表）
        _ = np.array(lines, dtype="float")
        z = np.where(_ == -9999.0, 0.0, _)

        # 最大最小经纬度
        min_long, min_lat, max_long, max_lat = 70.0, 10.0, 150.0, 60.0

        # 分辨率
        x_resolution = 0.25
        y_resolution = 0.25

        # 计算栅格的行和列
        width = int((max_long - min_long) / x_resolution)
        height = int((max_lat - min_lat) / y_resolution)

        # 创建GeoTIFF文件的变换矩阵
        transform = from_bounds(min_long, min_lat, max_long, max_lat, width, height)

        # 定义GeoTIFF文件的元数据
        metadata = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": rasterio.float32,
            "crs": CRS.from_epsg(4326),
            "transform": transform,
        }

        # 创建GeoTIFF文件
        with rasterio.open(output_name, "w", **metadata) as dst:
            dst.write(z, 1)  # 将数据写入波段1

def CMAQ_MEICAveEmission_generation(
    Workdir = r"", # 程序运行输出中间文件和最终结果的路径
    GRIDDESC_dir = "",
    GRIDNAME = "",
    GRIDCRO2D_dir = "",
    sectors = ['transportation', 'residential', 'power', 'agriculture', 'industry'], # 生成的部门，MEIC5部门中选择输出哪些
    MEIC_dir = "", # MEIC原始ASCII文件所在路径
    start_date = "YYYY-MM-DD", # 生成清单的开始日期
    end_date = "YYYY-MM-DD", # 生成清单的结束日期
    factor_csvfiles_dir = "", #存放分配系数的csv文件所在目录
):
    """
    基于MEIAT-CMAQ的相关方法，输出MEIC清单平均分配方式的排放清单，能直接用于CMAQ的运行，仅支持CB06，
    需要MEIAT-CMAQ格式的时间分配系数csv，下载页面：
    ①https://github.com/Airwhf/MEIAT-CMAQ/tree/main/species
    ②https://github.com/Airwhf/MEIAT-CMAQ/tree/main/temporal
    参考：MEIAT-CMAQ：https://github.com/Airwhf/MEIAT-CMAQ
    :return:
    """
    # PseudoNetCDF打开griddesc文件，并生成对应的nc文件框架
    os.makedirs(Workdir, exist_ok=True)
    grid_desc = GRIDDESC_dir
    grid_name = GRIDNAME
    GRIDCRO2D_dir = GRIDCRO2D_dir

    target_mechanism = 'CB06'

    sectors = sectors  # 生成的排放源部门

    # 转化MEIC原始数据为tiff：
    MEIC_tiffs_dir = Workdir + 'MEIC_tiffs//'
    os.makedirs(MEIC_tiffs_dir, exist_ok=True)
    # MEIC2Geotiff(MEIC_dir,MEIC_tiffs_dir)

    start_date = start_date
    end_date = end_date

    output_dir = Workdir + 'output_emission//'
    os.makedirs(output_dir, exist_ok=True)

    periods = pd.period_range(pd.to_datetime(start_date), pd.to_datetime(end_date), freq='D')

    # 从GRIDCRO2D获取经纬度范围信息
    GRIDCRO2D = nc.Dataset(GRIDCRO2D_dir)
    LON = np.array(GRIDCRO2D.variables['LON'][:][0][0])
    LAT = np.array(GRIDCRO2D.variables['LAT'][:][0][0])
    GRID_p_x = LON[0][1] - LON[0][0]  # GRID的xy分辨率
    GRID_p_y = LAT[1][0] - LAT[0][0]
    print(GRID_p_x, GRID_p_y)
    ROW = LON.shape[0]  # 以LON作为标准数据来获取数组行列
    COL = LON.shape[1]
    lonmin, latmax, lonmax, latmin = (LON.min(), LAT.max(),
                                      LON.max(), LAT.min())  # 获取griddesc区域矩形经纬度范围
    data_Temp = np.zeros((ROW, COL), dtype=float, order="c")  # griddesc空间格点对应的空数组

    data_allocators_interp = {}

    for date in periods:
        month = str(date).split('-')[1]
        w = datetime.datetime.strftime(pd.to_datetime(str(date)), "%w")  # 此日为星期几
        yyyymmdd = datetime.datetime.strftime(pd.to_datetime(str(date)), "%Y%m%d")
        yyyyjjj = datetime.datetime.strftime(pd.to_datetime(str(date)), "%Y%j")
        for sector in sectors:
            print(date,sector)
            _weekly_factor = pd.read_csv(rf"{factor_csvfiles_dir}weekly.csv")
            _hourly_factor = pd.read_csv(rf"{factor_csvfiles_dir}hourly.csv")
            weekly_factor = _weekly_factor[sector].values
            hourly_factor = _hourly_factor[sector].values

            gf = pnc.pncopen(
                grid_desc,
                GDNAM=grid_name,
                format="griddesc",
                SDATE=int(yyyyjjj),
                TSTEP=10000,
                withcf=False,
            )
            gf.updatetflag(overwrite=True)
            tmpf = gf.sliceDimensions(TSTEP=[0] * 25)
            max_col_index = getattr(tmpf, "NCOLS") - 1
            max_row_index = getattr(tmpf, "NROWS") - 1

            # Read species file.
            species_file = f"{factor_csvfiles_dir}MEIC-CB05_CB06_speciate_{sector}.csv"
            species_info = pd.read_csv(species_file)
            fname_list = species_info.pollutant.values
            var_list = species_info.emission_species.values
            factor_list = species_info.split_factor.values
            divisor_list = species_info.divisor.values
            origin_units = species_info.inv_unit.values
            target_units = species_info.emi_unit.values

            for emission_specie in species_info['emission_species']:

                if emission_specie == 'PMC':
                    continue
                colnum = list(species_info['emission_species']).index(emission_specie)  # 找到列号，以获取物种其他信息
                MEIC_tiff_dir = f"{MEIC_tiffs_dir}MEIC_2020_{month}__{sector}__{species_info['pollutant'][colnum]}.tiff"
                data_MEIC_pre = getLatLonAreaFormTiff(MEIC_tiff_dir, latmax, lonmin, latmin, lonmax)
                data_MEIC, LAT_MEIC, LON_MEIC = data_MEIC_pre[0], data_MEIC_pre[1], data_MEIC_pre[2]
                data_MEIC_sum = np.sum(data_MEIC)

                # 插值 MEIC 到目标分辨率
                for i in range(data_Temp.shape[0]):
                    for j in range(data_Temp.shape[1]):
                        lon = LON[i, j]
                        lat = LAT[i, j]
                        MEICloc_lat = getNearestPos(lat, lon, LAT_MEIC, LON_MEIC)
                        if MEICloc_lat.ndim == 2: MEICloc_lat = [MEICloc_lat[0][1], MEICloc_lat[1][1]]  # 处理特殊情况采样到多个点
                        data_Temp[i, j] = data_MEIC[MEICloc_lat[0], MEICloc_lat[1]]

                data_MEIC_interp_ = data_Temp * (data_MEIC_sum / np.sum(data_Temp))  # 总量不变

                # 空间分配
                predicted_grid = data_MEIC_interp_

                predicted_grid[np.isnan(predicted_grid)] = 0  # 设空值为0，一般在农业源的颗粒物排放会遇到
                # Convert monthly emission to weekly emission. 整月转换到周等级，以月分配系数计算，这里默认0.25 4周一月
                weekly_values = predicted_grid * 0.25
                # Convert weekly emission to daily emission. # 周分配系数转换到日，w为这周的星期几，即找到7个周分配系数中对应的某星期几
                daily_values = weekly_values * weekly_factor[int(w)]
                # 转换到小时分配，24个小时
                hourly_values = np.zeros([25, 1, ROW, COL])
                # print('sghh', hourly_values.shape)
                for hour in range(24):
                    hourly_values[hour, 0, :, :] += daily_values * hourly_factor[hour]
                # print('wdwfghgh', hourly_values.shape)

                # Convert original units to target units and input the split_factor. 单位转换以及split_factor加权
                origin_unit = species_info['inv_unit'][colnum]
                target_unit = species_info['emi_unit'][colnum]
                split_factor = species_info['split_factor'][colnum]
                divisor = species_info['divisor'][colnum]
                # print(origin_unit, target_unit, split_factor, divisor)
                # Convert original units to target units and input the split_factor.
                if origin_unit == "Mmol" and target_unit == "mol/s":
                    hourly_values = hourly_values * 1000000.0 / 3600.0 * split_factor
                elif origin_unit == "Mg" and target_unit == "g/s":
                    hourly_values = hourly_values * 1000000.0 / 3600.0 * split_factor
                elif origin_unit == "Mg" and target_unit == "mol/s":
                    hourly_values = (hourly_values * 1000000.0 / 3600.0 / divisor * split_factor)

                emission_specie_var = tmpf.createVariable(emission_specie, "f", ("TSTEP", "LAY", "ROW", "COL"))
                if target_unit == "mol/s":
                    emission_specie_var.setncatts(
                        dict(units="moles/s", long_name=emission_specie, var_desc=emission_specie))
                elif target_unit == "g/s":
                    emission_specie_var.setncatts(
                        dict(units="g/s", long_name=emission_specie, var_desc=emission_specie))
                # print('wdwfghgh', hourly_values.shape)
                emission_specie_var[:, 0, :, :] = hourly_values[:, 0, :, :]

            # Get rid of initial DUMMY variable
            del tmpf.variables["DUMMY"]

            # Update TFLAG to be consistent with variables
            tmpf.updatetflag(tstep=10000, overwrite=True)

            # Remove VAR-LIST so that it can be inferred
            delattr(tmpf, "VAR-LIST")
            tmpf.updatemeta()

            output_name = fr"{output_dir}/{target_mechanism}_{sector}_{grid_name}_{yyyymmdd}.nc"  #
            tmpf.save(output_name, format="NETCDF3_CLASSIC")
            tmpf.close()


if __name__ == "__main__":
    CMAQ_MEICAveEmission_generation(
            Workdir=r"E:\xyhfiles_runtest\testemission\\",  # 程序运行输出中间文件和最终结果的路径
            GRIDDESC_dir="E:\WCAS_serverfiles\GRIDDESC",
            GRIDNAME="CDsvSA_d03",
            GRIDCRO2D_dir="E:\WCAS_serverfiles\GRIDCRO2D_2024333.nc",
            sectors=['transportation', 'residential', 'power', 'agriculture', 'industry'],  # 生成的部门，MEIC5部门中选择输出哪些
            MEIC_dir=r"E:\SichuanCMAQPMtrends\MEIC\2020\\",  # MEIC原始ASCII文件所在路径
            start_date="2025-01-14",  # 生成清单的开始日期
            end_date="2025-01-14",  # 生成清单的结束日期
            factor_csvfiles_dir=r"E:\xyhfiles_runtest\factor_files\\",  # 存放分配系数的csv文件所在目录
    )
    pass

