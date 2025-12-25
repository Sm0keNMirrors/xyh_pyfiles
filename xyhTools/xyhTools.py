
import time
import matplotlib
import netCDF4 as nc
import os
import numpy as np
import rasterio
from pyproj import Proj
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import pandas as pd
# from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde
import subprocess

def generate_dates(year):
    # 创建一个空列表来存储结果
    date_list = []

    # 设置起始日期为该年的第一天
    start_date = datetime.datetime(year, 1, 1, 0, 0)

    # 生成该年份的每个小时
    for hour in range(24 * 365):  # 假设不考虑闰年
        # 计算当前的日期时间
        current_date = start_date + datetime.timedelta(hours=hour)

        # 将日期格式化为字符串并添加到列表中
        date_list.append(current_date.strftime('%Y%m%d%H'))

    return date_list

def pickle_data(pkl_data,pkl_data_name,pkl_dir,type):
    import pickle
    """
    将数据处理后的变量pickle为pkl文件，在绘图过程直接读取不需要再绘制
    pkl_data: 保存的变量,w时有效
    :param pkl_dir: pkl文件路径
    :param type: r-读取pkl w-写pkl
    :return: r时返回读取的变量
    """
    if type == 'w':
        with open(f'{pkl_dir}{pkl_data_name}.pkl', 'wb') as pickle_file:
            pickle.dump(pkl_data, pickle_file)
    if type == 'r':
        with open(f'{pkl_dir}{pkl_data_name}.pkl', 'rb') as pickle_file:
            load_data = pickle.load(pickle_file)
        return load_data
    

def jupyter_run_subprocess(cmd,cwd):
    """
    能在ipynb文件中执行cmd，并将输出打印在ipynb中的功能函数
    :param cmd: 执行的指令
    :param cwd: 执行的指令所在目录
    """
    p = subprocess.Popen(
        cmd,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    for line in p.stdout:      # 实时逐行打印
        print(line, end='')
    p.wait()
    print("\n=== RETURN CODE ===", p.returncode)

def tiff2array(file_path):
    try:
        with rasterio.open(file_path) as dataset:
            # 读取所有波段的数据为 NumPy 数组
            array = dataset.read()  # 读取所有波段，返回的形状为 (波段数, 高, 宽)
        return array
    except Exception as e:
        print(f"读取 TIFF 文件时发生错误: {e}")
        return None
    
import numpy as np
import xarray as xr

def save_3d_array2nc(
    data,
    out_path,
    var_name="var",
    time=None,
    row_name="south_north",
    col_name="west_east",
):
    """
    将一个 shape=(time, row, col) 的 3D 数组简单保存为 NetCDF 文件.

    参数
    ----
    data : np.ndarray
        3D 数组，形状必须为 (time, row, col)
    out_path : str
        输出 nc 文件路径
    var_name : str
        变量名，例如 "PRECIP_HOURLY"
    time : 1D array-like, optional
        时间坐标，长度等于 time 维度长度。
        - 若为 None，则用 0,1,2,... 作为时间索引
        - 可以传 numpy datetime64, pandas.DatetimeIndex, 或普通数值
    row_name : str
        行维度名称，默认 "south_north"
    col_name : str
        列维度名称，默认 "west_east"
    """

    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError(f"data 必须是 3 维 (time,row,col)，当前维度为 {data.ndim}")

    nt, ny, nx = data.shape

    # 如果没给时间轴，就用简单的 0..nt-1
    if time is None:
        time = np.arange(nt)

    if len(time) != nt:
        raise ValueError(f"time 长度({len(time)}) 与 data 的 time 维({nt}) 不一致")

    da = xr.DataArray(
        data,
        dims=("time", row_name, col_name),
        coords={"time": time},
        name=var_name,
    )

    ds = da.to_dataset()
    ds.to_netcdf(out_path)
    print(f"[save_3d_array_to_nc] Saved to: {out_path}")


def getLatLonArea(
        data_array,LAT,LON,
        latmax, lonmin, latmin, lonmax,
        withtstep = False,
):
    """
    根据输入的data,lat,lon
    计算获取对应纬度范围内的数据所在的array行列范围，同时范围对应的经纬度数组
    withtstep: 是否有时间维度，则处理时保留第一维时间维度
    :return:
    """
    leftup = getNearestPos(latmax, lonmin, LAT, LON)  # 从WRF得到站点格子
    rightdown = getNearestPos(latmin, lonmax, LAT, LON)  # 从WRF得到站点格子

    if withtstep == True:
        data = data_array[:,leftup[0]:rightdown[0], leftup[1]:rightdown[1]]
        LAT = LAT[leftup[0]:rightdown[0], leftup[1]:rightdown[1]]
        LON = LON[leftup[0]:rightdown[0], leftup[1]:rightdown[1]]
    else:
        data = data_array[leftup[0]:rightdown[0], leftup[1]:rightdown[1]]
        LAT = LAT[leftup[0]:rightdown[0], leftup[1]:rightdown[1]]
        LON = LON[leftup[0]:rightdown[0], leftup[1]:rightdown[1]]

    return [data, LAT, LON]

def getLatlonAera_Fromshape(
        lat,lon,shape_dir
):
    """
    输入经纬度网格二维数组，以及一个经纬度shape，返回
    """
    import numpy as np
    import geopandas as gpd
    from shapely.geometry import Point, Polygon


    lat_array = lat
    lon_array = lon

    shapefile_path = shape_dir  # 请替换为你的shp文件路径
    gdf = gpd.read_file(shapefile_path)

    # 假设你只关心第一个形状，可以根据需要选择具体的形状
    polygon = gdf.geometry[0]

    # 初始化掩码
    mask = np.zeros(lat.shape, dtype=bool)

    # 遍历格点，检查每个点是否在多边形内
    for i in range(lat.shape[0]):
        for j in range(lat.shape[1]):
            point = Point(lon[i, j], lat[i, j])  # 注意顺序是 (经度, 纬度)
            mask[i, j] = polygon.contains(point)

    return mask


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


def getArrayVertices(ARR):
    top_left = ARR[0, 0]
    top_right = ARR[0, ARR.shape[1] - 1]
    bottom_left = ARR[ARR.shape[0] - 1, 0]
    bottom_right = ARR[ARR.shape[0] - 1, ARR.shape[1] - 1]
    return [top_left, top_right, bottom_left, bottom_right]


def get_latlon_from_griddesc(griddesc_path, gridname):
    """
    使用 PseudoNetCDF 自动解析 GRIDDESC 并生成 LAT/LON
    ✅ 无需 camxfiles.griddesc
    ✅ 自动读取投影参数
    ✅ 自动生成二维经纬度
    """
    import PseudoNetCDF
    grd = PseudoNetCDF.pncopen(griddesc_path, format='griddesc')
    print(grd)
    # grid = grd.variables[gridname]       # 直接访问对应网格
    

    lat = grd.variables['latitude'][:]       # shape (nrows, ncols)
    lon = grd.variables['longitude'][:]

    return lat.astype('float32'), lon.astype('float32')

def calculate_average_wind_direction(wind_directions):
    """
    计算平均风向

    :param wind_speeds: 风速列表
    :param wind_directions: 风向列表（单位：度）
    :return: 平均风向（单位：度）
    """
    WD = wind_directions * np.pi / 180
    cosValue = np.nanmean(np.cos(WD))
    sinValue = np.nanmean(np.sin(WD))
    getWD = (np.arctan2(sinValue, cosValue) / np.pi * 180)
    if getWD < 0:
        FinalWD = getWD + 360
    else:
        FinalWD = getWD
    return FinalWD

def calculate_lambert_bounds(lat, lon, resolution_km, standard_parallels):
    """
    在某经纬度中心，计算前后总共resolution_km*2的方格经纬度范围bounds，坐标系为lambert
    可以用于计算CMAQgrid的每个格点范围
    :param lat:
    :param lon:
    :param resolution_km:
    :param standard_parallels:
    :return:
    """
    # 创建Lambert正形投影对象
    proj = Proj(proj='lcc', lat_1=standard_parallels[0], lat_2=standard_parallels[1], lon_0=standard_parallels[2])

    # 转换中心经纬度为投影坐标
    x_center, y_center = proj(lon, lat)

    # 计算x和y的变化量 (resolution)
    # 先将分辨率转换为投影坐标系的单位，假设分辨率为公里
    x_delta = resolution_km * 1000 * np.cos(lat * np.pi / 180)  # 根据纬度调整
    y_delta = resolution_km * 1000  # 直接使用公里(注意此处可能需要调整根据具体投影的特性)

    # 计算坐标范围
    x_min = x_center - x_delta
    x_max = x_center + x_delta
    y_min = y_center - y_delta
    y_max = y_center + y_delta

    # 将坐标范围转换回经纬度
    lon_min, lat_min = proj(x_min, y_min, inverse=True)
    lon_max, lat_max = proj(x_max, y_max, inverse=True)

    return (lat_min, lat_max, lon_min, lon_max)

def metstationFilesTo2CSV(file, outfile, year):
    """
    将气象站点数据转换为CSV,同时补充缺失数据,成逐小时的数据格式,缺失值用-9999替换
    可根据isdsite_metdata_select中的有效站点目录来进行
    :param file:
    :param outfile:
    :param year: 数据年份
    :return:
    """
    Sfile = open(file, "r")
    res = len(Sfile.readlines())  # 文件行数
    Sfile.close()


    labels = ['年份', '月', '日', '时', '温度', '露点温度', '海洋压强', '风向', '风速', '云量',
              '液体沉淀深度尺寸-1小时', '液体沉淀深度尺寸-6小时']
    result_csv_data = pd.DataFrame(columns=labels)  # 创建记录结果的csv
    date_list_hour = generate_dates(int(year))

        # 读取气象站文件并写入
    Sfile = open(file, "r")
    file_data = []
    for i in range(0, res):  # 站点文件每行处理
        a = Sfile.readline()
        a = a.split(" ")
        b = []

        for k in a:  # 获得气象站每行数据
            if k != "":
                b.append(k)
        b[-1] = b[-1].replace("\n", "")
        file_data.append(b)  # 把所有气象数据每行存入列表

        # for j in range(0, len(b)):  # 写入每行数据
        #     sheet.write(1 + i, j, float(b[j])) # float()可以将日期'07'转为7

    for x in tqdm(date_list_hour):  # tqdm用于for中可以显示进度条
        # 年 月 日 时 0 1 2 3
        target_row = date_list_hour.index(x) + 1
        result_csv_data.at[target_row, '年份'] = float(x[0:4])
        result_csv_data.at[target_row, '月'] = float(x[4:6])
        result_csv_data.at[target_row, '日'] = float(x[6:8])
        result_csv_data.at[target_row, '时'] = float(x[8:10])
        flag = 0
        for m in file_data:
            if m[0] == x[0:4] and m[1] == x[4:6] and m[2] == x[6:8] and m[3] == x[8:10]:  # 遍历气象站数据，找到写入的这天的数据
                result_csv_data.at[target_row, '温度'] = float(m[4])
                result_csv_data.at[target_row, '露点温度'] = float(m[5])
                result_csv_data.at[target_row, '海洋压强'] = float(m[6])
                result_csv_data.at[target_row, '风向'] = float(m[7])
                result_csv_data.at[target_row, '风速'] = float(m[8])
                result_csv_data.at[target_row, '云量'] = float(m[9])
                result_csv_data.at[target_row, '液体沉淀深度尺寸-1小时'] = float(m[10])
                result_csv_data.at[target_row, '液体沉淀深度尺寸-6小时'] = float(m[11])
                flag = 1
        if flag == 0:  # 没有找到数据，输入np.nan 以前为-9999
            for s in range(4, 12):
                result_csv_data.loc[target_row, ['温度', '露点温度', '海洋压强', '风向', '风速', '云量',
              '液体沉淀深度尺寸-1小时', '液体沉淀深度尺寸-6小时']] = np.nan

    # 添加一个合并的datetime格式的列
    date_strings = result_csv_data[['年份', '月', '日', '时']].astype(int).astype(str).agg('-'.join, axis=1)
    result_csv_data['datetime'] = pd.to_datetime(date_strings, format='%Y-%m-%d-%H')
    result_csv_data.to_csv(outfile,encoding='utf-8')

def metstationDataGetfromCSV(csvfile, daterange,metdata):
    """
    从isd气象站数据的csvfile来获取某个daterange下的气象站数据类型metdata
    :param csvfile:
    :param daterange:
    :param metdata:
    :return:
    """
    metdata_dataframe = pd.read_csv(csvfile)
    # 读取后的datetime因为被保存成为了str，需要在这里再转一下
    metdata_dataframe['datetime'] = pd.to_datetime(
        metdata_dataframe.apply(lambda row: f"{int(row['年份'])}-{int(row['月'])}-{int(row['日'])} {int(row['时'])}:00",
                              axis=1)
    )
    rows_in_range = metdata_dataframe[
        (metdata_dataframe['datetime'] >= daterange[0]) & (metdata_dataframe['datetime'] <= daterange[-1])].index.tolist()
    return metdata_dataframe.loc[rows_in_range, metdata].tolist()

def isdsite_metdata_select(
    latmin, latmax, lonmin, lonmax,
    year='2020',
    metstation_files_dir="",
    metstation_infofile_dir="",
):
    """
    在NCDCisd site数据中，找到经纬度范围内数据有效的气象站点，返回站点信息字典
    :param latmin:
    :param latmax:
    :param lonmin:
    :param lonmax:
    :param year: 为str格式
    :param metstation_files_dir:
    :param metstation_infofile_dir:
    :return:
    """
    metdata_dir = metstation_files_dir
    metstations_info_file_dir = metstation_infofile_dir

    metstations_infofile = pd.read_csv(metstations_info_file_dir)
    metstations = {}
    final_filtered_rows = metstations_infofile[ # 找到模拟范围内的站点
        (metstations_infofile['Lon'] > lonmin) & (metstations_infofile['Lon'] < lonmax) &
        (metstations_infofile['Lat'] > latmin) & (metstations_infofile['Lat'] < latmax)
        ]
    for row in final_filtered_rows.index.tolist():
        metstations.update({metstations_infofile.at[row, 'name']:
                                [float(metstations_infofile.at[row, 'Lon']),
                                 float(metstations_infofile.at[row, 'Lat']),
                                 f'{metstations_infofile.at[row, "stationid"]}0']}) # 站点信息文件的id相较于数据文件少了个0
        # 数据格式：站点名：经度 纬度 站点代号
    metdatas = os.listdir(metdata_dir)
    metdatas_year = [x for x in metdatas if x.split('-')[2] == year] # 筛选目标年份
    metdatas_id = [x.split('-')[0] for x in metdatas_year] # 筛选有数据的站点
    metstations_keytodel = []
    for metstation in metstations:    #气象站数据存在且有效，否则将筛选出的站点移除 数据文件大于1000字节的认为是有有效数据的气象站点文件
        if (str(metstations[metstation][2]) not in metdatas_id):
            metstations_keytodel.append(metstation)
            continue
        elif os.stat(metdata_dir + f'{metstations[metstation][2]}-99999-{year}').st_size < 1000:
            metstations_keytodel.append(metstation)
        else:
            pass
    for m in metstations_keytodel:
        del metstations[m]


    return metstations

def getAirStationsFromLatLon(uplat, downlat, leftlon, rightlon,airStation_infofile_dir):
    """
    从站点信息csv文件中，返回在某一个矩形经纬度范围内的站点信息，格式为验证代码中的字典格式
    :param airStation_infofile_dir: 站点位置信息csv所在目录
    :param uplat:
    :param downlat:
    :param leftlon:
    :param rightlon:
    :return: airStations: 空气质量站点信息
    """
    # airStation_infofile_dir = r"F:\全国空气质量\站点列表\站点列表-2022.02.13起.csv"
    airStation_infofile = pd.read_csv(airStation_infofile_dir)
    # sheet = airStation_infofile.sheet_by_index(0)
    airStations = {}

    for index, row in airStation_infofile.iterrows():
        stationinfo = row.values.tolist()
        # print(stationinfo[3])
        if stationinfo[3] != '' or stationinfo[3] != '-':
            if stationinfo[3] == '-': continue
            if leftlon < float(stationinfo[3]) < rightlon and downlat < float(stationinfo[4]) < uplat:
                airStations.update({stationinfo[1]: [float(stationinfo[3]), float(stationinfo[4]), stationinfo[0],stationinfo[2]]})
    # for line in range(1, sheet.nrows):
    #     stationinfo = sheet.row_values(line, 0, 5)
    #     if stationinfo[3] != '':
    #         if leftlon < float(stationinfo[3]) < rightlon and downlat < float(stationinfo[4]) < uplat:
    #             airStations.update({stationinfo[1]: [float(stationinfo[3]), float(stationinfo[4]), stationinfo[0],stationinfo[2]]})
    #                                     # 数据格式：站点名：经度 纬度 站点代号 所在城市
    return airStations




def getAirStationsFromShape(shapefile_dir,airStation_infofile_dir):
    import geopandas as gpd
    from shapely.geometry import Point
    """
    从站点信息csv文件中，返回在某一个闭合shape文件范围(例如行政区划)内的站点信息，格式为验证代码中的字典格式
    :param airStation_infofile_dir: 站点位置信息csv所在目录
    :param shapefile_dir: shape文件所在路径
    :return: airStations: 空气质量站点信息
    """
    airStation_infofile = pd.read_csv(airStation_infofile_dir)
    airStations = {}

    shape_polygon_ = gpd.read_file(shapefile_dir)
    shape_polygon = shape_polygon_.to_crs(epsg=4326)
    
    print(shape_polygon)
    for index, row in airStation_infofile.iterrows():
        stationinfo = row.values.tolist()
        if stationinfo[3] != '' or stationinfo[3] != '-':
            if stationinfo[3] == '-': continue
            if shape_polygon.geometry[0].contains(Point(float(stationinfo[3]), float(stationinfo[4]))):
                airStations.update({stationinfo[1]: [float(stationinfo[3]), float(stationinfo[4]), stationinfo[0],stationinfo[2]]})

    return airStations


def getIsdMetsite_Stations_FromShape(
    shapefile_dir,
    year='2020',
    metstation_files_dir="",
    metstation_infofile_dir="",
):
    import geopandas as gpd
    from shapely.geometry import Point
    """
    在NCDCisd site数据中，找到shape范围内数据有效的气象站点，返回站点信息字典
    :param shapefile_dir: 目标shape范围
    :param year: 为str格式
    :param metstation_files_dir: isd原始文件地址 metstation_files_dir='E:\气象站数据\china_isdsite_metdata\\',
    :param metstation_infofile_dir: 站点列表信息csv metstation_infofile_dir="E:\气象站数据\全国气象站位置信息\站点列表_原始数据.csv",
    :return:
    """
    metdata_dir = metstation_files_dir
    metstations_info_file_dir = metstation_infofile_dir

    # shape_polygon_ = gpd.read_file(shapefile_dir)
    # shape_polygon = shape_polygon_.to_crs(epsg=4326)

    metstations_infofile = pd.read_csv(metstations_info_file_dir)
    metstations = {}
    # print(metstations_infofile)

    geometry = [Point(xy) for xy in zip(metstations_infofile['Lon'], metstations_infofile['Lat'])]

    gdf = gpd.GeoDataFrame(metstations_infofile, geometry=geometry)
    shapefile_path = shapefile_dir  # 请替换为您的 shapefile 文件路径
    gdf_shape_ = gpd.read_file(shapefile_path)
    gdf_shape = gdf_shape_.to_crs(epsg=4326)
    # print(111,gdf_shape)

    # 第四步：检查 CRS（坐标参考系）
    # 确保 CSV 和 shapefile 的 CRS 一致
    # print(gdf.crs)  # 查看 GeoDataFrame 的 CRS
    # print(gdf_shape.crs)  # 查看 shapefile 的 CRS

    # 如果需要，您可以将一个 GeoDataFrame 转换为另一个 CRS
    # gdf = gdf.to_crs(gdf_shape.crs)

    # 第五步：空间连接筛选数据
    # 使用 spatial join 选择在形状内的点
    filtered_gdf = gpd.sjoin(gdf, gdf_shape, how='inner', predicate='within')
    # print(filtered_gdf)

    # # 第六步：保存结果
    # output_file_path = 'filtered_coordinates.csv'  # 输出文件路径
    # filtered_gdf.to_csv(output_file_path, index=False)
    # print(filtered_gdf)

    for row in filtered_gdf.index.tolist():
        metstations.update({metstations_infofile.at[row, 'name']:
                                [float(metstations_infofile.at[row, 'Lon']),
                                 float(metstations_infofile.at[row, 'Lat']),
                                 f'{metstations_infofile.at[row, "stationid"]}0']}) # 站点信息文件的id相较于数据文件少了个0
        print({metstations_infofile.at[row, 'name']:
                                [float(metstations_infofile.at[row, 'Lon']),
                                 float(metstations_infofile.at[row, 'Lat']),
                                 f'{metstations_infofile.at[row, "stationid"]}0']})
        # 数据格式：站点名：经度 纬度 站点代号
    metdatas = os.listdir(metdata_dir)
    metdatas_year = [x for x in metdatas if x.split('-')[2] == str(year)] # 筛选目标年份
    metdatas_id = [x.split('-')[0] for x in metdatas_year] # 筛选有数据的站点
    metstations_keytodel = []
    for metstation in metstations:    #气象站数据存在且有效，否则将筛选出的站点移除 数据文件大于1000字节的认为是有有效数据的气象站点文件
        print(metstation)
        if (str(metstations[metstation][2]) not in metdatas_id):
            metstations_keytodel.append(metstation)
            continue
        elif os.stat(metdata_dir + f'{metstations[metstation][2]}-99999-{year}').st_size < 1000:
            metstations_keytodel.append(metstation)
        else:
            pass
    for m in metstations_keytodel:
        del metstations[m]

    return metstations


def getWRFvarsCombined(WRFout_files_dir,metvar):
    """
    返回一个目录下所有WRFout文件的某个变量的conbine后的数组
    :param WRFout_files_dir:
    :param metvar:
    :return:
    """
    deg = 180.0 / np.pi
    rad = np.pi / 180.0

    WRF_file_list_pr = os.listdir(WRFout_files_dir)  # 获得WRF文件列表
    WRF_file_list = []
    WRF_file_list_time = {}
    for i in WRF_file_list_pr:
        if i[0:9] == 'wrfout_d0':  #
            WRF_file_list.append(i)
    # print(WRF_file_list)
    daycount = len(WRF_file_list)
    for i in WRF_file_list:
        time = int(i[19:21])  # 文件名称天数的位置
        WRF_file_list_time.update({i: time})
    WRF_file_list_time = sorted(WRF_file_list_time.items(), key=lambda x: x[1])  # 字典按时间排序
    WRF_file_list_time = dict(WRF_file_list_time)
    # print(WRF_file_list_time.keys())
    # print(WRF_file_list_time)

    WRFoutf = nc.Dataset(WRFout_files_dir + WRF_file_list[0])  # 打开WRF输出的HDF格式文件
    ROW = np.array(WRFoutf.variables["XLAT"]).shape[1]  # 获得数据格式shape
    COL = np.array(WRFoutf.variables["XLAT"]).shape[2]
    data_shape = np.array(WRFoutf.variables["XLAT"]).shape  # 获得数据shape来存放空变量
    metvar_now = np.zeros(data_shape, dtype=np.float64)
    metvar_next = np.zeros(data_shape, dtype=np.float64)
    WRFoutf.close()

    WRF_file_list_time_list = list(WRF_file_list_time.keys())  # 将WRF风场数据合并为1个文件，用来求8h平均
    # print(WRF_file_list_time_list)
    for n in WRF_file_list_time_list:
        n_num = WRF_file_list_time_list.index(n)
        # print(n_num)
        if n_num + 1 >= len(WRF_file_list_time_list):
            metvar_combine = metvar_now
            break
        n2 = WRF_file_list_time_list[n_num + 1]
        WRFoutf = nc.Dataset(WRFout_files_dir + n)  # 打开WRF输出的HDF格式文件
        WRFoutf_next = nc.Dataset(WRFout_files_dir + n2)  # 打开WRF输出的HDF格式文件

        metvar_next = np.array(WRFoutf_next.variables[metvar])  # 合并的WRF变量
        metvar_now = np.concatenate((metvar_now, metvar_next), axis=0)  # 合并

        WRFoutf.close()

    return metvar_combine


def getNCvarsTimeCombined(NC_files_glob,varname,isemis=False):
    """
    将一个目录下的相同数据格式类型的nc文件的varname变量按照时间进行combine
    返回最终combine后的varname数组
    :param NC_files_glob:glob.glob(路径)的返回值
    :param varname:
    :return:
    """

    # NC_file_list = os.listdir(NC_files_glob)  # 获得NC文件列表
    NC_file_list = sorted(NC_files_glob)  # 字典按时间排序

    NCfile = nc.Dataset(NC_file_list[0])  # 打开NC输出的HDF格式文件
    data_shape = np.array(NCfile.variables[varname]).shape  # 获得数据shape来存放空变量
    if isemis == True:
        var_now = np.array(NCfile.variables[varname])[0:24,:,:,:]
    else:
        var_now = np.array(NCfile.variables[varname])
    NCfile.close()

    for n in NC_file_list:
        n_num = NC_file_list.index(n)
        # print(n_num)
        if n_num + 1 >= len(NC_file_list):
            var_combine = var_now
            break
        n2 = NC_file_list[n_num + 1]
        NCfile = nc.Dataset(n)  # 打开WRF输出的HDF格式文件
        NCfile_next = nc.Dataset(n2)  # 打开WRF输出的HDF格式文件

        if isemis == True:
            var_next = np.array(NCfile_next.variables[varname])[0:24, :, :, :]
        else:
            var_next = np.array(NCfile_next.variables[varname])  # 合并的WRF变量
        var_now = np.concatenate((var_now, var_next), axis=0)  # 合并

        NCfile.close()

    return var_combine


def combine_cmaq_output(folder,outtype, varname="O3"):
        import PseudoNetCDF
        import glob
        """
        将指定目录下的 CMAQ CCTM_ACONC* 文件按时间合并为一个四维数组。
        参数:
            folder : str
                存放 CCTM_ACONC 文件的目录
            outtype:
                CCTM文件的类型，如CCTM_ACONC CCTM_IRR 
            varname : str
                要提取的变量名（例如 "O3", "NO2", "PM25_TOT"）
        返回:
            arr : numpy.ndarray
                合并后的数组，形状为 (T, LAY, ROW, COL)
            time_flags : numpy.ndarray
                对应的 CMAQ 时间标志 (T, 2)，单位为 [YYYYDDD, HHMMSS]
            info : dict
                文件维度与变量元信息
        """
        
        def safe_pncopen(f):
            """自动尝试多种格式读取 CMAQ 文件"""
            for fmt in ["ioapi","netcdf", "netcdf4"]:
                try:
                    ds = PseudoNetCDF.pncopen(f, format=fmt)
                    print(f"  ✅ 成功以格式 {fmt} 打开: {os.path.basename(f)}")
                    return ds
                except Exception as e:
                    print(f"  ⚠️ {fmt} 打开失败: {e}")
                    pass
            raise IOError(f"❌ 无法识别文件格式: {f}")


        # 找到所有 CCTM_ACONC 文件
        files = sorted(glob.glob(os.path.join(folder, f"{outtype}*.nc")))
        if not files:
            raise FileNotFoundError(f"目录中未找到 {outtype}*.nc 文件: {folder}")

        data_list = []
        time_list = []

        for i, f in enumerate(files):
            print(f"[{i+1}/{len(files)}] 读取 {os.path.basename(f)} ...")
            ds = safe_pncopen(f)

            if varname not in ds.variables:
                raise KeyError(f"变量 {varname} 不存在于文件 {f} 中，包含变量: {list(ds.variables.keys())}")

            v = ds.variables[varname][:]   # shape (TSTEP, LAY, ROW, COL)
            tflag = ds.variables["TFLAG"][:, 0, :]  # (TSTEP, 2)
            data_list.append(v)
            time_list.append(tflag)

            ds.close()

        # 沿时间拼接
        arr = np.concatenate(data_list, axis=0)
        time_flags = np.concatenate(time_list, axis=0)

        info = {
            "shape": arr.shape,
            "files": files,
            "varname": varname,
            "description": f"{varname} combined array (T,LAY,ROW,COL)"
        }

        # print("✅ 合并完成：")
        # print(f"   变量 {varname} 形状 = {arr.shape}")
        # print(f"   时间步数 T = {arr.shape[0]}")

        return arr







