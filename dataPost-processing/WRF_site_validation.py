import time
import matplotlib
import netCDF4 as nc
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import pandas as pd
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
    return ALL1 / ALL2


def calcuNME(modData, obsData):
    N = len(modData) if len(modData) < len(obsData) else len(obsData)  # 不过限度
    ALL1 = 0
    ALL2 = 0
    for i in range(0, N):
        if np.isnan(obsData[i]) == True:  # 判断nan
            continue
        ALL1 += abs(modData[i] - obsData[i])
        ALL2 += obsData[i]
    return ALL1 / ALL2


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

def generate_date_list_withhour(start_date_str,n):
    start_date_str += '-00'
    start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d-%H')
    date_list = []
    for i in range(n+1):  # 包括起始时间，所以是 745
        current_date = start_date + datetime.timedelta(hours=i)
        date_list.append(current_date)
    return date_list

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

def metstationFilesTo2CSV(file, outfile, year):
    """
    将气象站点数据转换为excel,同时补充缺失数据,成逐小时的数据格式,缺失值用-9999替换
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
        if flag == 0:  # 没有找到数据，输入-9999
            for s in range(4, 12):
                result_csv_data.loc[target_row, ['温度', '露点温度', '海洋压强', '风向', '风速', '云量',
              '液体沉淀深度尺寸-1小时', '液体沉淀深度尺寸-6小时']] = -9999

    result_csv_data.to_csv(outfile,encoding='utf-8')

def isdsite_metdata_select(
    latmin, latmax, lonmin, lonmax,
    year='2020',
    metstation_files_dir="",
    metstation_infofile_dir="",
):
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
                                 metstations_infofile.at[row, 'stationid']]})
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

    # 没在站点位置文件的统计中，但属于四川盆地且文件中存在的两个气象站，手动添加：
    metstations.update({'成都':['104.02', '30.67',562940]})
    # metstations.update({'南充': ['106.08', '30.78', 574110]})

    return metstations


def WRF_site_validation(
    start_date='YYYY-MM-DD',
    daycount=5,  #
    wrfout_files_dir='',
    target_domain='',
    metstation_files_dir = "",
    metstation_infofile_dir = "",
    result_pic_types=[],
    out_dir="",  #
    result_csv_name="",  #
    suffix="",
):
    """
    使用全球isdsite站点逐小时气象数据来验证WRF模拟气象场

    :param start_date: 验证的开始时期，确定观测值数据获取
    :param daycount: 验证持续天数
    :param wrfout_files_dir: 模拟的所有wrfout文件所在的目录
    :param target_domain: wrfout的domain，格式为'd01'，'d02'等
    :param metstation_files_dir: isdsite气象数据文件所在的目录
    :param metstation_infofile_dir: 全国isd气象站点位置信息csv文件所在目录
    :param result_pic_types: 输出哪些类型的结果图，'line'：双折线图，'scatter':散点回归线图
    :param out_dir: 验证结果输出的文件夹
    :param result_csv_name: 包含验证参数计算结果的输出csv文件名称
    :param suffix: 验证过程后缀，用于区分
    """
    deg = 180.0 / np.pi
    rad = np.pi / 180.0
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    if os.path.exists(out_dir) is False: os.mkdir(out_dir)
    input_year = start_date.split('-')[0]

    wrfout_files = os.listdir(wrfout_files_dir) # 获得wrfout文件列表
    wrfout_init = nc.Dataset(wrfout_files_dir+wrfout_files[0], "r")  #打开所有wrfout其中之一来获取模拟区域经纬度范围
    var_lon = np.array(wrfout_init.variables['XLONG'][:][0])
    var_lat = np.array(wrfout_init.variables['XLAT'][:][0])
    lonmin, latmax, lonmax, latmin = (var_lon.min(), var_lat.max(),
                                      var_lon.max(), var_lat.min())
    metstations = isdsite_metdata_select(
        latmin,latmax,lonmin,lonmax,
        year=input_year,
        metstation_files_dir=metstation_files_dir,
        metstation_infofile_dir=metstation_infofile_dir,
    )

    # 转换气象站数据格式为易处理的csv
    for metstation in tqdm(metstations,desc="转换气象站数据格式为易处理的csv"):
        metinfo = metstations[metstation]
        filename = f'{metinfo[2]}-99999-{str(input_year)}'
        outfilename = f'{metstation}-{str(input_year)}'
        metstation_file_dir = f'{metstation_files_dir}{filename}'
        metstationFilesTo2CSV(metstation_file_dir,f'{out_dir}{outfilename}.csv',input_year)

    labels = ['站点', 'MAE','RSME','MB', 'R', 'NMB', 'NME', 'MFE', 'MFB', '站点经度', '站点纬度']
    para_data_T2 = pd.DataFrame(columns=labels)  # 验证参数的pd dataframe，导出时不导出csv，而是导出具有多个sheet的xlsx
    para_data_RH = pd.DataFrame(columns=labels)
    para_data_WS = pd.DataFrame(columns=labels)
    para_data_WD = pd.DataFrame(columns=labels)

    print('气象验证进行中...')
    csv_row = 1 # csv非表头的第1行开始写入
    for station in tqdm(metstations,desc='输出每个站点的验证结果表和图'):
        stname = station
        station_lat = float(metstations[station][1])
        station_lon = float(metstations[station][0])
        WRF_file_list_pr = os.listdir(wrfout_files_dir)  # 获得WRF文件列表
        WRF_file_list = []
        WRF_file_list_time = {}
        for i in WRF_file_list_pr:
            if i[0:10] == f'wrfout_{target_domain}':
                WRF_file_list.append(i)
        for i in WRF_file_list:
            time = int(i[19:21])  # 文件名称天数的位置
            WRF_file_list_time.update({i: time})
        WRF_file_list_time = sorted(WRF_file_list_time.items(), key=lambda x: x[1])  # 字典按时间排序
        WRF_file_list_time = dict(WRF_file_list_time)
        WRFoutf = nc.Dataset(wrfout_files_dir + WRF_file_list[0], "r")  # 打开WRF输出的HDF格式文件
        XLAT = np.array(WRFoutf.variables["XLAT"][0,:,:])  # 获得经纬度数据集，并通过numpy转化为数组
        XLONG = np.array(WRFoutf.variables["XLONG"][0,:,:])
        nearpos = getNearestPos(station_lat, station_lon, XLAT, XLONG)
        nearlat = nearpos[0]
        nearlon = nearpos[1]
        WRFoutf.close()
        T2_WRF, RH_WRF, WS_WRF, WD_WRF = [], [], [], []
        for i in WRF_file_list_time.keys():
            WRFoutf = nc.Dataset(wrfout_files_dir + i, "r")  # 打开WRF输出的HDF格式文件
            T2_K = np.array(WRFoutf.variables["T2"])  # 从WRFout得到地面2m温度 单位为K
            P = np.array(WRFoutf.variables["PSFC"])  # 气压 单位为Pa
            SH = np.array(WRFoutf.variables["Q2"])  # 比湿
            WS_V = np.array(WRFoutf.variables["V10"])  # 风速经向分量
            WS_U = np.array(WRFoutf.variables["U10"])  # 风速纬向分量
            T2 = T2_K - 273.15  # K转化为摄氏度
            RH = 0.236 * P * SH * np.exp((17.67 * T2) / (T2_K - 29.65)) ** (-1)  # 相对湿度
            WS = np.sqrt(WS_V[:] ** 2 + WS_U[:] ** 2)  # 分速度求和速度
            WD = 180.0 + np.arctan2(WS_U, WS_V) * deg  # 计算风向
            if list(WRF_file_list_time.keys()).index(i) == 0:  # 第一个数据就创建数组
                T2_WRF = list(T2[:, nearlat, nearlon])  # WRF中站点温度值
                RH_WRF = list(RH[:, nearlat, nearlon])  # WRF中得到相对湿度
                WS_WRF = list(WS[:, nearlat, nearlon])  # WRF中得到风速
                WD_WRF = list(WD[:, nearlat, nearlon])  # WRF中得到风向
            else:
                T2_WRF += list(T2[:, nearlat, nearlon])
                RH_WRF += list(RH[:, nearlat, nearlon])
                WS_WRF += list(WS[:, nearlat, nearlon])
                WD_WRF += list(WD[:, nearlat, nearlon])
            WRFoutf.close()

            # print('温度：   ', T2_station)

        meteoStation_csv_dir = f"{out_dir}{stname}-{str(input_year)}.csv"
        meteoStation_csv = pd.read_csv(meteoStation_csv_dir)  # 修复了8月10日气象站数据有确实的问题，导致了绘图的错误
        target_month = start_date.split('-')[1]
        target_day = start_date.split('-')[2]
        target_meteoStationData = {}  # WRF模拟的从开始时间到结束的站点数据
        target_row = meteoStation_csv[(meteoStation_csv['年份'] == float(input_year)) &
                                      (meteoStation_csv['月'] == float(target_month)) &
                                      (meteoStation_csv['日'] == float(target_day))].index[0]
        print(target_row)
        target_T2 = np.array(meteoStation_csv['温度'].iloc[target_row:target_row + daycount*24])
        target_T2[target_T2 == -9999] = np.nan
        target_T2 /= 10 # 处理scale
        target_meteoStationData.update({'温度': target_T2})
        target_DPT = np.array(meteoStation_csv['露点温度'].iloc[target_row:target_row + daycount*24])
        target_DPT[target_DPT == -9999] = np.nan
        target_DPT /= 10 # 处理scale
        target_meteoStationData.update({'露点温度': target_DPT})
        target_WD = np.array(meteoStation_csv['风向'].iloc[target_row:target_row + daycount * 24])
        target_WD[target_WD == -9999] = np.nan
        target_meteoStationData.update({'风向': target_WD})
        target_WS = np.array(meteoStation_csv['风速'].iloc[target_row:target_row + daycount*24])
        target_WS[target_WS == -9999] = np.nan
        target_WS /= 10 # 处理scale
        target_meteoStationData.update({'风速': target_WS})

        # print(target_meteoStationData['温度'])
        # print(target_meteoStationData['露点温度'])
        # print(target_meteoStationData['风速'])
        # print(target_meteoStationData['风向'])

        # 气象站数据计算相对湿度
        T2_st = target_meteoStationData['温度']
        DewT_st = target_meteoStationData['露点温度']
        E = 6.112 * np.exp((17.67 * T2_st) / (T2_st + 243.5))
        e = 6.112 * np.exp((17.67 * DewT_st) / (DewT_st + 243.5))
        RH_st = e / E * 100

        # 空值处理，不在合理范围内的值为空值
        T2_st[T2_st <= -100] = None
        DewT_st[DewT_st <= -1000] = None
        RH_st[RH_st == 100] = None
        WS_st = np.array(target_meteoStationData['风速'])
        WS_st[WS_st <= -10] = None
        WD_st = np.array(target_meteoStationData['风向'])
        WD_st[WD_st <= -10] = None

        # print('相对湿度，观测值:   ', RH_st)
        #
        # print('list长度：   ',len(T2_station))

        # k.append(pollution1) # 气象站字典的列表里再加一个污染物字典

        fig2 = plt.figure(figsize=(5, 2), dpi=200)

        final_datas = {'T2':[T2_WRF,T2_st],'RH':[RH_WRF,RH_st],'WS':[WS_WRF,WS_st],'WD':[WD_WRF,WD_st]}

        # 绘制不同类型的结果图
        xdate = generate_date_list_withhour(start_date, daycount * 24)
        xdate.pop()  # 会多一个h
        for data in final_datas:
            if 'line' in result_pic_types:
                if os.path.exists(f'{out_dir}pics_line_{data}\\') is False: os.mkdir(f'{out_dir}pics_line_{data}\\')
                fig2 = plt.figure(figsize=(5, 2), dpi=200)
                ax2 = fig2.add_subplot(111)
                line1, = ax2.plot(xdate, final_datas[data][1], linewidth=0.5, label='Obs',
                                  color='#258080')  # 要用legend画图例，这里必须,=
                line2, = ax2.plot(xdate, final_datas[data][0], linewidth=0.5, label='Sim', color='red')
                ax2.set_ylabel('μg/m$^{3}$', fontsize=10)
                ax2.set_xlabel('Date', fontsize=10)
                plt.title(f'{data} validation at site {stname}', fontsize=10)
                plt.xticks(fontsize=5)  # xticks必须在这个位置才生效
                plt.yticks(fontsize=5)
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))  # 设置不显示年份
                plt.legend((line1, line2), ('Obs', 'Sim'), loc='upper right', frameon=False, framealpha=0.5,
                           fontsize=5)
                plt.savefig(f'{out_dir}pics_line_{data}\\{stname}_{data}_{suffix}.png')
                plt.close()
            if 'scatter' in result_pic_types:
                if os.path.exists(f'{out_dir}pics_scatter_{data}\\') is False: os.mkdir(f'{out_dir}pics_scatter_{data}\\')
                plt.figure(figsize=(6, 6), dpi=200)
                observed = np.array(final_datas[data][1])
                simulated = np.array(final_datas[data][0])  # 转为数组后才能进行掩码处理
                # 去除 NaN 值
                mask = ~np.isnan(observed)  # 创建一个掩码，选择观测值不是 NaN 的索引
                observed_clean = observed[mask]
                simulated_clean = simulated[mask]
                xy = np.vstack([observed_clean, simulated_clean])
                z = gaussian_kde(xy)(xy)
                scatter = plt.scatter(observed_clean, simulated_clean, c=z, cmap='viridis', alpha=0.6)
                model = LinearRegression()
                model.fit(observed_clean.reshape(-1, 1), simulated_clean)
                x_fit = np.linspace(min(observed_clean), max(observed_clean), 100).reshape(-1, 1)
                y_fit = model.predict(x_fit)
                slope = model.coef_[0]
                intercept = model.intercept_
                equation = f'y = {slope:.2f}x + {intercept:.2f}'
                # plt.text(0.9 * max(observed_clean), 0.9 * max(simulated_clean), equation, fontsize=12, color='red',
                #          ha='center', transform=plt.gca().transAxes)
                plt.plot(x_fit, y_fit, color='red', linewidth=2, label=equation)
                # cbar = plt.colorbar(scatter)
                # cbar.set_label('scatter density')
                plt.xlabel('Obs')
                plt.ylabel('Mod')
                plt.title(f'Scatter and regression line at site {stname}')
                plt.legend()
                # plt.grid()

                plt.savefig(f'{out_dir}pics_scatter_{data}\\{stname}_{data}_{suffix}.png')
                plt.close()
            """
            计算精度系数
            """
            para_data_target = locals()[f'para_data_{data}'] # 以locals()通过变量获取变量 通过字符串
            substanceMAE = calcuMAE(final_datas[data][0], final_datas[data][1])
            substanceRSME = calcuRSME(final_datas[data][0], final_datas[data][1])
            substanceMB = calcuMB(final_datas[data][0], final_datas[data][1])
            substanceR = calcuR(final_datas[data][0], final_datas[data][1])
            substanceNMB = calcuNMB(final_datas[data][0], final_datas[data][1])
            substanceNME = calcuNME(final_datas[data][0], final_datas[data][1])
            substanceMFE = calcuMFE(final_datas[data][0], final_datas[data][1])
            substanceMFB = calcuMFB(final_datas[data][0], final_datas[data][1])
            para_data_target.at[csv_row, '站点'] = stname
            para_data_target.at[csv_row, 'MAE'] = substanceMAE
            para_data_target.at[csv_row, 'R'] = substanceR
            para_data_target.at[csv_row, 'NMB'] = substanceNMB
            para_data_target.at[csv_row, 'NME'] = substanceNME
            para_data_target.at[csv_row, 'MFE'] = substanceMFE
            para_data_target.at[csv_row, 'MFB'] = substanceMFB
            para_data_target.at[csv_row, '站点经度'] = station_lon
            para_data_target.at[csv_row, '站点纬度'] = station_lat
            csv_row += 1

    with pd.ExcelWriter(f'{out_dir}{result_csv_name}_{suffix}.xlsx', engine='xlsxwriter') as writer:
        para_data_T2.to_excel(writer, sheet_name='T2', index=False)  # 将 df1 写入 Sheet1
        para_data_RH.to_excel(writer, sheet_name='RH', index=False)
        para_data_WS.to_excel(writer, sheet_name='WS', index=False)
        para_data_WD.to_excel(writer, sheet_name='WD', index=False)

if __name__ == '__main__':
    WRF_site_validation(
        start_date='2022-08-01',
        daycount=31,
        wrfout_files_dir=r'E:\CMAQdata_chengdu202208\WRFd03_2\\',
        target_domain='d03',
        metstation_files_dir='E:\气象站数据\china_isdsite_metdata\\',
        metstation_infofile_dir="E:\气象站数据\全国气象站位置信息\站点列表_原始数据.csv",
        result_pic_types=['line', 'scatter'],
        out_dir=r'E:\Emission_update\\test_wrf_validation\\',
        result_csv_name='validationPara',
        suffix='CD202208'
    )
    pass