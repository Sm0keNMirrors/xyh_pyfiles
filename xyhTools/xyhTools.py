
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