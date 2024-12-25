"""
Author: Yaohan Xian
GitHub: https://github.com/Sm0keNMirrors
Last update: 2024年12月24日
"""
import os

import netCDF4 as nc
import numpy as np
import pandas as pd


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

def CMAQ_calcuISAMcontribute(
    combinef_dir = "", # CMAQISAM 的combine后文件位置，暂时只支持单个combine文件
    GRIDCRO2D_dir="",
    result_dir = "", # 贡献率计算结果输出位置，包括csv和结果图，当前版本仅支持csv
    result_suffix = "", # 结果名称后缀 用于区分
    start_date = "YYYY-MM-DD",
    Regions_locations = {}, # 被各类源影响区域名称及坐标字典，以位置点来代表区域，每个区域可多个点，格式为'成都': [[104.07, 30.67]]
    tags_labels = {}, # 每个tag的中文名，格式为：{'CDEM':'成都区域源'} 必须以完全对应的tag名
    target_vars = [], # 需要计算的被标记源，每个物质输出一个结果，必须是被标记的物质的不带tag名称
):
    # 获取ISAM源tag
    ISAM_tags = ['_ALL'] # 最初
    combinef = nc.Dataset(combinef_dir)  # 先打开一个文件，获取数据信息
    all_vars_pre = list(combinef.variables.keys())
    for x in all_vars_pre:  # 从所有变量中找到需要combine的PM组分
        subname = x.split('_')[0]  # 去掉ISAMtag
        if len(x.split('_')) == 2:
            tag = x.split('_')[1]
            if '_' + tag not in ISAM_tags:
                ISAM_tags.append('_' + tag)
    init_data = np.array(combinef.variables[target_vars[0]+ISAM_tags[1]]) #直接
    list_starttime = 16
    daynum = int(init_data.shape[0]/24)-1 # UTC导致少一天
    list_endtime = 16 + 24 * daynum
    

    Emisson_label = ['BCIC', 'Industrial', 'Resident', 'Transportation', 'Power', 'Biogenic']
    color_list = ['#505050', '#AAAAAA', '#DDDDDD', '#F5A1D5', '#BF7D4E', '#DAF7A6', '#6AB04C', '#1B4F72',
                  '#0074D9', '#39CCCC', '#3D9970', '#FF851B', '#FFDC00', '#FF4136', '#B10DC9', '#3D00B0', '#F211EE',
                  '#C70039', '#900C3F']
    Regions_count = len(Regions_locations.keys())  # 区域个数

    """
    各类标签数据字典初始化
    """
    for sub in target_vars:
        sub_data_lists_monthmeanPercent_Regions = {}
        sub_data_lists = {}
        sub_data_lists_daymean = {}
        sub_data_lists_daymeanPercent = {}
        sub_data_lists_daymeanPercent_draw = {}
        sub_data_lists_monthmean = {}
        sub_data_lists_monthmeanPercent = {}
        sub_data_lists_monthmeanPercent_draw = {}
        for x in ISAM_tags:  # 获得所有污染物列表
            sub_data_lists.update({sub + x: []})
            sub_data_lists_daymean.update({sub + x: []})
            sub_data_lists_daymeanPercent.update({sub + x : []})
            sub_data_lists_daymeanPercent_draw.update({sub + x: []})
            sub_data_lists_monthmean.update({sub + x: []})
            sub_data_lists_monthmeanPercent.update({sub + x: []})
            sub_data_lists_monthmeanPercent_draw.update({sub + x: []})
    
        for k in Regions_locations.values(): # 每个区域的排放来计算一次，每完成一次循环得到某个区域排放给其他所有区域的排放贡献，这个是半年平均值
            sub_data_lists_monthmeanPercent_Regions = {}
            sub_data_lists = {}
            sub_data_lists_daymean = {}
            sub_data_lists_daymeanPercent = {}
            sub_data_lists_daymeanPercent_draw = {}
            for x in ISAM_tags:  # 获得所有污染物列表
                sub_data_lists.update({sub + x: []})
                sub_data_lists_daymean.update({sub + x: []})
                sub_data_lists_daymeanPercent.update({sub + x: []})
                sub_data_lists_daymeanPercent_draw.update({sub + x: []})
    
            stname = list(Regions_locations.keys())[list(Regions_locations.values()).index(
                k)]  # 通过值k获取字典dic对应键的公式： list(dic.keys())[list(dic.values()).index(k)]
            Regions_num = list(Regions_locations.keys()).index(stname)  # 进行到第几个区域
            # print(stname)
            sp_count = len(k)  # 样点个数
            for sp in k:  # 对每个大区域的样点进行处理
                sp_num = k.index(sp)  # 第几个样点
                CMAQGRIDf = nc.Dataset(GRIDCRO2D_dir, "r")
                CMAQXLAT = np.array(CMAQGRIDf.variables['LAT'][:][0])
                CMAQXLONG = np.array(CMAQGRIDf.variables['LON'][:][0])
                nearpos = getNearestPos(sp[1], sp[0], CMAQXLAT, CMAQXLONG)  # 从WRF得到站点格子
                nearlat = nearpos[1]  # 与WRF获取最近点不同
                nearlon = nearpos[2]
    
                if sp_num == 0:
                    for n in sub_data_lists:  # 获得样点所有PM2.5tag数据，一个月的逐小时数据列表
                        if n == f"{sub}_ALL":  # 还未计算出ALL，先跳过
                            continue
                        if n in [f"{sub}_BCO", f"{sub}_ICO", f"{sub}_OTH"]:  # 不统计BCIC
                            continue
                        substance = np.array(combinef.variables[n][:])  # 获得对应物质数据
                        target = substance[:, 0, nearlat, nearlon]
                        substance_station = target[list_starttime:list_endtime]
                        sub_data_lists[n] = np.array(substance_station)  # 初始化每个样点相加的数据数组
                else:
                    for n in sub_data_lists:  # 获得样点所有PM2.5tag数据，一个月的逐小时数据列表
                        if n == f"{sub}_ALL":  # 还未计算出ALL，先跳过
                            continue
                        if n in [f"{sub}_BCO", f"{sub}_ICO", f"{sub}_OTH"]:  # 不统计BCIC
                            continue
                        substance = np.array(combinef.variables[n][:])  # 获得对应物质数据
                        target = substance[:, 0, nearlat, nearlon]
                        substance_station = target[list_starttime:list_endtime]
                        sub_data_lists[n] += np.array(substance_station)  # 相加

            for n in sub_data_lists:  # 样点平均
                if n == f"{sub}_ALL":  # 还未计算出ALL，先跳过
                    continue
                if n in [f"{sub}_BCO", f"{sub}_ICO", f"{sub}_OTH"]:  # 不统计BCIC
                    continue
                sub_data_lists[n] /= sp_count

            ALL = np.zeros(list_endtime - list_starttime)  # 创建一个空array来作为每个区域所有tag的和ALL的计算，仅包含列表
            for n in sub_data_lists:
                if n == f"{sub}_ALL":  # 还未计算出ALL，先跳过
                    continue
                if n in [f"{sub}_BCO", f"{sub}_ICO", f"{sub}_OTH"]: # 不统计BCIC
                    continue
                ALL += np.array(sub_data_lists[n])  # 先转换成array计算ALL
            sub_data_lists[f"{sub}_ALL"] = ALL  # 在转换为列表

            for n in sub_data_lists:  # 对每一天进行日均统计
                if n in [f"{sub}_BCO", f"{sub}_ICO", f"{sub}_OTH"]: # 不统计BCIC
                    continue
                for i in range(0, len(sub_data_lists[n]), 24):  # 总时间列表拆分
                    day_splt_list = sub_data_lists[n][i + 11:i + 19]
                    sub_data_lists_daymean[n].append(np.nanmean(day_splt_list))  # 日均值得到存入列表


            for n in sub_data_lists_daymean:  # 对每一天进行日均统计求百分比
                if n in [f"{sub}_BCO", f"{sub}_ICO", f"{sub}_OTH"]: # 不统计BCIC
                    continue
                percent = list(np.array(sub_data_lists_daymean[n]) / np.array(sub_data_lists_daymean[f"{sub}_ALL"]))
                sub_data_lists_daymeanPercent[n] = percent

            # 计算月均，即日均总和/天数
            for n in sub_data_lists_daymean:
                month_mean = 0
                for i in range(0, len(sub_data_lists_daymean[n])):  # 每个区域月均 (测试数据现在只有5天！！！)
                    month_mean += sub_data_lists_daymean[n][i]
                # print(len(sub_data_lists_daymean[n]))
                month_mean /= daynum
                # print(month_mean)
                sub_data_lists_monthmean[n].append(month_mean)

            for n in sub_data_lists_monthmean:  # 对每一天进行日均统计求百分比
                percent = np.array(sub_data_lists_monthmean[n]) / np.array(sub_data_lists_monthmean[f"{sub}_ALL"])
                sub_data_lists_monthmeanPercent[n].append(list(percent))

        for n in sub_data_lists_monthmeanPercent:  # 获得用于画条形图的百分比数据
            if n not in [f"{sub}_ALL",f"{sub}_BCO", f"{sub}_ICO", f"{sub}_OTH"]:
                sub_data_lists_monthmeanPercent_draw[n] = np.array(sub_data_lists_monthmeanPercent[n][-1])
                bottom_shape = sub_data_lists_monthmeanPercent_draw[n].shape

        # 输出贡献率结果cs
        contri_df = pd.DataFrame(columns=['各类排放源贡献率(%)']+list(Regions_locations.keys()))
        contri_emicolumns = []
        for x in list(sub_data_lists_monthmeanPercent_draw.keys()):
            if x == f"{sub}_ALL" :continue
            if x in [f"{sub}_BCO", f"{sub}_ICO", f"{sub}_OTH"]: continue # 不统计BCIC
            contri_emicolumns.append(tags_labels[x.replace(sub,'')])
        contri_df['各类排放源贡献率(%)'] = contri_emicolumns # 第一列为每个tag排放源的中文名
        for y in range(len(Regions_locations.keys())): # 每列受影响区域点
            contri_per = []
            for x in list(sub_data_lists_monthmeanPercent_draw.keys()): # 每行源贡献
                if x == f"{sub}_ALL": continue
                if x in [f"{sub}_BCO", f"{sub}_ICO", f"{sub}_OTH"]: continue  # 不统计BCIC
                contri_per.append(float(f"{sub_data_lists_monthmeanPercent_draw[x][y] * 100:.2f}"))
            contri_df[list(Regions_locations.keys())[y]] = contri_per

        print(contri_df)
        os.makedirs(result_dir, exist_ok=True)
        contri_df.to_csv(f"{result_dir}Contribution_{result_suffix}.csv")


if __name__ == "__main__":
    Regions_locations = {'成都主城区': [[104.07, 30.67]],
                         '崇州': [[103.69, 30.62]],
                         '都江堰': [[103.60, 30.98]],
                         '大邑': [[103.48, 30.60]],
                         '金堂': [[104.46, 30.85]],
                         '简阳': [[104.52, 30.40]],
                         '龙泉驿': [[104.29, 30.57]],
                         '郫都': [[103.87, 30.83]],
                         '蒲江': [[103.51, 30.23]],
                         '彭州': [[103.93, 31.01]],
                         '青白江': [[104.29, 30.86]],
                         '邛崃': [[103.44, 30.42]],
                         '双流': [[104.0, 30.50]],
                         '温江': [[103.82, 30.71]],
                         '新都': [[104.11, 30.85]],
                         '新津': [[103.81, 30.42]], }  # 所有的主要区域以及样点
    Regions_labels = {'_CDEM': "成都主城区区域源",
                         '_CZEM': "崇州区域源",
                         '_DJYEM': "都江堰区域源",
                         '_DYEM': "大邑区域源",
                         '_JTEM': "金堂区域源",
                         '_JYEM': "简阳区域源",
                         '_LQYEM': "龙泉驿区域源",
                         '_PDEM': "郫都区域源",
                         '_PJEM': "蒲江区域源",
                         '_PZEM': "彭州区域源",
                         '_QBJEM': "青白江区域源",
                         '_QLEM': "邛崃区域源",
                         '_SLEM': "双流区域源",
                         '_WJEM': "温江区域源",
                         '_XDEM': "新都区域源",
                         '_XJEM': "新津区域源", }
    emission_labels = {'_INDEM': "工业源",
                      '_AGREM': "农业源",
                      '_BOVEM': "生物源",
                      '_TRAEM': "交通源",
                      '_RESEM': "居民源",
                      '_POWEM': "电力源",
                         }  # 所有的主要区域以及样点

    CMAQ_calcuISAMcontribute(
            combinef_dir="E:\WCAS_serverfiles\cctm\cctmcombine_repm25.nc",  # CMAQISAM 的combine后文件位置，暂时只支持单个combine文件
            GRIDCRO2D_dir="E:\WCAS_serverfiles\GRIDCRO2D_2024333.nc",
            result_dir=r"E:\WCAS_serverfiles\\ISAMcontri\\",
            result_suffix = "repm25",
            start_date="2024-11-30",
            Regions_locations=Regions_locations,  # 被各类源影响区域名称及坐标字典，以位置点来代表区域，每个区域可多个点，格式为'CD': [[104.07, 30.67]]
            tags_labels = Regions_labels,
            target_vars=['PM25'],  # 需要计算的被标记源，每个物质输出一个结果，必须是被标记的物质
    )
    CMAQ_calcuISAMcontribute(
        combinef_dir="E:\WCAS_serverfiles\cctm\cctmcombine_empm25.nc",  # CMAQISAM 的combine后文件位置，暂时只支持单个combine文件
        GRIDCRO2D_dir="E:\WCAS_serverfiles\GRIDCRO2D_2024333.nc",
        result_dir=r"E:\WCAS_serverfiles\\ISAMcontri\\",
        result_suffix="empm25",
        start_date="2024-11-30",
        Regions_locations=Regions_locations,  # 被各类源影响区域名称及坐标字典，以位置点来代表区域，每个区域可多个点，格式为'CD': [[104.07, 30.67]]
        tags_labels=emission_labels,
        target_vars=['PM25'],  # 需要计算的被标记源，每个物质输出一个结果，必须是被标记的物质
    )
    CMAQ_calcuISAMcontribute(
            combinef_dir="E:\WCAS_serverfiles\cctm\cctmcombine_reo3.nc",  # CMAQISAM 的combine后文件位置，暂时只支持单个combine文件
            GRIDCRO2D_dir="E:\WCAS_serverfiles\GRIDCRO2D_2024333.nc",
            result_dir=r"E:\WCAS_serverfiles\\ISAMcontri\\",
            result_suffix = "reo3",
            start_date="2024-11-30",
            Regions_locations=Regions_locations,  # 被各类源影响区域名称及坐标字典，以位置点来代表区域，每个区域可多个点，格式为'CD': [[104.07, 30.67]]
            tags_labels = Regions_labels,
            target_vars=['O3'],  # 需要计算的被标记源，每个物质输出一个结果，必须是被标记的物质
    )
    CMAQ_calcuISAMcontribute(
        combinef_dir="E:\WCAS_serverfiles\cctm\cctmcombine_emo3.nc",  # CMAQISAM 的combine后文件位置，暂时只支持单个combine文件
        GRIDCRO2D_dir="E:\WCAS_serverfiles\GRIDCRO2D_2024333.nc",
        result_dir=r"E:\WCAS_serverfiles\\ISAMcontri\\",
        result_suffix="emo3",
        start_date="2024-11-30",
        Regions_locations=Regions_locations,  # 被各类源影响区域名称及坐标字典，以位置点来代表区域，每个区域可多个点，格式为'CD': [[104.07, 30.67]]
        tags_labels=emission_labels,
        target_vars=['O3'],  # 需要计算的被标记源，每个物质输出一个结果，必须是被标记的物质
    )



    pass