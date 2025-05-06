"""
Author: Yaohan Xian
GitHub: https://github.com/Sm0keNMirrors
Last update: 2025年1月14日
"""

import os
import re

import matplotlib
import numpy as np
import netCDF4 as nc
from tqdm import tqdm
import rasterio
from rasterio.transform import from_origin
import cartopy.crs as ccrs
from matplotlib import colors, font_manager, rcParams, pyplot as plt
import cartopy.feature as cfeat
from cartopy.io.shapereader import Reader
from datetime import datetime, timedelta


def save_2tiff(out_path, tif_data, ROW, COL, lonmin, lon_res, latmax, lat_res):
    # 创建GeoTIFF文件
    transform = from_origin(lonmin, latmax, lon_res, lat_res)  # 设置变换参数

    with rasterio.open(
            out_path, 'w',
            driver='GTiff',
            height=ROW,
            width=COL,
            count=1,
            dtype='float32',  # 数据类型
            crs='EPSG:4326',  # 设置坐标系为 WGS 84
            transform=transform
    ) as dst:
        dst.write(tif_data, 1)  # 将数据写入第一波段


def translate_expression_and_evaluate(expression,input_vars_get,np):  # 对公式进行表达式的切换
    # 替换表达式中的变量
    converted_expression = re.sub(r'~(.*?)~', r'input_vars_get["\1"]', expression)

    # 使用eval来计算表达式
    result = eval(converted_expression, {"np": np, "input_vars_get": input_vars_get})
    return result

def translate_expression_and_evaluate_2(expression,input_vars_get2,np):  # 对公式进行表达式的切换
    # 替换表达式中的变量
    converted_expression = re.sub(r'~(.*?)~', r'input_vars_get2["\1"]', expression)
    # 使用eval来计算表达式
    result = eval(converted_expression, {"np": np, "input_vars_get": input_vars_get2})
    return result

def getArrayVertices(ARR):
    top_left = ARR[0, 0]
    top_right = ARR[0, ARR.shape[1] - 1]
    bottom_left = ARR[ARR.shape[0] - 1, 0]
    bottom_right = ARR[ARR.shape[0] - 1, ARR.shape[1] - 1]
    return [top_left, top_right, bottom_left, bottom_right]

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
    daycount = len(WRF_file_list)
    for i in WRF_file_list:
        time = int(i[19:21])  # 文件名称天数的位置
        WRF_file_list_time.update({i: time})
    WRF_file_list_time = sorted(WRF_file_list_time.items(), key=lambda x: x[1])  # 字典按时间排序
    WRF_file_list_time = dict(WRF_file_list_time)

    WRFoutf = nc.Dataset(WRFout_files_dir + WRF_file_list[0], "r")  # 打开WRF输出的HDF格式文件
    data_shape = np.array(WRFoutf.variables["XLAT"]).shape  # 获得数据shape来存放空变量
    # metvar_now = np.zeros(data_shape, dtype=np.float64)
    metvar_next = np.zeros(data_shape, dtype=np.float64)
    WRFoutf.close()

    WRF_file_list_time_list = list(WRF_file_list_time.keys())  # 将WRF风场数据合并为1个文件，用来求8h平均
    metvar_now = np.array(nc.Dataset(WRFout_files_dir + WRF_file_list_time_list[0], "r").variables[
                              metvar])  # 合并的初始文件是第一个 拥有第一个wrfout的初始数据
    for n in tqdm(WRF_file_list_time_list,desc=f'读取并合并WRF输出参数{metvar}:'):
        n_num = WRF_file_list_time_list.index(n)
        if n_num + 1 >= len(WRF_file_list_time_list):
            metvar_combine = metvar_now
            continue
        n2 = WRF_file_list_time_list[n_num + 1]
        WRFoutf = nc.Dataset(WRFout_files_dir + n, "r")  # 打开WRF输出的HDF格式文件
        WRFoutf_next = nc.Dataset(WRFout_files_dir + n2, "r")  # 打开WRF输出的HDF格式文件

        metvar_next = np.array(WRFoutf_next.variables[metvar])  # 合并的WRF变量
        metvar_now = np.concatenate((metvar_now, metvar_next), axis=0)  # 合并

        WRFoutf.close()

    return metvar_combine

def WRFCMAQ_var2pic(
    start_date='YYYY-MM-DD',
    daycount=5,  #
    CMAQcombine_inithour=16,  #
    WRFout_files_dir = "",
    CMAQcombine_file_dir = "",
    GRIDCRO2D_file_dir = "",
    WRFout_files_winddata_dir = "",
    shp_files_dir = "",
    bgimage_dir_alpha = ["",0.5], # 是否添加底图tiff，以及添加后结果图的透明度,底图tiff经纬度原始范围
    input_vars ={'WRF':[],'CMAQ':[]}, # {'WRF':[],'CMAQ':[]}读取的变量，用于输出或者计算最终输出变量，来源模型:变量组
    target_vars = {'direct':[],'indirect':[]}, # 输出的变量，direct：直接读取后输出，indirect：计算后输出，计算方式即对应输出变量名称自行在函数中添加
    indirect_vars_fomular = {'RH' : "0.236 * ~PSFC~ * ~Q2~ * np.exp((17.67 * (~T2~ - 273.15)) / (~T2~ - 29.65)) ** (-1)"}, # indirect变量的计算公式，例如：'RH' : "0.236 * ~PSFC~ * ~Q2~ * np.exp((17.67 * (~T2~ - 273.15)) / (~T2~ - 29.65)) ** (-1)"
    target_vars_ifdrawwind = [], # 对应上面每个输出变量的图，哪个输出风向箭头，参数为布尔值序列[True,False,False]
    target_vars_cmap = [], # 输出变量的颜色条类型，依照target_vars顺序输入，包括常用的：jet、rainbow、pollution
    target_vars_unit = [], # 输出变量的单位，依照target_vars顺序输入，有μg/m$^{3}$，℃，m等
    target_vars_name = [], # 每个变量在图标题和bar标题显示的名称，而不是直接的变量名，依照target_vars顺序输入
    target_vars_cbarticksFormat = [], # cbar的标识数值保留小数几位，依照target_vars顺序输入，当数值小的时候需要设置，'%i' '%.1f'
    Molar_mass ={'target_var_name':0}, # 输出变量若需要进行ppm转换的摩尔质量：输出变量名：摩尔质量大小
    result_data_types=[], # 输出结果图的包含哪些时平均结果，包括hourly daily monthly monthly_max8h allmean
    cbar_min_max = [], # 制定绘图色条的最大值和最小值，不填写则默认最大最小值

    fig_size = (5,5), # 输出图像范围的大致长宽程度
    cbar_positions = ['horizontal',[0.1, 0.1, 0.8, 0.025]], # color的位置信息，[水平垂直信息, axe信息]
    result_positions = [0.12,0.88,0.2,0.95,0.2,0.2], # 经纬度图的绘制位置信息，即plt.subplots_adjust(left=0.12, right=0.88, bottom=0.2, top=0.95, wspace=0.2, hspace=0.2)的参数，
    result_pic_type='', #输出结果图类型contourf, pcolormesh
    manmual_extent = [], # 手动设置ax显示的经纬度范围，若为默认的[]，则是直接以nc经纬度的范围输出
    wind_scale = [5,0.003], # 风向箭头大小数据，scale和width，单位为inches

    out_dir = "",
    save2npy = False, # 是否将计算出的各类均值变量输出为npy文件，方便后续其他处理，默认输出到图片位置路径，同时保存绘图的x y LATLONG
    save2tiff = False, # 是否将计算出的各类均值变量输出为tiff文件，方便后续其他处理，默认输出到图片位置路径
    suffix = "",

    ifpicdif = False, # 是否绘制差值图像，即上面所有相关变量输入数据为被减数，基于下面的输入数据为减数的差值图输出，要求作差的两次数据格式一致
    WRFout_files2_dir = "",
    CMAQcombine_file2_dir = "", # 仅给出
    difresult_data_types=['allmean'], # 差值输出结果图的包含哪些平均结果，暂时仅支持allmean
    difresult_forceminmax = [], # 强制修正对应变量差值结果图的最大最小值，即部分异常值，[A,B] 或False格式 A最小修正 B为最大修正
    difpic_title = "", # 差值图自定义标题，表示哪两次模拟的
):
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(out_dir) is False: os.mkdir(out_dir)

    if CMAQcombine_file_dir != "":CMAQf = nc.Dataset(CMAQcombine_file_dir)
    if CMAQcombine_file2_dir != "": CMAQf2 = nc.Dataset(CMAQcombine_file2_dir)
    if GRIDCRO2D_file_dir != "": GRIDCRO2D = nc.Dataset(GRIDCRO2D_file_dir, 'r')
    if WRFout_files_dir != "": WRFoutfiles = os.listdir(WRFout_files_dir)
    if WRFout_files2_dir != "": WRFoutfiles2 = os.listdir(WRFout_files2_dir)
    if WRFout_files_winddata_dir != "": WRFoutfiles_winddata = os.listdir(WRFout_files_winddata_dir)
    if shp_files_dir != "": shpfiles = [x for x in os.listdir(shp_files_dir) if x.split('.')[1] == 'shp' and len(x.split('.')) == 2]

    if WRFout_files_dir != "":
        WRFoutf = nc.Dataset(WRFout_files_dir + WRFoutfiles[0], "r")  # 获得风场数据的grid
        lon = WRFoutf.variables['XLONG'][:][0]
        lat = WRFoutf.variables['XLAT'][:][0]
        ROW = np.array(WRFoutf.variables["XLAT"]).shape[1]  # 获得数据格式shape1
        COL = np.array(WRFoutf.variables["XLAT"]).shape[2]  # 获得数据格式shape2
        WRFoutf.close()

        # 用于输出tiff的参数 for WRF
        WRF_lonmin, WRF_latmax, WRF_lonmax, WRF_latmin = (lon.min(), lat.max(),lon.max(), lat.min())
        len_lat = ROW
        len_lon = COL
        WRF_ROW = ROW
        WRF_COL = COL
        WRF_lon_res = (WRF_lonmax - WRF_lonmin) / (len_lon - 1.0)
        WRF_lat_res = (WRF_latmax - WRF_latmin) / (len_lat - 1.0)

    # if GRIDCRO2D_file_dir != "": # 有gridcro2d就以其作为两个模型的grid
    #     lon = GRIDCRO2D.variables['LON'][:][0][0]
    #     lat = GRIDCRO2D.variables['LAT'][:][0][0]
    #     ROW = lon.shape[0] # 获得输出目标domian的row col，WRF CMAQ 均以CMAQ的grid 为主
    #     COL = lon.shape[1]
    if WRFout_files_winddata_dir != "":
        WRFoutf = nc.Dataset(WRFout_files_winddata_dir + WRFoutfiles_winddata[0], "r")  # 获得风场数据的grid
        ROW_wind = np.array(WRFoutf.variables["XLAT"]).shape[1]  # 获得数据格式shape1
        COL_wind = np.array(WRFoutf.variables["XLAT"]).shape[2]  # 获得数据格式shape2
        WRFoutf.close()
        wind_XLAT = getWRFvarsCombined(WRFout_files_winddata_dir, 'XLAT')
        wind_XLONG = getWRFvarsCombined(WRFout_files_winddata_dir, 'XLONG')
        wind_WSV = getWRFvarsCombined(WRFout_files_winddata_dir, 'V10')
        wind_WSU = getWRFvarsCombined(WRFout_files_winddata_dir, 'U10')
        wind_WD = 180.0 + np.arctan2(wind_WSU, wind_WSV) * 180.0 / np.pi  # 计算风向
        wind_XLAT_one = wind_XLAT[0,:,:]
        wind_XLONG_one = wind_XLONG[0, :, :]
        windshape = (1,ROW_wind,COL_wind)




    cmapdict = ['white', '#75bbfd', 'green', 'yellow', 'red', 'maroon']  # 自定义colorbar的颜色
    cmap_pollution = colors.LinearSegmentedColormap.from_list("name", cmapdict)
    target_vars_cmap_ = [cmap_pollution if x == 'pollution' else x for x in target_vars_cmap]

    lon_c = lon[0:lon.shape[0] - 2, 0:lon.shape[1] - 2]
    lat_c = lat[0:lat.shape[0] - 2, 0:lat.shape[1] - 2]
    # 用于输出tiff的参数 for CMAQ
    CMAQ_lonmin, CMAQ_latmax, CMAQ_lonmax, CMAQ_latmin = (lon_c.min(), lat_c.max(), lon_c.max(), lat_c.min())
    len_lat = ROW
    len_lon = COL
    CMAQ_ROW = ROW
    CMAQ_COL = COL
    CMAQ_lon_res = (CMAQ_lonmax - CMAQ_lonmin) / (len_lon - 1.0)
    CMAQ_lat_res = (CMAQ_latmax - WRF_latmin) / (len_lat - 1.0)
    if save2npy: np.save(out_dir + f'LONG_cmaq.npy'.replace(':', "-"), lon_c.filled(fill_value=0))
    if save2npy: np.save(out_dir + f'LAT_cmaq.npy'.replace(':', "-"), lat_c.filled(fill_value=0))
    if save2npy: np.save(out_dir + f'LONG_wrf.npy'.replace(':', "-"), lon.filled(fill_value=0))
    if save2npy: np.save(out_dir + f'LAT_wrf.npy'.replace(':', "-"), lat.filled(fill_value=0))


    input_vars_get = {}
    if ifpicdif == True: input_vars_get2 = {}
    for model in input_vars:
        if model == 'WRF':
            for var in input_vars[model]:
                input_vars_get.update({var:getWRFvarsCombined(WRFout_files_dir,var)})
                if ifpicdif == True: input_vars_get2.update({var:getWRFvarsCombined(WRFout_files2_dir,var)})
        if model == 'CMAQ':
            for var in input_vars[model]:
                input_vars_get.update({var:CMAQf[var][CMAQcombine_inithour:CMAQcombine_inithour+24*daycount]})
                if ifpicdif == True: input_vars_get2.update({var:CMAQf2[var][CMAQcombine_inithour:CMAQcombine_inithour+24*daycount]})

    output_vars = {}
    if ifpicdif == True: output_vars2 = {}
    for type in target_vars:
        if type == 'direct':
            for var in target_vars[type]:
                output_vars.update({var:input_vars_get[var]})
                if ifpicdif == True:output_vars2.update({var:input_vars_get2[var]})
    for type in target_vars:
        if target_vars['indirect'] == []: continue
        if type == 'indirect':
            for var in target_vars[type]:
                # 输入对应indirect的变量的计算方法，通过确定的input变量来计算,如RH，ISAM总浓度等
                # =====================😼😼😼😼😼😼😼😼😼########😼😼😼😼😼😼😼😼😼😼😼😼😼=====================


                # if var == 'RH':
                #     vardata = 0.236 * input_vars_get['PSFC'] * input_vars_get['Q2'] * np.exp((17.67 * (input_vars_get['T2'] - 273.15)) / (input_vars_get['T2'] - 29.65)) ** (-1)
                #     output_vars.update({var: vardata})
                #     if ifpicdif == True:
                #         vardata2 = 0.236 * input_vars_get2['PSFC'] * input_vars_get2['Q2'] * np.exp(
                #             (17.67 * (input_vars_get2['T2'] - 273.15)) / (input_vars_get2['T2'] - 29.65)) ** (-1)
                #         output_vars2.update({var: vardata2})
                # if var == 'T2_C':
                #     vardata = input_vars_get['T2'] - 273.15
                #     output_vars.update({var: vardata})
                #     if ifpicdif == True:
                #         vardata2 = input_vars_get2['T2'] - 273.15
                #         output_vars2.update({var: vardata2})
                # if var == 'TSK_C':
                #     vardata = input_vars_get['TSK'] - 273.15
                #     output_vars.update({var: vardata})
                #     if ifpicdif == True:
                #         vardata2 = input_vars_get2['TSK'] - 273.15
                #         output_vars2.update({var: vardata2})
                # if var == 'WS':
                #     vardata = np.sqrt(input_vars_get['V10'] ** 2 + input_vars_get['U10'] ** 2)  # 分速度求和速度
                #     output_vars.update({var: vardata})
                #     if ifpicdif == True:
                #         vardata2 = np.sqrt(input_vars_get2['V10'] ** 2 + input_vars_get2['U10'] ** 2)  # 分速度求和速度
                #         output_vars2.update({var: vardata2})
                # if var == 'VC':
                #     vardata = np.sqrt(input_vars_get['V10'] ** 2 + input_vars_get['U10'] ** 2) * input_vars_get['PBLH']
                #     output_vars.update({var: vardata})
                #     if ifpicdif == True:
                #         vardata2 = np.sqrt(input_vars_get2['V10'] ** 2 + input_vars_get2['U10'] ** 2)  * input_vars_get2['PBLH']
                #         output_vars2.update({var: vardata2})


                vardata = translate_expression_and_evaluate(indirect_vars_fomular[var],input_vars_get,np)
                output_vars.update({var: vardata})
                if ifpicdif == True:
                    vardata2 = translate_expression_and_evaluate_2(indirect_vars_fomular[var],input_vars_get2,np)
                    output_vars2.update({var: vardata2})



                # =====================😼😼😼😼😼😼😼😼😼########😼😼😼😼😼😼😼😼😼😼😼😼😼=====================
    hourly_datas,daily_datas,allmean_datas = {},{},{} # 用于输出直接的array结果
    for var in target_vars['direct']+target_vars['indirect']:
        hourly_datas.update({var:[]})
        daily_datas.update({var: []})
        allmean_datas.update({var: []})

    # 绘制图片基本框架
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    font_path = r"F:\wrfdata\times+simsun.ttf"
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    matplotlib.rcParams['font.family'] = 'sans-serif'  # 使用字体中的无衬线体
    rcParams['font.sans-serif'] = prop.get_name()  # 根据名称设置字体
    rcParams['font.size'] = 13  # 设置字体大小
    rcParams['axes.unicode_minus'] = False  # 使坐标轴刻度标签正常显示正负号

    if 'hourly' in result_data_types:
        start_date_o = datetime.strptime(start_date, '%Y-%m-%d')
        for hour in tqdm(range(0+16,daycount*24),desc="输出逐小时结果图："):
            hourly_out_dir = out_dir + "hourly\\"
            hour_BJ = hour-16 # 北京时间 显示在title的 hour是utc
            if os.path.exists(hourly_out_dir) is False: os.mkdir(hourly_out_dir)
            for var in output_vars:
                hourly_var_out_dir = hourly_out_dir + f"{var}\\"
                if os.path.exists(hourly_var_out_dir) is False: os.mkdir(hourly_var_out_dir)

                date_now = start_date_o + timedelta(hours=hour_BJ)
                index = list(output_vars.keys()).index(var)
                if output_vars[var].ndim == 4 : hourlydata = output_vars[var][hour,0,:,:]
                if output_vars[var].ndim == 3: hourlydata = output_vars[var][hour, :, :]
                if cbar_min_max == []:
                    cbarmax = np.max(output_vars[var])
                    cbarmin = np.min(output_vars[var])
                else: cbarmin,cbarmax = cbar_min_max[0],cbar_min_max[1]
                # if cbarmin == cbarmax: continue # 说明数据为空 不输出
                if cbarmin == cbarmax: cbarmax,cbarmin = 1,0  # 数据为空但继续输出

                proj = ccrs.PlateCarree()  # 创建坐标系
                fig = plt.figure(figsize=fig_size,dpi=150)  # 创建页面
                ax = fig.subplots(1, 1, subplot_kw={'projection': proj})
                # 读取所有shp并绘制
                for shp in shpfiles:
                    shp_c = cfeat.ShapelyFeature(Reader(shp_files_dir + shp).geometries(), proj, edgecolor='k',
                                                 facecolor='none')
                    ax.add_feature(shp_c, lw=0.6, zorder=2)
                position = fig.add_axes(cbar_positions[1])  # colorbar位置
                if result_pic_type == 'contourf':
                    if hourlydata.shape != lon.shape: # 因为前面让所有输出的latlon都以WRF为基准，且CMAQ的数据行列相当于WRF少2个行列，则要对用于绘图的latlon进行裁剪
                        lon_c = lon[0:lon.shape[0]-2,0:lon.shape[1]-2]
                        lat_c = lat[0:lat.shape[0] - 2, 0:lat.shape[1] - 2]
                        data_pic = ax.contourf(lon_c, lat_c, hourlydata, cmap=target_vars_cmap_[index],
                                               levels=np.linspace(cbarmin, cbarmax, 80))
                        LAT_vert = getArrayVertices(lat_c)  # 获取经纬度数组的四个顶点数据列表
                        LON_vert = getArrayVertices(lon_c)
                        ax.set_extent([LON_vert[0], LON_vert[1], LAT_vert[0], LAT_vert[2]])  # 显示范围
                        tifftypeflag = 'CMAQ'
                    else:
                        data_pic = ax.contourf(lon, lat, hourlydata, cmap=target_vars_cmap_[index],levels=np.linspace(cbarmin, cbarmax, 80))
                        LAT_vert = getArrayVertices(lat)  # 获取经纬度数组的四个顶点数据列表
                        LON_vert = getArrayVertices(lon)
                        ax.set_extent([LON_vert[0], LON_vert[1], LAT_vert[0], LAT_vert[2]])  # 显示范围
                        tifftypeflag = 'WRF'
                if result_pic_type == 'pcolormesh':
                    if hourlydata.shape != lon.shape: # CMAQ的数据行列相当于WRF少2个行列，要对latlon进行裁剪
                        lon_c = lon[0:lon.shape[0]-2,0:lon.shape[1]-2]
                        lat_c = lat[0:lat.shape[0] - 2, 0:lat.shape[1] - 2]
                        data_pic = ax.pcolormesh(lon_c, lat_c, hourlydata, cmap=target_vars_cmap_[index],vmin=cbarmin,vmax=cbarmax)
                        LAT_vert = getArrayVertices(lat_c)  # 获取经纬度数组的四个顶点数据列表
                        LON_vert = getArrayVertices(lon_c)
                        ax.set_extent([LON_vert[0], LON_vert[1], LAT_vert[0], LAT_vert[2]])  # 显示范围
                        tifftypeflag = 'CMAQ'
                    else:
                        data_pic = ax.pcolormesh(lon, lat, hourlydata, cmap=target_vars_cmap_[index],vmin=cbarmin,vmax=cbarmax)
                        LAT_vert = getArrayVertices(lat)  # 获取经纬度数组的四个顶点数据列表
                        LON_vert = getArrayVertices(lon)
                        ax.set_extent([LON_vert[0], LON_vert[1], LAT_vert[0], LAT_vert[2]])  # 显示范围
                        tifftypeflag = 'WRF'
                if manmual_extent != []:ax.set_extent(manmual_extent)  # 显示范围 当需要强制控制时修改
                cb_ticks = np.linspace(cbarmin, cbarmax, 7)
                cb = fig.colorbar(data_pic, cax=position, orientation=cbar_positions[0], extend='both',ticks=cb_ticks, format=target_vars_cbarticksFormat[index], fraction=0.2)
                cb.ax.tick_params(labelsize=17)  # 刻度字体大小
                cb.set_label(label=target_vars_name[index]+' '+f'(units: {target_vars_unit[index]})', fontsize=12)  # 设置colorbar的标签字体及其大小

                if WRFout_files_winddata_dir != "" and target_vars_ifdrawwind[index] == True:
                    wind_WSU_d = wind_WSU[hour, :, :]
                    wind_WSV_d = wind_WSV[hour, :, :]
                    ax.quiver(wind_XLONG_one[:, :], wind_XLAT_one[:, :], wind_WSU_d, wind_WSV_d,
                              transform=proj, scale=wind_scale[0],
                              scale_units='inches', width=wind_scale[1])
                gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1.2, color='k', alpha=0.2,
                                  linestyle='--')
                if bgimage_dir_alpha[0] != "": # 绘制底图
                    ax.imshow(plt.imread(bgimage_dir_alpha[0]), origin='upper',zorder=1, transform=proj, extent=bgimage_dir_alpha[2], alpha=bgimage_dir_alpha[1])
                hourly_datas[var].append(hourlydata) # 存放结果array
                plt.subplots_adjust(left=result_positions[0], right=result_positions[1], bottom=result_positions[2], top=result_positions[3], wspace=result_positions[4], hspace=result_positions[5])
                ax.set_title(f'{target_vars_name[index]} {str(date_now)}小时均值',fontsize=15,pad=30)
                gl.xlabel_style = {'size': 15}  # 设置经度标签字体大小
                gl.ylabel_style = {'size': 15}  # 设置纬度标签字体大小
                plt.savefig(hourly_var_out_dir+f'{var} {str(date_now)}'.replace(':',"-"))
                plt.close()
                if save2npy: np.save(hourly_var_out_dir + f'{var} {str(date_now)}.npy'.replace(':', "-"), hourlydata)
                if save2tiff:
                    if tifftypeflag == 'WRF': # WRF和CMAQ的tiff输出格式参数不一样
                        save_2tiff(hourly_var_out_dir + f'{var} {str(date_now)}.tiff'.replace(':', "-"),np.flipud(hourlydata),WRF_ROW,WRF_COL,
                                   WRF_lonmin,WRF_lon_res,WRF_latmax,WRF_lat_res)
                    if tifftypeflag == 'CMAQ':
                        save_2tiff(hourly_var_out_dir + f'{var} {str(date_now)}.tiff'.replace(':', "-"),np.flipud(hourlydata),CMAQ_ROW,CMAQ_COL,
                                   CMAQ_lonmin,CMAQ_lon_res,CMAQ_latmax,CMAQ_lat_res)

    if 'daily' in result_data_types:
        start_date_o = datetime.strptime(start_date, '%Y-%m-%d')
        for day in tqdm(range(0,daycount),desc="输出逐日平均结果图："):
            daily_out_dir = out_dir + "daily\\"
            if os.path.exists(daily_out_dir) is False: os.mkdir(daily_out_dir)
            for var in output_vars:
                daily_var_out_dir = daily_out_dir + f"{var}\\"
                if os.path.exists(daily_var_out_dir) is False: os.mkdir(daily_var_out_dir)

                date_now = start_date_o + timedelta(days=day)
                index = list(output_vars.keys()).index(var)
                if output_vars[var].ndim == 4 :
                    dailydata = np.zeros(shape=(output_vars[var].shape[2], output_vars[var].shape[3]))
                    for hour in range(0,24):
                        dailydata += output_vars[var][day*24+hour+16,0,:,:]
                    dailydata /= 24
                if output_vars[var].ndim == 3:
                    dailydata = np.zeros(shape=(output_vars[var].shape[1], output_vars[var].shape[2]))
                    for hour in range(0,24):
                        dailydata += output_vars[var][day*24+hour+16,:,:]
                    dailydata /= 24
                if cbar_min_max == []:
                    cbarmax = np.max(dailydata)
                    cbarmin = np.min(dailydata)
                else: cbarmin,cbarmax = cbar_min_max[0],cbar_min_max[1]
                # if cbarmin == cbarmax: continue # 说明数据为空 不输出
                if cbarmin == cbarmax: cbarmax,cbarmin = 1,0  # 数据为空但继续输出

                proj = ccrs.PlateCarree()  # 创建坐标系
                fig = plt.figure(figsize=fig_size,dpi=150)  # 创建页面
                ax = fig.subplots(1, 1, subplot_kw={'projection': proj})
                # 读取所有shp并绘制
                for shp in shpfiles:
                    shp_c = cfeat.ShapelyFeature(Reader(shp_files_dir + shp).geometries(), proj, edgecolor='k',
                                                 facecolor='none')
                    ax.add_feature(shp_c, lw=0.6, zorder=2)
                position = fig.add_axes(cbar_positions[1])  # colorbar位置
                if result_pic_type == 'contourf':
                    if dailydata.shape != lon.shape: # CMAQ的数据行列相当于WRF少2个行列，要对latlon进行裁剪
                        lon_c = lon[0:lon.shape[0]-2,0:lon.shape[1]-2]
                        lat_c = lat[0:lat.shape[0] - 2, 0:lat.shape[1] - 2]
                        data_pic = ax.contourf(lon_c, lat_c, dailydata, cmap=target_vars_cmap_[index],
                                               levels=np.linspace(cbarmin, cbarmax, 80))
                        LAT_vert = getArrayVertices(lat_c)  # 获取经纬度数组的四个顶点数据列表
                        LON_vert = getArrayVertices(lon_c)
                        ax.set_extent([LON_vert[0], LON_vert[1], LAT_vert[0], LAT_vert[2]])  # 显示范围
                        tifftypeflag = 'CMAQ'
                    else:
                        data_pic = ax.contourf(lon, lat, dailydata, cmap=target_vars_cmap_[index],levels=np.linspace(cbarmin, cbarmax, 80))
                        LAT_vert = getArrayVertices(lat)  # 获取经纬度数组的四个顶点数据列表
                        LON_vert = getArrayVertices(lon)
                        ax.set_extent([LON_vert[0], LON_vert[1], LAT_vert[0], LAT_vert[2]])  # 显示范围
                        tifftypeflag = 'WRF'
                if result_pic_type == 'pcolormesh':
                    if dailydata.shape != lon.shape: # CMAQ的数据行列相当于WRF少2个行列，要对latlon进行裁剪
                        lon_c = lon[0:lon.shape[0]-2,0:lon.shape[1]-2]
                        lat_c = lat[0:lat.shape[0] - 2, 0:lat.shape[1] - 2]
                        data_pic = ax.pcolormesh(lon_c, lat_c, dailydata, cmap=target_vars_cmap_[index],vmin=cbarmin,vmax=cbarmax)
                        LAT_vert = getArrayVertices(lat_c)  # 获取经纬度数组的四个顶点数据列表
                        LON_vert = getArrayVertices(lon_c)
                        ax.set_extent([LON_vert[0], LON_vert[1], LAT_vert[0], LAT_vert[2]])  # 显示范围
                        tifftypeflag = 'CMAQ'
                    else:
                        data_pic = ax.pcolormesh(lon, lat, dailydata, cmap=target_vars_cmap_[index],vmin=cbarmin,vmax=cbarmax)
                        LAT_vert = getArrayVertices(lat)  # 获取经纬度数组的四个顶点数据列表
                        LON_vert = getArrayVertices(lon)
                        ax.set_extent([LON_vert[0], LON_vert[1], LAT_vert[0], LAT_vert[2]])  # 显示范围
                        tifftypeflag = 'WRF'
                if manmual_extent != []: ax.set_extent(manmual_extent)  # 显示范围 当需要强制控制时修改
                cb_ticks = np.linspace(cbarmin, cbarmax, 7)
                cb = fig.colorbar(data_pic, cax=position, orientation=cbar_positions[0], extend='both',ticks=cb_ticks, format=target_vars_cbarticksFormat[index], fraction=0.2)
                cb.ax.tick_params(labelsize=17)  # 刻度字体大小
                cb.set_label(label=target_vars_name[index]+' '+f'(units: {target_vars_unit[index]})', fontsize=12)  # 设置colorbar的标签字体及其大小
                if WRFout_files_winddata_dir != "" and target_vars_ifdrawwind[index] == True:
                    wind_WSU_d = np.zeros(windshape, dtype=np.float64)
                    wind_WSV_d = np.zeros(windshape, dtype=np.float64)
                    for h in range(0, 24):
                        wind_WSU_d += wind_WSU[day*24+hour+16, :, :]
                        wind_WSV_d += wind_WSV[day*24+hour+16, :, :]
                    wind_WSV_d /= 24
                    wind_WSU_d /= 24
                    ax.quiver(wind_XLONG_one[:, :], wind_XLAT_one[:, :], wind_WSU_d[0, :, :], wind_WSV_d[0, :, :],
                              transform=proj, scale=wind_scale[0],
                              scale_units='inches', width=wind_scale[1])
                gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1.2, color='k', alpha=0.2,
                                  linestyle='--')
                if bgimage_dir_alpha[0] != "": # 绘制底图
                    ax.imshow(plt.imread(bgimage_dir_alpha[0]), origin='upper',zorder=1, transform=proj, extent=bgimage_dir_alpha[2], alpha=bgimage_dir_alpha[1])
                daily_datas[var].append(dailydata)  # 存放结果array
                plt.subplots_adjust(left=result_positions[0], right=result_positions[1], bottom=result_positions[2],
                                    top=result_positions[3], wspace=result_positions[4], hspace=result_positions[5])
                ax.set_title(f'{target_vars_name[index]} {str(date_now)}日均值',fontsize=15,pad=30)
                gl.xlabel_style = {'size': 15}  # 设置经度标签字体大小
                gl.ylabel_style = {'size': 15}  # 设置纬度标签字体大小
                plt.savefig(daily_var_out_dir+f'{var} {str(date_now)}'.replace(':',"-"))
                plt.close()
                if save2npy: np.save(daily_var_out_dir + f'{var} {str(date_now)}.npy'.replace(':', "-"), dailydata)
                if save2tiff:
                    if tifftypeflag == 'WRF': # WRF和CMAQ的tiff输出格式参数不一样
                        save_2tiff(daily_var_out_dir + f'{var} {str(date_now)}.tiff'.replace(':', "-"),np.flipud(dailydata),WRF_ROW,WRF_COL,
                                   WRF_lonmin,WRF_lon_res,WRF_latmax,WRF_lat_res)
                    if tifftypeflag == 'CMAQ':
                        save_2tiff(daily_var_out_dir + f'{var} {str(date_now)}.tiff'.replace(':', "-"),np.flipud(dailydata),CMAQ_ROW,CMAQ_COL,
                                   CMAQ_lonmin,CMAQ_lon_res,CMAQ_latmax,CMAQ_lat_res)

    if 'allmean' in result_data_types:
        start_date_o = datetime.strptime(start_date, '%Y-%m-%d')
        for day in tqdm(range(0,1),desc="输出模拟时段总平均值："):
            allmean_out_dir = out_dir + "allmean\\"
            if os.path.exists(allmean_out_dir) is False: os.mkdir(allmean_out_dir)
            for var in output_vars:
                allmean_var_out_dir = allmean_out_dir + f"{var}\\"
                if os.path.exists(allmean_var_out_dir) is False: os.mkdir(allmean_var_out_dir)

                index = list(output_vars.keys()).index(var)
                if output_vars[var].ndim == 4 :
                    allmeandata = np.zeros(shape=(output_vars[var].shape[2], output_vars[var].shape[3]))
                    for hour in range(0,24*daycount):
                        allmeandata += output_vars[var][hour,0,:,:]
                    allmeandata /= 24*daycount
                if output_vars[var].ndim == 3:
                    allmeandata = np.zeros(shape=(output_vars[var].shape[1], output_vars[var].shape[2]))
                    for hour in range(0,24*daycount):
                        allmeandata += output_vars[var][hour,:,:]
                    allmeandata /= 24*daycount
                if cbar_min_max == []:
                    # cbarmax = np.ceil(np.max(allmeandata) / 10) * 10
                    # cbarmin = np.floor(np.min(allmeandata) / 10) * 10
                    cbarmax = np.max(allmeandata)
                    cbarmin = np.min(allmeandata)
                else: cbarmin,cbarmax = cbar_min_max[0],cbar_min_max[1]
                # if cbarmin == cbarmax: continue # 说明数据为空 不输出
                # if cbarmin == cbarmax: cbarmax,cbarmin = 1,0  # 数据为空但继续输出

                proj = ccrs.PlateCarree()  # 创建坐标系
                fig = plt.figure(figsize=fig_size,dpi=150)  # 创建页面
                ax = fig.subplots(1, 1, subplot_kw={'projection': proj})
                # 读取所有shp并绘制
                for shp in shpfiles:
                    shp_c = cfeat.ShapelyFeature(Reader(shp_files_dir + shp).geometries(), proj, edgecolor='k',
                                                 facecolor='none')
                    ax.add_feature(shp_c, lw=0.6, zorder=2)
                position = fig.add_axes(cbar_positions[1])  # colorbar位置
                if result_pic_type == 'contourf':
                    if allmeandata.shape != lon.shape: # CMAQ的数据行列相当于WRF少2个行列，要对latlon进行裁剪
                        lon_c = lon[0:lon.shape[0]-2,0:lon.shape[1]-2]
                        lat_c = lat[0:lat.shape[0] - 2, 0:lat.shape[1] - 2]
                        data_pic = ax.contourf(lon_c, lat_c, allmeandata, cmap=target_vars_cmap_[index],
                                               levels=np.linspace(cbarmin, cbarmax, 80))
                        LAT_vert = getArrayVertices(lat_c)  # 获取经纬度数组的四个顶点数据列表
                        LON_vert = getArrayVertices(lon_c)
                        ax.set_extent([LON_vert[0], LON_vert[1], LAT_vert[0], LAT_vert[2]])  # 显示范围
                        tifftypeflag = 'CMAQ'
                    else:
                        data_pic = ax.contourf(lon, lat, allmeandata, cmap=target_vars_cmap_[index],levels=np.linspace(cbarmin, cbarmax, 80))
                        LAT_vert = getArrayVertices(lat)  # 获取经纬度数组的四个顶点数据列表
                        LON_vert = getArrayVertices(lon)
                        ax.set_extent([LON_vert[0], LON_vert[1], LAT_vert[0], LAT_vert[2]])  # 显示范围
                        tifftypeflag = 'WRF'
                if result_pic_type == 'pcolormesh':
                    if allmeandata.shape != lon.shape: # CMAQ的数据行列相当于WRF少2个行列，要对latlon进行裁剪
                        lon_c = lon[0:lon.shape[0]-2,0:lon.shape[1]-2]
                        lat_c = lat[0:lat.shape[0] - 2, 0:lat.shape[1] - 2]
                        data_pic = ax.pcolormesh(lon_c, lat_c, allmeandata, cmap=target_vars_cmap_[index],vmin=cbarmin,vmax=cbarmax)
                        LAT_vert = getArrayVertices(lat_c)  # 获取经纬度数组的四个顶点数据列表
                        LON_vert = getArrayVertices(lon_c)
                        ax.set_extent([LON_vert[0], LON_vert[1], LAT_vert[0], LAT_vert[2]])  # 显示范围
                        tifftypeflag = 'CMAQ'
                    else:
                        data_pic = ax.pcolormesh(lon, lat, allmeandata, cmap=target_vars_cmap_[index],vmin=cbarmin,vmax=cbarmax)
                        LAT_vert = getArrayVertices(lat)  # 获取经纬度数组的四个顶点数据列表
                        LON_vert = getArrayVertices(lon)
                        ax.set_extent([LON_vert[0], LON_vert[1], LAT_vert[0], LAT_vert[2]])  # 显示范围
                        tifftypeflag = 'WRF'
                if manmual_extent != []: ax.set_extent(manmual_extent)  # 显示范围 当需要强制控制时修改
                cb_ticks = np.linspace(cbarmin, cbarmax, 7)
                cb = fig.colorbar(data_pic, cax=position, orientation=cbar_positions[0], extend='both',ticks=cb_ticks, format=target_vars_cbarticksFormat[index], fraction=0.2)
                cb.ax.tick_params(labelsize=17)  # 刻度字体大小
                cb.set_label(label=target_vars_name[index]+' '+f'(units: {target_vars_unit[index]})', fontsize=12)  # 设置colorbar的标签字体及其大小
                if WRFout_files_winddata_dir != "" and target_vars_ifdrawwind[index] == True:
                    wind_WSU_d = np.zeros(windshape, dtype=np.float64)
                    wind_WSV_d = np.zeros(windshape, dtype=np.float64)
                    for h in range(0,24*daycount):
                        wind_WSU_d += wind_WSU[h, :, :]
                        wind_WSV_d += wind_WSV[h, :, :]
                    wind_WSV_d /= 24*daycount
                    wind_WSU_d /= 24*daycount
                    ax.quiver(wind_XLONG_one[:,:], wind_XLAT_one[:,:], wind_WSU_d[0,:,:], wind_WSV_d[0,:,:], transform=proj, scale=wind_scale[0],
                          scale_units='inches', width=wind_scale[1])
                gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1.2, color='k', alpha=0.2,
                                  linestyle='--')
                if bgimage_dir_alpha[0] != "": # 绘制底图
                    ax.imshow(plt.imread(bgimage_dir_alpha[0]), origin='upper',zorder=1, transform=proj, extent=bgimage_dir_alpha[2], alpha=bgimage_dir_alpha[1])
                allmean_datas[var].append(allmeandata)  # 存放结果array
                plt.subplots_adjust(left=result_positions[0], right=result_positions[1], bottom=result_positions[2],
                                    top=result_positions[3], wspace=result_positions[4], hspace=result_positions[5])
                ax.set_title(f'{target_vars_name[index]} 模拟时段总均值',fontsize=15,pad=30)
                gl.xlabel_style = {'size': 15}  # 设置经度标签字体大小
                gl.ylabel_style = {'size': 15}  # 设置纬度标签字体大小
                plt.savefig(allmean_var_out_dir+f'{var} allmean'.replace(':',"-"))
                plt.close()
                if save2npy: np.save(allmean_var_out_dir + f'{var} allmean.npy'.replace(':', "-"), allmeandata)
                if save2tiff:
                    if tifftypeflag == 'WRF': # WRF和CMAQ的tiff输出格式参数不一样
                        save_2tiff(allmean_var_out_dir + f'{var} allmean.tiff'.replace(':', "-"),np.flipud(allmeandata),WRF_ROW,WRF_COL,
                                   WRF_lonmin,WRF_lon_res,WRF_latmax,WRF_lat_res)
                    if tifftypeflag == 'CMAQ':
                        save_2tiff(allmean_var_out_dir + f'{var} allmean.tiff'.replace(':', "-"),np.flipud(allmeandata),CMAQ_ROW,CMAQ_COL,
                                   CMAQ_lonmin,CMAQ_lon_res,CMAQ_latmax,CMAQ_lat_res)

    if ifpicdif == True:
        for day in tqdm(range(0,1),desc="输出模拟时段总平均值 2次模拟差值："):
            allmean_out_dir = out_dir + "allmeandif\\"
            if os.path.exists(allmean_out_dir) is False: os.mkdir(allmean_out_dir)
            for var in output_vars:
                allmean_var_out_dir = allmean_out_dir + f"{var}\\"
                if os.path.exists(allmean_var_out_dir) is False: os.mkdir(allmean_var_out_dir)

                index = list(output_vars.keys()).index(var)
                if output_vars[var].ndim == 4 :
                    allmeandata1 = np.zeros(shape=(output_vars[var].shape[2], output_vars[var].shape[3]))
                    for hour in range(0,24*daycount):
                        allmeandata1 += output_vars[var][hour,0,:,:]
                    allmeandata1 /= 24*daycount
                    allmeandata2 = np.zeros(shape=(output_vars2[var].shape[2], output_vars2[var].shape[3]))
                    for hour in range(0, 24 * daycount):
                        allmeandata2 += output_vars2[var][hour, 0, :, :]
                    allmeandata2 /= 24 * daycount
                    allmeandatadif = allmeandata1 - allmeandata2
                if output_vars[var].ndim == 3:
                    allmeandata1 = np.zeros(shape=(output_vars[var].shape[1], output_vars[var].shape[2]))
                    for hour in range(0,24*daycount):
                        allmeandata1 += output_vars[var][hour,:,:]
                    allmeandata1 /= 24*daycount
                    allmeandata2 = np.zeros(shape=(output_vars2[var].shape[1], output_vars2[var].shape[2]))
                    for hour in range(0, 24 * daycount):
                        allmeandata2 += output_vars2[var][hour, :, :]
                    allmeandata2 /= 24 * daycount
                    allmeandatadif = allmeandata1 - allmeandata2

                if difresult_forceminmax[index] != False:
                    allmeandatadif[allmeandatadif >= difresult_forceminmax[index][1]] = difresult_forceminmax[index][1]
                    allmeandatadif[allmeandatadif <= difresult_forceminmax[index][0]] = difresult_forceminmax[index][0]

                if cbar_min_max == []:
                    cbarmax = np.ceil(np.max(allmeandatadif) / 10) * 10
                    cbarmin = np.floor(np.min(allmeandatadif) / 10) * 10
                else: cbarmin,cbarmax = cbar_min_max[0],cbar_min_max[1]
                # if cbarmin == cbarmax: continue # 说明数据为空 不输出
                if cbarmin == cbarmax: cbarmax,cbarmin = 1,0  # 数据为空但继续输出
                barmax = np.max(abs(allmeandatadif))
                cbarmax=barmax
                cbarmin=-barmax

                proj = ccrs.PlateCarree()  # 创建坐标系
                fig = plt.figure(figsize=fig_size,dpi=150)  # 创建页面
                ax = fig.subplots(1, 1, subplot_kw={'projection': proj})
                # 读取所有shp并绘制
                for shp in shpfiles:
                    shp_c = cfeat.ShapelyFeature(Reader(shp_files_dir + shp).geometries(), proj, edgecolor='k',
                                                 facecolor='none')
                    ax.add_feature(shp_c, lw=0.6, zorder=2)
                position = fig.add_axes(cbar_positions[1])  # colorbar位置
                if result_pic_type == 'contourf':
                    if allmeandatadif.shape != lon.shape: # CMAQ的数据行列相当于WRF少2个行列，要对latlon进行裁剪
                        lon_c = lon[0:lon.shape[0]-2,0:lon.shape[1]-2]
                        lat_c = lat[0:lat.shape[0] - 2, 0:lat.shape[1] - 2]
                        data_pic = ax.contourf(lon_c, lat_c, allmeandatadif, cmap='bwr',
                                               levels=np.linspace(cbarmin, cbarmax, 80))
                        LAT_vert = getArrayVertices(lat_c)  # 获取经纬度数组的四个顶点数据列表
                        LON_vert = getArrayVertices(lon_c)
                        ax.set_extent([LON_vert[0], LON_vert[1], LAT_vert[0], LAT_vert[2]])  # 显示范围
                        tifftypeflag = 'CMAQ'
                    else:
                        data_pic = ax.contourf(lon, lat, allmeandatadif, cmap='bwr',levels=np.linspace(cbarmin, cbarmax, 80))
                        LAT_vert = getArrayVertices(lat)  # 获取经纬度数组的四个顶点数据列表
                        LON_vert = getArrayVertices(lon)
                        ax.set_extent([LON_vert[0], LON_vert[1], LAT_vert[0], LAT_vert[2]])  # 显示范围
                        tifftypeflag = 'WRF'
                if result_pic_type == 'pcolormesh':
                    if allmeandatadif.shape != lon.shape: # CMAQ的数据行列相当于WRF少2个行列，要对latlon进行裁剪
                        lon_c = lon[0:lon.shape[0]-2,0:lon.shape[1]-2]
                        lat_c = lat[0:lat.shape[0] - 2, 0:lat.shape[1] - 2]
                        data_pic = ax.pcolormesh(lon_c, lat_c, allmeandatadif, cmap='bwr',vmin=cbarmin,vmax=cbarmax)
                        LAT_vert = getArrayVertices(lat_c)  # 获取经纬度数组的四个顶点数据列表
                        LON_vert = getArrayVertices(lon_c)
                        ax.set_extent([LON_vert[0], LON_vert[1], LAT_vert[0], LAT_vert[2]])  # 显示范围
                        tifftypeflag = 'CMAQ'
                    else:
                        data_pic = ax.pcolormesh(lon, lat, allmeandatadif, cmap='bwr',vmin=cbarmin,vmax=cbarmax)
                        LAT_vert = getArrayVertices(lat)  # 获取经纬度数组的四个顶点数据列表
                        LON_vert = getArrayVertices(lon)
                        ax.set_extent([LON_vert[0], LON_vert[1], LAT_vert[0], LAT_vert[2]])  # 显示范围
                        tifftypeflag = 'WRF'
                if manmual_extent != []: ax.set_extent(manmual_extent)  # 显示范围 当需要强制控制时修改
                cb_ticks = np.linspace(cbarmin, cbarmax, 7)
                cb = fig.colorbar(data_pic, cax=position, orientation=cbar_positions[0], extend='both',ticks=cb_ticks, format=target_vars_cbarticksFormat[index], fraction=0.2)
                cb.ax.tick_params(labelsize=17)  # 刻度字体大小
                cb.set_label(label=target_vars_name[index]+' '+f'(units: {target_vars_unit[index]})', fontsize=12)  # 设置colorbar的标签字体及其大小

                gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1.2, color='k', alpha=0.2,
                                  linestyle='--')
                if bgimage_dir_alpha[0] != "": # 绘制底图
                    ax.imshow(plt.imread(bgimage_dir_alpha[0]), origin='upper',zorder=1, transform=proj, extent=bgimage_dir_alpha[2], alpha=bgimage_dir_alpha[1])
                allmean_datas[var].append(allmeandatadif)  # 存放结果array
                plt.subplots_adjust(left=result_positions[0], right=result_positions[1], bottom=result_positions[2],
                                    top=result_positions[3], wspace=result_positions[4], hspace=result_positions[5])
                ax.set_title(f'{difpic_title}{target_vars_name[index]}模拟时段总均值变化量',fontsize=15,pad=30)
                gl.xlabel_style = {'size': 15}  # 设置经度标签字体大小
                gl.ylabel_style = {'size': 15}  # 设置纬度标签字体大小
                plt.savefig(allmean_var_out_dir+f'{var} allmeandif'.replace(':',"-"))
                plt.close()
                if save2npy: np.save(allmean_var_out_dir+f'{var} allmeandif.npy'.replace(':', "-"), allmeandatadif)
                if save2tiff:
                    if tifftypeflag == 'WRF': # WRF和CMAQ的tiff输出格式参数不一样
                        save_2tiff(allmean_var_out_dir+f'{var} allmeandif.tiff'.replace(':', "-"),np.flipud(allmeandatadif),WRF_ROW,WRF_COL,
                                   WRF_lonmin,WRF_lon_res,WRF_latmax,WRF_lat_res)
                    if tifftypeflag == 'CMAQ':
                        save_2tiff(allmean_var_out_dir+f'{var} allmeandif.tiff'.replace(':', "-"),np.flipud(allmeandatadif),CMAQ_ROW,CMAQ_COL,
                                   CMAQ_lonmin,CMAQ_lon_res,CMAQ_latmax,CMAQ_lat_res)




if __name__ == '__main__':


    pass

