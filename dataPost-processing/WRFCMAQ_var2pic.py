import os
import matplotlib
import numpy as np
import netCDF4 as nc
from tqdm import tqdm
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib import colors
import cartopy.feature as cfeat
from cartopy.io.shapereader import Reader
from datetime import datetime, timedelta

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
    metvar_now = np.zeros(data_shape, dtype=np.float64)
    metvar_next = np.zeros(data_shape, dtype=np.float64)
    WRFoutf.close()

    WRF_file_list_time_list = list(WRF_file_list_time.keys())  # 将WRF风场数据合并为1个文件，用来求8h平均
    for n in tqdm(WRF_file_list_time_list,desc=f'读取并合并WRF输出参数{metvar}:'):
        n_num = WRF_file_list_time_list.index(n)
        if n_num + 1 >= len(WRF_file_list_time_list):
            metvar_combine = metvar_now
            break
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
    input_vars ={'WRF':[],'CMAQ':[]}, # {'WRF':[],'CMAQ':[]}读取的变量，用于输出或者计算最终输出变量，来源模型:变量组
    target_vars = {'direct':[],'indirect':[]}, # 输出的变量，direct：直接读取后输出，indirect：计算后输出，计算方式自行在函数中添加
    target_vars_cmap = [], # 输出变量的颜色条类型，依照target_vars顺序输入，包括常用的：jet、rainbow、pollution
    target_vars_unit = [], # 输出变量的单位，依照target_vars顺序输入，有μg/m$^{3}$，℃，m等
    Molar_mass ={'target_var_name':0}, # 输出变量若需要进行ppm转换的摩尔质量：输出变量名：摩尔质量大小
    result_data_types=[], # 输出结果图的包含哪些时平均结果，包括hourly daily monthly monthly_max8h allmean
    fig_size = (5,5), # 输出图像范围的大致长宽程度
    result_pic_type='', #输出结果图类型contourf, pcolormesh
    out_dir = "",
    suffix = "",
):
    if os.path.exists(out_dir) is False: os.mkdir(out_dir)

    if CMAQcombine_file_dir != "":CMAQf = nc.Dataset(CMAQcombine_file_dir)
    if GRIDCRO2D_file_dir != "": GRIDCRO2D = nc.Dataset(GRIDCRO2D_file_dir, 'r')
    if WRFout_files_dir != "": WRFoutfiles = os.listdir(WRFout_files_dir)
    if WRFout_files_winddata_dir != "": WRFoutfiles_winddata = os.listdir(WRFout_files_winddata_dir)
    if shp_files_dir != "": shpfiles = [x for x in os.listdir(shp_files_dir) if x.split('.')[1] == 'shp' and len(x.split('.')) == 2]

    if WRFout_files_dir != "":
        WRFoutf = nc.Dataset(WRFout_files_dir + WRFoutfiles[0], "r")  # 获得风场数据的grid
        lon = WRFoutf.variables['XLONG'][:][0]
        lat = WRFoutf.variables['XLAT'][:][0]
        ROW = np.array(WRFoutf.variables["XLAT"]).shape[1]  # 获得数据格式shape1
        COL = np.array(WRFoutf.variables["XLAT"]).shape[2]  # 获得数据格式shape2
        WRFoutf.close()
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

        winddata = [wind_XLAT, wind_XLONG, wind_WSV, wind_WSU, wind_WD]


    cmapdict = ['white', '#75bbfd', 'green', 'yellow', 'red', 'maroon']  # 自定义colorbar的颜色
    cmap_pollution = colors.LinearSegmentedColormap.from_list("name", cmapdict)
    target_vars_cmap_ = [cmap_pollution if x == 'pollution' else x for x in target_vars_cmap]


    input_vars_get = {}
    for model in input_vars:
        if model == 'WRF':
            for var in input_vars[model]:
                input_vars_get.update({var:getWRFvarsCombined(WRFout_files_dir,var)})
        if model == 'CMAQ':
            for var in input_vars[model]:
                input_vars_get.update({var:CMAQf[var][CMAQcombine_inithour:CMAQcombine_inithour+24*daycount]})

    output_vars = {}
    for type in target_vars:
        if type == 'direct':
            for var in target_vars[type]:
                output_vars.update({var:input_vars_get[var]})
    for type in target_vars:
        if type == 'indirect':
            for var in target_vars[type]:
                # 输入对应indirect的变量的计算方法，通过确定的input变量来计算,如RH，ISAM总浓度等
                # =====================😼😼😼😼😼😼😼😼😼########😼😼😼😼😼😼😼😼😼😼😼😼😼=====================




                if var == 'RH':
                    vardata = 0.236 * input_vars_get['PSFC'] * input_vars_get['Q2'] * np.exp((17.67 * input_vars_get['T2']) / (input_vars_get['T2'] - 273.15 - 29.65)) ** (-1)
                    output_vars.update({var: vardata})




                # =====================😼😼😼😼😼😼😼😼😼########😼😼😼😼😼😼😼😼😼😼😼😼😼=====================

    # 绘制图片基本框架
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号


    # hourlydata = np.zeros((1, 1, ROW, COL), dtype=np.float64)
    for result_data_type in result_data_types:
        start_date_o = datetime.strptime(start_date, '%Y-%m-%d')
        if result_data_type == 'hourly':
            for hour in tqdm(range(0,daycount*24),desc="输出逐小时结果图："):
                hourly_out_dir = out_dir + "hourly\\"
                if os.path.exists(hourly_out_dir) is False: os.mkdir(hourly_out_dir)
                for var in output_vars:
                    hourly_var_out_dir = hourly_out_dir + f"{var}\\"
                    if os.path.exists(hourly_var_out_dir) is False: os.mkdir(hourly_var_out_dir)

                    date_now = start_date_o + timedelta(hours=hour)
                    index = list(output_vars.keys()).index(var)
                    if output_vars[var].ndim == 4 : hourlydata = output_vars[var][hour,0,:,:]
                    if output_vars[var].ndim == 3: hourlydata = output_vars[var][hour, :, :]
                    cbarmax = np.ceil(np.max(hourlydata) / 10) * 10
                    cbarmin = np.floor(np.min(hourlydata) / 10) * 10
                    # if cbarmin == cbarmax: continue # 说明数据为空 不输出
                    if cbarmin == cbarmax: cbarmax,cbarmin = 1,0  # 数据为空但继续输出

                    proj = ccrs.PlateCarree()  # 创建坐标系
                    fig = plt.figure(figsize=fig_size,dpi=150)  # 创建页面
                    ax = fig.subplots(1, 1, subplot_kw={'projection': proj})
                    LAT_vert = getArrayVertices(lat)  # 获取经纬度数组的四个顶点数据列表
                    LON_vert = getArrayVertices(lon)
                    # 读取所有shp并绘制
                    for shp in shpfiles:
                        shp_c = cfeat.ShapelyFeature(Reader(shp_files_dir + shp).geometries(), proj, edgecolor='k',
                                                     facecolor='none')
                        ax.add_feature(shp_c, lw=0.6, zorder=2)
                    ax.set_extent([LON_vert[0], LON_vert[1], LAT_vert[0], LAT_vert[2]])  # 显示范围
                    # position = fig.add_axes([0.25, 0.15, 0.5, 0.025])  # 图标位置
                    # position = fig.add_axes([0.85, 0.1, 0.025, 0.7])  # 图标位置
                    position = fig.add_axes([0.1, 0.1, 0.8, 0.025])  # colorbar位置
                    if result_pic_type == 'contourf':
                        data_pic = ax.contourf(lon, lat, hourlydata, cmap=target_vars_cmap_[index],levels=np.linspace(cbarmin, cbarmax, 80))
                    if result_pic_type == 'pcolormesh':
                        data_pic = ax.pcolormesh(lon, lat, hourlydata, cmap=target_vars_cmap_[index])
                    cb_ticks = np.linspace(cbarmin, cbarmax, 7)
                    cb = fig.colorbar(data_pic, cax=position, orientation='horizontal', extend='both',ticks=cb_ticks, format='%i', fraction=0.2)
                    cb.ax.tick_params(labelsize=17)  # 刻度字体大小
                    cb.set_label(label=var+' concentration'+f'(units: {target_vars_unit[index]})', fontsize=12)  # 设置colorbar的标签字体及其大小
                    if WRFout_files_winddata_dir != "":
                        ax.quiver(winddata[0][0], winddata[1][0], winddata[2][0], winddata[3][0], transform=proj, scale=8,
                              scale_units='inches', width=0.0015)
                    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1.2, color='k', alpha=0.2,
                                      linestyle='--')
                    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.2, top=0.8, wspace=0.2, hspace=0.2)
                    ax.set_title(f'Hourly {var} at {str(date_now)}',fontsize=15,pad=30)
                    gl.xlabel_style = {'size': 15}  # 设置经度标签字体大小
                    gl.ylabel_style = {'size': 15}  # 设置纬度标签字体大小
                    plt.savefig(hourly_var_out_dir+f'{var} {str(date_now)}'.replace(':',"-"))
                    plt.close()



if __name__ == '__main__':
    # WRFCMAQ_var2pic(
    #     start_date='2022-08-01',
    #     daycount=31,
    #     WRFout_files_dir='E:\CMAQdata_chengdu202208\WRFd03_2\\',
    #     # CMAQcombine_file_dir='',
    #     WRFout_files_winddata_dir='E:\CMAQdata_chengdu202208\WRFd02\\',
    #     shp_files_dir='E:\CMAQdata_chengdu202208\shps\\',
    #     input_vars={'WRF':['PSFC','Q2','T2']},
    #     target_vars={'direct':['T2'],'indirect':['RH']},
    #     target_vars_cmap=['jet','ocean'],
    #     target_vars_unit=['℃','%'],
    #     result_pic_type='contourf',
    #     result_data_types=['hourly'],
    #     out_dir=r'E:\Emission_update\test_var2pic\\',
    #     suffix='test'
    # )

    WRFCMAQ_var2pic(
        start_date='2022-08-18',
        daycount=5,
        WRFout_files_dir=r'E:\LQSlanduse\2020\wrfout\\',
        CMAQcombine_file_dir=r"E:\LQSlanduse\2020\COMBINE_ACONC_v532_gcc_20220818_202208.nc",
        GRIDCRO2D_file_dir=r"E:\LQSlanduse\GRIDCRO2D_2022230.nc",
        WRFout_files_winddata_dir="",
        shp_files_dir='E:\CMAQdata_chengdu202208\shps\\',
        input_vars={'WRF':['PBLH'],'CMAQ':['O3']},
        target_vars={'direct':['PBLH']},
        target_vars_cmap=['jet','pollution'],
        target_vars_unit=['m','ug/m3'],
        Molar_mass={'O3':48},
        result_pic_type='contourf',
        result_data_types=['hourly'],
        fig_size = (6, 10),
        out_dir=r'E:\LQSlanduse\2020\\picout\\',
        suffix='2020landuse'
    )
    pass

