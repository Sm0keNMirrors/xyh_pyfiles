"""
Author: Yaohan Xian
GitHub: https://github.com/Sm0keNMirrors
Last update: 2024年12月25日
"""
import multiprocessing
import os
import datetime
import PseudoNetCDF as pnc
import netCDF4 as nc
import numpy as np
import pandas as pd
from tqdm import tqdm


def Multi_CombineVars(x,PM25_DIAGpara,to_combine_files_list,CCTM_dir,CCTM_day,all_vars_lists,combined_tagged_subs,DIAG_files_list):
    """
    用于多线程的函数
    :return:
    """
    if x not in PM25_DIAGpara:  # 合并一般物质
        for i in to_combine_files_list:
            CONCf = nc.Dataset(CCTM_dir + i)
            day = to_combine_files_list.index(i) + 1  # 合并到第几天
            n_next = to_combine_files_list.index(i) + 1  # 下一天文件的index
            if n_next >= CCTM_day:
                all_vars_lists.update({x: data_now})  # 写入合并后的数据
                break
            CONCf_next = nc.Dataset(CCTM_dir + to_combine_files_list[n_next])

            data_next = np.array(CONCf_next.variables[x][:])
            if to_combine_files_list.index(i) == 0:
                data_now = np.array(CONCf.variables[x][:])  # 首次输入文件
                data_now = np.concatenate((data_now, data_next), axis=0)  # ACONC没有多的1h，不用删除处理

            if to_combine_files_list.index(i) != 0:
                data_now = np.concatenate((data_now, data_next), axis=0)
    if 'PM25' in combined_tagged_subs:
        if x in PM25_DIAGpara:  # 合并PM2.5计算参数，文件不一样
            for i in DIAG_files_list:
                CONCf = nc.Dataset(CCTM_dir + i)
                day = DIAG_files_list.index(i) + 1  # 合并到第几天
                n_next = DIAG_files_list.index(i) + 1  # 下一天文件的index
                if n_next >= CCTM_day:
                    all_vars_lists.update({x: data_now})  # 写入合并后的数据
                    break
                CONCf_next = nc.Dataset(CCTM_dir + DIAG_files_list[n_next])

                data_next = np.array(CONCf_next.variables[x][:])

                if DIAG_files_list.index(i) == 0:
                    data_now = np.array(CONCf.variables[x][:])  # 首次输入文件
                    data_now = np.concatenate((data_now, data_next), axis=0)  # ACONC没有多的1h，不用删除处理

                if DIAG_files_list.index(i) != 0:
                    data_now = np.concatenate((data_now, data_next), axis=0)

def CMAQ_CombineWithISAM(
    start_date = "YYYY-MM-DD" ,#combine文件开始日期
    CCTM_dir = "", # 需要合并的已经分类的CCTM文件夹路径
    Combine_file_outdir = "",  # 合并后的文件名称 完整路径和拓展名
    GRIDDECfile_dir = "", # 嵌套信息文件GRIDDESC位置
    GRIDNAME = "",# GRIDDESC中的gridname
    CMAQISAM_version = "", #版本不同决定了PM2.5的合并机制，有"v54" "v53"两种
    combined_tagged_subs = [], # 被标记的需要combine的物质的变量名称,'O3','PM25'等，其中PM25输入后会根据标记物质计算，而其他则直接由其变量名_tag合并 不需要计算
    cores = 12,# 多线程运行核心数
):
    # 5.3版本CMAQISAM合并PM2.5的物种，物种不完全，仅能标记部分
    substance_PM25_CMAQv53 = ['ASO4I','ANO3I','ANH4I','ANAI','ACLI','AECI','AOTHRI',
                         'APOCI','APNCOMI',

                         'ASO4J','ANO3J','ANH4J','ANAJ','ACLJ','AECJ',
                         'AOTHRJ','AFEJ','ASIJ','ATIJ','ACAJ',
                         'AMGJ','AMNJ','AALJ','AKJ','APOCJ','APNCOMJ',] # 合并的PM2.5组分
     # 5.4版本CMAQISAM合并PM2.5的物种，物种完全，与普通CMAQ模拟的物种一致
    substance_PM25_CMAQv54 = ['ASO4I' ,'ANO3I' ,'ANH4I' ,'ANAI' ,'ACLI' ,'AECI' ,'AOTHRI'  ,# 合并的PM2.5组分
                         'APOCI' ,'APNCOMI' ,'ALVOO1I' ,'ALVOO2I' ,'ASVOO1I' ,'ASVOO2I',
                         'ALVPO1I' ,'ASVPO1I' ,'ASVPO2I',

                         'ASO4J' ,'ANO3J' ,'ANH4J' ,'ANAJ' ,'ACLJ' ,'AXYL1J' ,'AXYL2J',
                         'AXYL3J' ,'ATOL1J' ,'ATOL2J' ,'ATOL3J' ,'ABNZ1J' ,'ABNZ2J',
                         'ABNZ3J' ,'AISO1J' ,'AISO2J' ,'AISO3J' ,'ATRP1J' ,'ATRP2J',
                         'ASQTJ' ,'AALK1J' ,'AALK2J' ,'APAH1J' ,'APAH2J' ,'APAH3J',
                         'AORGCJ' ,'AOLGBJ' ,'AOLGAJ' ,'ALVOO1J' ,'ALVOO2J' ,'ASVOO1J',
                         'ASVOO2J' ,'ASVOO3J' ,'APCSOJ' ,'ALVPO1J' ,'ASVPO1J' ,'ASVPO2J',
                         'ASVPO3J' ,'AIVPO1J' ,'AOTHRJ' ,'AFEJ' ,'ASIJ' ,'ATIJ' ,'ACAJ',
                         'AMGJ' ,'AMNJ' ,'AALJ' ,'AKJ',

                         'ASOIL' ,'ACORS' ,'ASEACAT' ,'ACLK' ,'ASO4K' ,'ANO3K' ,'ANH4K',]
    # 所有需要Combine的总物质，即 带标签的OPM各模态 和 带标签的O3等物种
    if 'PM25' in combined_tagged_subs:
        combined_tagged_subs2 = combined_tagged_subs.copy() # 必须copy 直接复制会导致2个同时被remove
        combined_tagged_subs2.remove('PM25')
        if CMAQISAM_version == "v53":
            Combine_substance = substance_PM25_CMAQv53+combined_tagged_subs2
            PMDIAGfilename = 'APMDIAG' # PM诊断文件名，53 54有所区别
            PM25_DIAGpara = ['PM25AT', 'PM25AC', 'PM25CO']  # 计算PM2.5浓度的诊断参数
        elif CMAQISAM_version == "v54":
            Combine_substance = substance_PM25_CMAQv54+combined_tagged_subs2
            PMDIAGfilename = 'AELMO' # PM诊断文件名，53 54有所区别
            PM25_DIAGpara = ['FPM25AIT', 'FPM25ACC', 'FPM25COR']
        else:
            print("若需要combinePM25，请输入CMAQISAM版本")
            return 1
    else:
        PM25_DIAGpara = []
        Combine_substance = combined_tagged_subs

    # 准备好combine的文件群并查找变量信息
    combine_file_namestr = 'SAACONC'  # 合并的CCTM文件名称类型，CCTM_ 之后的2个_隔开的名称相加
    SA_CCTM_file_list = os.listdir(CCTM_dir)  # CCTM
    to_combine_files_list = []
    DIAG_files_list = []
    for i in SA_CCTM_file_list:  # 找出对应文件列表
        if i[-3:] == '.nc':
            if i.split("_")[1] + i.split("_")[2] == combine_file_namestr:
                to_combine_files_list.append(i)
            if 'PM25' in combined_tagged_subs:
                if i.split("_")[1] == PMDIAGfilename:
                    DIAG_files_list.append(i)
    Combinevars = []
    ISAM_tags = []
    CONCf_pre = nc.Dataset(CCTM_dir + to_combine_files_list[0])  # 先打开一个文件，获取数据信息
    all_vars_pre = list(CONCf_pre.variables.keys())
    for x in all_vars_pre: # 从所有变量中找到需要combine的PM组分
        subname = x.split('_')[0] # 去掉ISAMtag
        if subname in Combine_substance: # 需要combine的带tag变量
            Combinevars.append(x)
        if len(x.split('_')) == 2:
            tag = x.split('_')[1]
            if '_'+tag not in ISAM_tags:
                ISAM_tags.append('_'+tag)
    if 'PM25' in combined_tagged_subs:
        for x in PM25_DIAGpara: # 把计算参数加上
            Combinevars.append(x)
            
    # 进行文件组各类物种时间序列合并的过程：
    all_vars_lists = {}
    CCTM_day = len(to_combine_files_list)
    daynum = CCTM_day  # 模拟的天数
    for x in tqdm(Combinevars,desc="SAACONC被tag变量数据读取合并..."):
        if x not in PM25_DIAGpara:  # 合并一般物质
            for i in to_combine_files_list:
                CONCf = nc.Dataset(CCTM_dir + i)
                day = to_combine_files_list.index(i) + 1  # 合并到第几天
                n_next = to_combine_files_list.index(i) + 1  # 下一天文件的index
                if n_next >= CCTM_day:
                    all_vars_lists.update({x: data_now})  # 写入合并后的数据
                    CONCf.close()
                    break
                CONCf_next = nc.Dataset(CCTM_dir + to_combine_files_list[n_next])

                data_next = np.array(CONCf_next.variables[x][:])
                if to_combine_files_list.index(i) == 0:
                    data_now = np.array(CONCf.variables[x][:])  # 首次输入文件
                    data_now = np.concatenate((data_now, data_next), axis=0)  # ACONC没有多的1h，不用删除处理

                if to_combine_files_list.index(i) != 0:
                    data_now = np.concatenate((data_now, data_next), axis=0)
                CONCf_next.close()
                CONCf.close()
        if 'PM25' in combined_tagged_subs:
            if x in PM25_DIAGpara:  # 合并PM2.5计算参数，文件不一样
                for i in DIAG_files_list:
                    CONCf = nc.Dataset(CCTM_dir + i)
                    day = DIAG_files_list.index(i) + 1  # 合并到第几天
                    n_next = DIAG_files_list.index(i) + 1  # 下一天文件的index
                    if n_next >= CCTM_day:
                        all_vars_lists.update({x: data_now})  # 写入合并后的数据
                        CONCf.close()
                        break
                    CONCf_next = nc.Dataset(CCTM_dir + DIAG_files_list[n_next])

                    data_next = np.array(CONCf_next.variables[x][:])

                    if DIAG_files_list.index(i) == 0:
                        data_now = np.array(CONCf.variables[x][:])  # 首次输入文件
                        data_now = np.concatenate((data_now, data_next), axis=0)  # ACONC没有多的1h，不用删除处理

                    if DIAG_files_list.index(i) != 0:
                        data_now = np.concatenate((data_now, data_next), axis=0)
                    CONCf_next.close()
                    CONCf.close()

    # 进行文件组各类物种时间序列合并的过程（多线程）：
    # pool = multiprocessing.Pool(cores)
    # arg_pool = []
    # for x in tqdm(Combinevars,desc="SAACONC被tag变量数据读取合并..."):
    #     arg = (x,PM25_DIAGpara,to_combine_files_list,CCTM_dir,CCTM_day,all_vars_lists,combined_tagged_subs,DIAG_files_list)
    #     arg_pool.append(arg)
    # results = pool.starmap(Multi_CombineVars, arg_pool)


    # 以GRIDDESC创建CMAQnc文件作为combine的nc 并计算总PM25
    yyyyjjj = datetime.datetime.strftime(pd.to_datetime(str(start_date)), "%Y%j")
    gf = pnc.pncopen(
        GRIDDECfile_dir,
        GDNAM=GRIDNAME,
        format="griddesc",
        SDATE=int(yyyyjjj),
        TSTEP=10000,
        withcf=False,
    )
    gf.updatetflag(overwrite=True)
    COMBINE_file = gf.sliceDimensions(TSTEP=[0] * 24 * daynum) # CMAQnc文件框架
    for i in tqdm(all_vars_lists,desc="导入combine后的nc文件..."):
        emission_specie_var = COMBINE_file.createVariable(i, "f", ("TSTEP", "LAY", "ROW", "COL"))
        if 'PM25' in combined_tagged_subs:
            if i in PM25_DIAGpara:
                emission_specie_var.setncatts(
                    dict(units=' ', long_name=i, var_desc=i)) # 计算系数无单位
            else:
                emission_specie_var.setncatts(
                    dict(units='ug/m-3 or ppvm', long_name=i, var_desc=i))  # 计算系数无单位
        else:
            emission_specie_var.setncatts(
                dict(units='ug/m-3 or ppvm', long_name=i, var_desc=i))  # 计算系数无单位
        emission_specie_var[:,:,:,:] = all_vars_lists[i][:, :, :, :]

    #若有PM25，进一步根据版本机制计算TAG的总PM
    if "PM25" in combined_tagged_subs:
        PM25_data_lists = []
        for i in ISAM_tags:
            PM25_data_lists.append('PM25' + i)
        for x in PM25_data_lists:  # 每个tag
            index = PM25_data_lists.index(x)
            tag_now = ISAM_tags[index]  # 现在处理的tag数据
            AJ_tag, AK_tag, AI_tag = [], [], []  # 创建每个模态类的判断池
            for m in Combinevars:
                subname = m.split('_')[0]  # 去掉ISAMtag
                if subname[0] == 'A' and subname[-1] == 'J':
                    AJ_tag.append(subname + tag_now)
                if CMAQISAM_version == "v54": # 5.4的ISAM才会有K模态物质
                    if (subname[0] == 'A' and subname[-1] == 'K') or (subname in ['ASOIL', 'ACORS', 'ASEACAT']):
                        AK_tag.append(subname + tag_now)
                if subname[0] == 'A' and subname[-1] == 'I':
                    AI_tag.append(subname + tag_now)

            data_shape = np.array(COMBINE_file.variables[Combinevars[0]][:]).shape  # 获得数据shape来存放all
            PM25_data = np.zeros(data_shape, 'float64')  # PM25数组初始化
            for t in tqdm(range(0, data_shape[0]),desc="PM25" + tag_now + "每小时浓度计算..."):  # 每个小时的数据相加
                if CMAQISAM_version == "v53":
                    PM25_AT_data = np.array(COMBINE_file.variables['PM25AT'][:])
                    PM25_AC_data = np.array(COMBINE_file.variables['PM25AC'][:])
                    PM25_CO_data = np.array(COMBINE_file.variables['PM25CO'][:])
                if CMAQISAM_version == "v54":
                    PM25_AT_data = np.array(COMBINE_file.variables['FPM25AIT'][:])
                    PM25_AC_data = np.array(COMBINE_file.variables['FPM25ACC'][:])
                    PM25_CO_data = np.array(COMBINE_file.variables['FPM25COR'][:])
                PM25_AI_data = np.zeros(data_shape, 'float64')
                PM25_AJ_data = np.zeros(data_shape, 'float64')  # PM25数组初始化
                PM25_AK_data = np.zeros(data_shape, 'float64')
                for i in all_vars_lists:
                    if i not in PM25_DIAGpara:
                        if i in AI_tag:
                            PM_sub_data = np.array(COMBINE_file.variables[i][:])
                            PM25_AI_data[t, :, :, :] += PM_sub_data[t, :, :, :]
                        if i in AJ_tag:
                            PM_sub_data = np.array(COMBINE_file.variables[i][:])
                            PM25_AJ_data[t, :, :, :] += PM_sub_data[t, :, :, :]
                        if CMAQISAM_version == "v54":
                            if i in AK_tag:
                                PM_sub_data = np.array(COMBINE_file.variables[i][:])
                                PM25_AK_data[t, :, :, :] += PM_sub_data[t, :, :, :]
                if CMAQISAM_version == "v54":
                    PM25_data[t, :, :, :] = PM25_AI_data[t, :, :, :] * PM25_AT_data[t, :, :, :] + \
                                            PM25_AJ_data[t, :, :, :] * PM25_AC_data[t, :, :, :] + \
                                            PM25_AK_data[t, :, :, :] * PM25_CO_data[t, :, :, :]
                if CMAQISAM_version == "v53":
                    PM25_data[t, :, :, :] = PM25_AI_data[t, :, :, :] * PM25_AT_data[t, :, :, :] + \
                                            PM25_AJ_data[t, :, :, :] * PM25_AC_data[t, :, :, :]

            PM25_data_name = 'PM25' + tag_now
            data_id = COMBINE_file.createVariable(PM25_data_name, 'f', ("TSTEP", "LAY", "ROW", "COL"))  # 必须指明变量类型
            data_id.long_name = PM25_data_name + (16 - len(PM25_data_name)) * ' '
            data_id.units = 'ug/m-3'
            data_id.var_desc = PM25_data_name
            data_id[:] = PM25_data

    # 生成Combine nc 文件
    del COMBINE_file.variables["DUMMY"]
    COMBINE_file.updatetflag(tstep=10000, overwrite=True)
    delattr(COMBINE_file, "VAR-LIST")
    COMBINE_file.updatemeta()
    output_name = Combine_file_outdir
    COMBINE_file.save(output_name, format="NETCDF3_CLASSIC")
    COMBINE_file.close()


if __name__ == "__main__":

    # CMAQ_CombineWithISAM(
    #     start_date="2024-11-30",
    #     CCTM_dir=r"E:\WCAS_serverfiles\cctm\cctm_emo3\\",
    #     Combine_file_outdir=r"E:\WCAS_serverfiles\cctm\\cctmcombine_emo3.nc",
    #     GRIDDECfile_dir=r"E:\WCAS_serverfiles\GRIDDESC",
    #     GRIDNAME="CDsvSA_d03",
    #     CMAQISAM_version="v53",
    #     combined_tagged_subs=["O3"],
    # )

    # CMAQ_CombineWithISAM(
    #     start_date="2024-11-30",
    #     CCTM_dir=r"E:\WCAS_serverfiles\cctm\cctm_empm25\\",
    #     Combine_file_outdir=r"E:\WCAS_serverfiles\cctm\\cctmcombine_empm25.nc",
    #     GRIDDECfile_dir=r"E:\WCAS_serverfiles\GRIDDESC",
    #     GRIDNAME="CDsvSA_d03",
    #     CMAQISAM_version="v53",
    #     combined_tagged_subs=["PM25"],
    # )

    # CMAQ_CombineWithISAM(
    #     start_date="2024-11-30",
    #     CCTM_dir=r"E:\WCAS_serverfiles\cctm\cctm_repm25\\",
    #     Combine_file_outdir=r"E:\WCAS_serverfiles\cctm\\cctmcombine_repm25.nc",
    #     GRIDDECfile_dir=r"E:\WCAS_serverfiles\GRIDDESC",
    #     GRIDNAME="CDsvSA_d03",
    #     CMAQISAM_version="v53",
    #     combined_tagged_subs=["PM25"],
    # )

    CMAQ_CombineWithISAM(
        start_date="2024-11-30",
        CCTM_dir=r"E:\WCAS_serverfiles\cctm\cctm_reo3\\",
        Combine_file_outdir=r"E:\WCAS_serverfiles\cctm\\cctmcombine_reo3.nc",
        GRIDDECfile_dir=r"E:\WCAS_serverfiles\GRIDDESC",
        GRIDNAME="CDsvSA_d03",
        CMAQISAM_version="v53",
        combined_tagged_subs=["O3"],
    )

    pass