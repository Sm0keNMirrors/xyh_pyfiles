"""
Author: Yaohan Xian
GitHub: https://github.com/Sm0keNMirrors
Last update: 2025年1月13日
"""
import os
import rasterio
import geopandas as gpd
import netCDF4 as nc
import numpy as np
import PseudoNetCDF as pnc
from rasterio.transform import from_origin
from rasterio.features import geometry_mask

def CMAQ_ISAMmask_generation(
    Workdir = "x/x/", #
    mask_shapes_dir = "x/x/", #
    GRIDCRO2D_dir = "",
    GRIDDESC_dir = "",
    GRIDNAME = "",
):
    """
    :param Workdir: arcpy处理过程存放中间文件的文件夹
    :param mask_shapes_dir: 制作mask的shp文件所在的文件夹，必须为只有一个闭合区域的shp，一个shp一个mask，注意shp文件名即为ISAM_REGION.nc的mask变量名，三个以内的大写字母简写
    :param GRIDCRO2D_dir: 用于制作mask的CMAQ某个domian的mcipGRIDCRO2D.nc文件
    :param GRIDDESC_dir: mcip输出的目标domian的GRIDDESC文件所在位置，用于制作ISAM_REGION.nc
    :param GRIDNAME: GRIDDESC中的domain名，pseudoNetCDF必须手动输入
    :return:
    """

    def GRIDCRO2D2temptiff(GRIDCRO2D_dir,outdir):
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

        data = nc.Dataset(GRIDCRO2D_dir)

        var_lon = np.array(data.variables['LON'][:][0][0])
        var_lat = np.array(data.variables['LAT'][:][0][0])
        ROW = var_lon.shape[0]
        COL = var_lon.shape[1]

        # 获取影像左下角和右下角坐标
        lonmin, latmax, lonmax, latmin = (var_lon.min(), var_lat.max(),
                                          var_lon.max(), var_lat.min())
        # print(lonmin, latmax, lonmax, latmin)

        # 计算分辨率
        len_lat = ROW
        len_lon = COL
        # print(len_lon, len_lat)
        lon_res = (lonmax - lonmin) / (len_lon - 1.0)
        lat_res = (latmax - latmin) / (len_lat - 1.0)

        # 获取数据
        TEMP = np.zeros((ROW, COL), dtype=float, order="c")
        save_2tiff(outdir + r"temp.tif", TEMP, ROW, COL, lonmin, lon_res, latmax, lat_res)

    def mask_the_temp(input_tiff,output_tiff,maskshp):
        # 加载tif文件
        tif_path = input_tiff
        output_tif_path = output_tiff
        shp_path = maskshp
        # 读取栅格数据
        with rasterio.open(tif_path) as src:
            profile = src.profile
            data = src.read(1)  # 读取第一波段数据
            transform = src.transform

        # 读取矢量数据
        shapefile = gpd.read_file(shp_path)
        # 创建掩码，设置在shapefile内的区域为 False（保留），外部区域为 True（遮罩）
        mask = geometry_mask([geom for geom in shapefile.geometry],
                             transform=transform,
                             invert=False,
                             out_shape=data.shape)
        # 创建新数组，初始化为0
        new_data = np.zeros(data.shape, dtype=np.uint8)
        # 将掩码为True的区域设置为1
        new_data[~mask] = 1
        # 更新栅格文件的配置
        profile.update(dtype=rasterio.uint8, count=1)
        # 保存新的栅格文件
        with rasterio.open(output_tif_path, 'w', **profile) as dst:
            dst.write(new_data, 1)
        print(f"{maskshp} mask计算完成")


    Regins_tiff_dir = Workdir+r"Regionstifs\\"  # 溯源区域tiff所在dir，为此过程的最终生成的结果

    print("① GRIDCRO2D转换为框架temptiff...")
    GRIDCRO2D2temptiff(GRIDCRO2D_dir,Workdir)
    print("① GRIDCRO2D转换为框架temptiff 完成")

    mask_shapes_pre = os.listdir(mask_shapes_dir)
    mask_shapes = []
    for f in mask_shapes_pre:
        if f.split('.')[1] == 'shp':
            mask_shapes.append(f)

    print("② 将shpdir的区域tif与temp进行mask计算...")
    rnames = []
    if os.path.exists(Regins_tiff_dir) is False: os.mkdir(Regins_tiff_dir)
    for shpf in mask_shapes:
        rname = shpf.split('.')[0]
        rnames.append(rname)
        mask_the_temp(Workdir+"temp.tif",Regins_tiff_dir+rname+'.tiff',mask_shapes_dir+shpf)
    print("② 将shpdir的区域tif与temp进行mask计算 完成")

    print("③ 使用PseudoNetCDF读取GRIDDESC制作ISAM_REGION.nc...")
    gf = pnc.pncopen(
        GRIDDESC_dir,
        GDNAM=GRIDNAME,
        format="griddesc",
        SDATE=int(2025001), # ISAMREGION无时间
        TSTEP=10000,
        withcf=False,
    )
    gf.updatetflag(overwrite=True)
    ISAM_REGION = gf.sliceDimensions(TSTEP=[0]*1)  # CMAQnc文件框架
    for r in rnames:
        region_var = ISAM_REGION.createVariable(r, "f", ("TSTEP", "LAY", "ROW", "COL"))
        region_var.setncatts(
            dict(units='fraction', long_name=r, var_desc=r + ' fractional area per grid cell'))
        with rasterio.open(Regins_tiff_dir+r+'.tiff') as target_data:
            data = target_data.read(1)
            data[data != 0] = 1  # mask为1 其他为0
            data[data == 0] = 0
        data = np.flipud(data)  # 经CMAQ输出发现，MASK图像要翻转一下
        region_var[:] = data
    del ISAM_REGION.variables['DUMMY']
    # ISAM_REGION的tstep必须为0，使CMAQ认为其为time-independent，否则报错
    delattr(ISAM_REGION, 'VAR-LIST')
    ISAM_REGION.updatemeta()
    ISAM_REGION.variables['TFLAG'][:] = 0
    ISAM_REGION.SDATE = -635
    savedf = ISAM_REGION.save(Workdir+'ISAM_REGION.nc', verbose=0, complevel=1)
    savedf.close()
    print("③ 使用PseudoNetCDF读取GRIDDESC制作ISAM_REGION.nc 完成")



if __name__ == "__main__":
    CMAQ_ISAMmask_generation(
        Workdir=r"E:\xyhfiles_runtest\\",
        mask_shapes_dir=r"E:\ArcGISFiles\chengdu_county\\",
        GRIDCRO2D_dir = r"E:\WCAS_serverfiles\GRIDCRO2D_2024333.nc",
        GRIDDESC_dir = r"E:\WCAS_serverfiles\GRIDDESC",
        GRIDNAME = 'CDsvSA_d03',
    )
    pass