import os

from PIL import Image



def pics2gif(
    pics_dir = "", # 需要制成GIF的所有png图片所在的路径，一般都保持类型相同
    outfile_dir = "",
    duration = 200,
):
    pic_files_ = os.listdir(pics_dir)
    pic_files = [x for x in pic_files_ if x[-4:] == '.png']
    print(pics_dir+pic_files[0])
    with Image.open(pics_dir+pic_files[0]) as im:width, height = im.size # 打开一个图像获得宽高
    # 创建一个新的 GIF 图像，指定模式和尺寸
    gif_image = Image.new('RGBA', (width, height))

    pic_files_dir = [pics_dir+x for x in pic_files]
    pic_files_opens = [Image.open(x) for x in pic_files_dir]
    print(pic_files_dir[1:])

    # 将每个图像逐一添加到 GIF 图像中，并指定每一帧的显示时间
    for image in pic_files_dir:
        with Image.open(image) as im:
            gif_image.paste(im)

    # 保存 GIF 图像，并指定每一帧的显示时间为200毫秒（duration）
    gif_image.save(outfile_dir, save_all=True, append_images=pic_files_opens[1:], duration=duration,loop=0)




if __name__ == "__main__":

    pass