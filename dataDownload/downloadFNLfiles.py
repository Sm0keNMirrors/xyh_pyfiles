import ssl
import datetime
from urllib.request import build_opener
from tqdm import tqdm
import os
import subprocess

def check_command(command):
    try:
        # try if command available
        subprocess.run([command, '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print(f"The command '{command}' is available.")
        return True
    except subprocess.CalledProcessError:
        print(f"The command '{command}' is not available.")
        return False
    except FileNotFoundError:
        print(f"The command '{command}' is not found.")
        return False

def downLoadFNLfiles(start_date, end_date):

    result = []
    date_start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    date_end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    result.append(date_start.strftime('%Y-%m-%d'))
    while date_start < date_end:
        date_start += datetime.timedelta(days=1)
        result.append(date_start.strftime('%Y-%m-%d'))

    date_list = result
    hour_list = ['00', '06', '12', '18']
    filelist = []
    ssl._create_default_https_context = ssl._create_unverified_context  # To solve the SLL problem encontered when download grib1 files in Linux system.

    for date in date_list:
        year = date.split('-')[0]
        month = date.split('-')[1]
        day = date.split('-')[2]
        if int(year) <= 2007:  # Before 2007.12.06-06, fnlfiles are grib1 type.
            baseURL = 'https://data.rda.ucar.edu/ds083.2/grib1/'
            filelist_onlymonth = baseURL + str(year) + '/' + str(year) + '.' + month
            if month == '12' and day == '06':  # Before 2007.12.06-06, fnlfiles are grib1 type.
                for hour in hour_list[0:2]:
                    filename = filelist_onlymonth + '/' + 'fnl_' + str(year) + month + day + '_' + hour + '_00.grib1'
                    filelist.append(filename)
                for hour in hour_list[2:4]:  # After 2007.12.06-06, fnlfiles are grib2 type.
                    baseURL = 'https://data.rda.ucar.edu/ds083.2/grib2/'
                    filelist_onlymonth = baseURL + str(year) + '/' + str(year) + '.' + month
                    filename = filelist_onlymonth + '/' + 'fnl_' + str(year) + month + day + '_' + hour + '_00.grib2'
                    filelist.append(filename)
            else:
                for hour in hour_list:  # get the month
                    filename = filelist_onlymonth + '/' + 'fnl_' + str(year) + month + day + '_' + hour + '_00.grib1'
                    filelist.append(filename)

        else:
            # baseURL = 'https://stratus.rda.ucar.edu/ds083.2/grib2/'
            baseURL = 'https://data.rda.ucar.edu/ds083.2/grib2/'
            filelist_onlymonth = baseURL + str(year) + '/' + str(year) + '.' + month
            for hour in hour_list:
                filename = filelist_onlymonth + '/' + 'fnl_' + str(year) + month + day + '_' + hour + '_00.grib2'
                filelist.append(filename)

    opener = build_opener()
    if check_command('aria2c') == True:
        print("Found available download tool: ari2c")
        download_type = 'aria2c'
    elif check_command('wget') == True:
        print("Found available download tool: wget")
        download_type = 'wget'
    else:
        print("no available download tool, download with opener")
        download_type = 'opener'

    for file in tqdm(filelist):
        # use the found download tool
        if download_type == 'aria2c':
            if os.system(f'aria2c --file-allocation=none --check-certificate=false {file}') == 0:  # 路径中有预设给服务器的下载文件存放路径
                pass
            else:
                tempfile = os.path.basename(file)
                if os.name == 'nt':
                    os.system(f'echo. > {tempfile}')
                if os.name == 'posix':
                    os.system(f'touch {tempfile}')
        elif download_type == 'wget':
            if os.system(f'wget --no-check-certificate  {file} -P . 2>&1') == 0:  # 路径中有预设给服务器的下载文件存放路径
                pass
            else:
                tempfile = os.path.basename(file)
                if os.name == 'nt':
                    os.system(f'echo. > {tempfile}')
                if os.name == 'posix':
                    os.system(f'touch {tempfile}')
        elif download_type == 'opener':
            ofile = os.path.basename(file)
            # print("downloading " + ofile + " ... ")os.system('wget {0} -P {1}'.format(gfs_url, save_date_hour_path))
            infile = opener.open(file)
            outfile = open(ofile, "wb")
            outfile.write(infile.read())
            outfile.close()

if __name__ == "__main__":
    # downLoadFNLfiles('2024-08-16','2024-08-16')
    pass
