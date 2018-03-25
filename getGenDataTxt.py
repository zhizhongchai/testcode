# import nipy

# epi_img = nipy.load_image('volume-0.nii')
# print(epi_img.coordmap)
# print(epi_img.coordmap.affine)
import os
from PIL import Image

f = open('alltraindata_mutil.txt', 'w')
filepath = 'data/volume_png/'


def getfileline(v_num, num):
    i = num - 1
    j = num + 1

    f.write(v_num + ' ' + str(i) + ' ' + str(num) + ' ' + str(j) + ' ' + '\n')


def traverse(filepath):
    fs = os.listdir(filepath)
    for f1 in fs:
        print(len(fs))
        tmp_path = os.path.join(filepath, f1)
        if not os.path.isdir(tmp_path):
            print('文件: %s' % tmp_path)
            print(filepath.split('/')[2])
            if (f1.split('.')[0] == '1'):
                getfileline(filepath.split('/')[2], eval(f1.split('.')[0]) + 1)
            elif (f1.split('.')[0] == str(len(fs))):
                getfileline(filepath.split('/')[2], eval(f1.split('.')[0]) - 1)
            else:
                getfileline(filepath.split('/')[2], eval(f1.split('.')[0]))
        else:
            print('文件夹：%s' % tmp_path)
            traverse(tmp_path)


traverse(filepath)
