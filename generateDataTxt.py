# import nipy

# epi_img = nipy.load_image('volume-0.nii')
# print(epi_img.coordmap)
# print(epi_img.coordmap.affine)
import os
from PIL import Image

f = open('alltraindata.txt', 'w')
filepath = 'data/liver_seg_png/'


def traverse(filepath):
    fs = os.listdir(filepath)
    for f1 in fs:
        # print(len(fs))
        tmp_path = os.path.join(filepath, f1)
        if not os.path.isdir(tmp_path):
            print('文件: %s' % tmp_path)
            volpath = filepath.replace('liver_seg_png', 'volume_png')
            liverpath = filepath
            print(f1)
            if (f1.split('.')[0] == '1'):
                i = eval(f1.split('.')[0]) + 1
                j = eval(f1.split('.')[0]) + 2
                f.write(volpath +'/'+ f1 + ' ' + liverpath+'/' + f1 + ' ' + volpath+'/' + str(
                    i) + '.png' + ' ' + liverpath+'/' + str(
                    i) + '.png' + ' ' + volpath+'/' + str(
                    j) + '.png' + ' ' + liverpath+'/' + str(
                    j) + '.png' + '\n')
            elif (f1.split('.')[0] == str(len(fs))):
                i = eval(f1.split('.')[0]) - 2
                j = eval(f1.split('.')[0]) - 1
                f.write(volpath+'/' + str(i) + '.png' + ' ' + liverpath+'/' + str(
                    i) + '.png' + ' ' + volpath+'/' + str(
                    j) + '.png' + ' ' + liverpath+'/' + str(
                    j) + '.png' + ' ' + volpath+'/' + f1 + ' ' + liverpath+'/' + f1 + '\n')
            else:
                i = eval(f1.split('.')[0]) - 1
                j = eval(f1.split('.')[0]) + 1

                f.write(volpath+'/' + str(i) + '.png' + ' ' + liverpath+'/' + str(
                    i) + '.png' + ' ' + volpath+'/' + f1 + ' ' + liverpath+'/' + f1 + ' ' + volpath+'/' + str(
                    j) + '.png' + ' ' + liverpath+'/' + str(j) + '.png' + '\n')
        else:
            print('文件夹：%s' % tmp_path)
            traverse(tmp_path)


traverse(filepath)
