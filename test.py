from PIL import Image
import os
import SimpleITK as sitk

# image = sitk.ReadImage('volume-0.nii')
# image_array = sitk.GetArrayFromImage(image)  # z, y, x
# for i in range(image_array.shape[0]):
#     img = Image.fromarray(image_array[i]).convert('L')
#     img.save(str(i) + '.png')
#     print(i)
# print(image_array.shape)

filepath = 'data/liver_seg_png/'
savepath = 'data/new/liver_seg_png/'


def traverse(filepath):
    fs = os.listdir(filepath)
    for f1 in fs:
        print(len(fs))
        tmp_path = os.path.join(filepath, f1)
        if not os.path.isdir(tmp_path):
            print('文件: %s' % filepath + '/' + f1)
            img = Image.open(filepath + '/' + f1)
            # img = img.rotate(270)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            print(filepath.replace('data/', 'data/new/') + '/' + f1)
            path = filepath.replace('data/', 'data/new/') + '/'
            isExists = os.path.exists(path)
            if not isExists:
                os.makedirs(path)
            img.save(path + f1)

        else:
            print('文件夹：%s' % tmp_path)
            traverse(tmp_path)


# img = Image.open('data/liver_seg/0/60.png')
# img = img.rotate(270)
# img.save('60.png')

traverse(filepath)
