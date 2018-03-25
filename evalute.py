from keras.models import load_model
from PIL import Image
import numpy as np


def cropImage(imagepath):
    image = Image.open(imagepath)
    image = image.resize((224, 224))
    return np.array(image)


model = load_model('weights.19-0.00557.h5')

line = 'data/test/test_volume_png/0/122.png data/test/test_volume_png/0/123.png data/test/test_volume_png/0/124.png'
tmp = line.strip().split(' ')
imageslice1 = tmp[0]
imageslice2 = tmp[1]
imageslice3 = tmp[2]

s1 = cropImage(imageslice1)
s2 = cropImage(imageslice2)
s3 = cropImage(imageslice3)

img = np.zeros((224, 224, 3))
img[:, :, 0] = s1
img[:, :, 1] = s2
img[:, :, 2] = s3

img = img / 255

input = img[np.newaxis, :]
# print(input.shape)

result = model.predict(input)

# result = result[0]
# print(result[0], result[0].shape)
label = result[0]
label[label > 0.5] = 1
label[label <= 0.5] = 0
print(label, label.shape)
# result_img = Image.fromarray(label * 255)
# result_img.save('result.png')
# print(result.shape)

from keras.preprocessing.image import array_to_img

result_img = array_to_img(label * 255)
result_img = result_img.resize((512, 512))
result_img = result_img.transpose(Image.FLIP_LEFT_RIGHT)
result_img.save('result.png')
