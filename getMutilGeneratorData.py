from PIL import Image
import numpy as np
import os
from keras.utils import np_utils
import threading
import random
from keras.preprocessing.image import img_to_array, array_to_img
import scipy


def cropImage(imagepath):
    image = Image.open(imagepath)
    image = image.resize((224, 224))
    return np.array(image)


def process_line(line):
    tmp = line.strip().split(' ')
    r1, x, y, z = tmp[0], tmp[1], tmp[2], tmp[3]
    volpath1 = 'data/volume_png/' + r1 + '/' + x + '.png'
    volpath2 = 'data/volume_png/' + r1 + '/' + y + '.png'
    volpath3 = 'data/volume_png/' + r1 + '/' + z + '.png'

    forepath2 = volpath2.replace('volume_png/', 'train_volume_seg_png/fore_seg_png/')

    itempath1 = volpath1.replace('volume_png/', 'train_volume_seg_png/item_seg_png/')
    itempath2 = volpath2.replace('volume_png/', 'train_volume_seg_png/item_seg_png/')
    itempath3 = volpath3.replace('volume_png/', 'train_volume_seg_png/item_seg_png/')

    liverpath1 = volpath1.replace('volume_png/', 'train_volume_seg_png/liver_seg_withoutlesion_png/')
    liverpath2 = volpath2.replace('volume_png/', 'train_volume_seg_png/liver_seg_withoutlesion_png/')
    liverpath3 = volpath3.replace('volume_png/', 'train_volume_seg_png/liver_seg_withoutlesion_png/')

    # print(volpath1,volpath2,volpath3)
    # print(liverpath1, liverpath2, liverpath3)
    # print(itempath1, itempath2, itempath3)

    s1 = cropImage(volpath1)
    s2 = cropImage(volpath2)
    s3 = cropImage(volpath3)

    l1 = cropImage(forepath2)
    l2 = cropImage(liverpath2)
    l3 = cropImage(itempath2)

    img = np.zeros((224, 224, 3))
    img[:, :, 0] = s1
    img[:, :, 1] = s2
    img[:, :, 2] = s3
    # img = preprocess_img(img, 3)
    # print(img.shape)

    label = np.zeros((224, 224, 3))
    label[:, :, 0] = l1
    label[:, :, 1] = l2
    label[:, :, 2] = l3

    img = img / 255

    # max_mask = np.max(label) * 0.5
    # label = np.greater(label, max_mask)

    label /= 255
    label[label > 0.5] = 1
    label[label <= 0.5] = 0

    # print(np.array(img).shape)
    # print(np.array(label).shape)

    # print(img)
    # print(label)

    # print(np.sum(label[:, :, 0]), np.sum(label[:, :, 1]), np.sum(label[:, :, 2]))
    # print(np.sum(label[:, :, 0]) + np.sum(label[:, :, 1]) + np.sum(label[:, :, 2]))

    return img, label


def preprocess_input(x):
    x = np.divide(x, 255.0)
    x = np.subtract(x, 0.5)
    x = np.multiply(x, 2.0)
    return x


# class threadsafe_iter:
#     """Takes an iterator/generator and makes it thread-safe by
#     serializing call to the `next` method of given iterator/generator.
#     """
#
#     def __init__(self, it):
#         self.it = it
#         self.lock = threading.Lock()
#
#     def __iter__(self):
#         return self
#
#     def next(self):
#         with self.lock:
#             return self.it.next()


# # process_line('trainImages/9947.png 1\n')
# def threadsafe_generator(f):
#     """A decorator that takes a generator function and makes it thread-safe.
#     """
#
#     def g(*a, **kw):
#         return threadsafe_iter(f(*a, **kw))
#
#     return g
#
#
# @threadsafe_generator


def preprocess_img(image, number_slices):
    """Preprocess the image to adapt it to network requirements
    Args:
    Image we want to input the network (W,H,3) numpy array
    Returns:
	Image ready to input the network (1,W,H,3)
    """
    images = [[] for i in range(np.array(image).shape[0])]

    if number_slices > 2:
        for j in range(np.array(image).shape[0]):
            if type(image) is not np.ndarray:
                for i in range(number_slices):
                    images[j].append(np.array(scipy.io.loadmat(image[0][i])['section'], dtype=np.float32))
            else:
                img = image
    else:
        for j in range(np.array(image).shape[0]):
            for i in range(3):
                images[j].append(np.array(scipy.io.loadmat(image[0][0])['section'], dtype=np.float32))
    in_ = np.array(images[0])
    in_ = in_.transpose((1, 2, 0))
    in_ = np.expand_dims(in_, axis=0)
    return in_


def preprocess_labels(label, number_slices):
    """Preprocess the labels to adapt them to the loss computation requirements
    Args:
    Label corresponding to the input image (W,H) numpy array
    Returns:
    Label ready to compute the loss (1,W,H,1)
    """
    labels = [[] for i in range(np.array(label).shape[0])]

    for j in range(np.array(label).shape[0]):
        if type(label) is not np.ndarray:
            for i in range(number_slices):
                labels[j].append(np.array(Image.open(label[0][i]), dtype=np.uint8))

    label = np.array(labels[0])
    label = label.transpose((1, 2, 0))
    max_mask = np.max(label) * 0.5
    label = np.greater(label, max_mask)
    label = np.expand_dims(label, axis=0)

    return label


def generate_weighted_arrays(batch_size, path):
    while 1:
        f = open(path)

        list = []
        for line in f:
            list.append(line)
        random.shuffle(list)

        X = []
        Y = []
        W = []
        cnt = 0
        for count in range(len(list)):
            # print(list[count])
            x, y = process_line(list[count])

            w1 = np.sum(y[:, :, 0]) / (y.shape[0] * y.shape[1])
            w2 = np.sum(y[:, :, 1]) / (y.shape[0] * y.shape[1])
            w3 = np.sum(y[:, :, 2]) / (y.shape[0] * y.shape[1])

            w = np.zeros((224, 224, 3))
            w[:, :, 0] = y[:, :, 0] * w1
            w[:, :, 1] = y[:, :, 1] * w2
            w[:, :, 2] = y[:, :, 2] * w3

            X.append(x)
            Y.append(y)
            W.append(w)

            cnt += 1
            if cnt == batch_size:
                cnt = 0
                # Y = np_utils.to_categorical(Y, 2)
                yield ([[np.array(X), np.array(Y)], np.array(W)])
                # print('read')
                X = []
                Y = []
                W = []
    f.close()


def generate_arrays(batch_size, path):
    while 1:
        f = open(path)

        list = []
        for line in f:
            list.append(line)
        random.shuffle(list)

        X = []
        Y = []
        cnt = 0
        for count in range(len(list)):
            # print(list[count])
            x, y = process_line(list[count])

            # y = y / 255
            # x = preprocess_input(x)
            X.append(x)
            Y.append(y)

            cnt += 1
            if cnt == batch_size:
                cnt = 0
                # Y = np_utils.to_categorical(Y, 2)
                yield (np.array(X), np.array(Y))
                # print('read')
                X = []
                Y = []
    f.close()

#
# f = open('train_mutil.txt')
# for line in f:
#     process_line(line)
