import numpy as np

# img = np.ones((2, 2, 3))
# img[:, :, 0] = [[1, 2], [3, 4]]
# img[:, :, 1] = [[5, 6], [7, 8]]
# img[:, :, 2] = [[9, 10], [11, 12]]
# img = img / 14
# img[img >= 0.8] = 1
# img[img < 0.8] = 0
# print(img)
#
# s1 = np.sum(img) / 12
# s2 = 1 - s1
# print(s1, s2)
#
# condition = img > 0
# rep = np.where(condition, s1, s2)
# print(rep)

# rep = ['d' if x == 0 else x for x in img.tolist()]
#
# print(rep)

# a = np.random.randint(1, 10, (5, 5, 3))
# print(a/10)
# a=a/10
# a[a >= 0.5] = 1
#
# a[a < 0.5] = 0
# print(a)

from keras import losses
losses.binary_crossentropy()

