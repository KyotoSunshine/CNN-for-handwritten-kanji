# Run this after running create_dataset.py first
from __future__ import division
from scipy import misc
import pylab
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.signal import convolve2d

f = open("characters_dataset", "rb")
X_train = np.load(f)
y_train = np.load(f)
X_val = np.load(f)
y_val = np.load(f)
X_test = np.load(f)
y_test = np.load(f)
label_names = np.load(f)
f.close()


# See http://arxiv.org/pdf/1003.0358.pdf for the description of the method
def elastic_distortion(image, kernel_dim=31, sigma=6, alpha=47):

    # Returns gaussian kernel in two dimensions
    # d is the square kernel edge size, it must be an odd number.
    # i.e. kernel is of the size (d,d)
    def gaussian_kernel(d, sigma):
        if d % 2 == 0:
            raise ValueError("Kernel edge size must be an odd number")

        cols_identifier = np.int32(
            np.ones((d, d)) * np.array(np.arange(d)))
        rows_identifier = np.int32(
            np.ones((d, d)) * np.array(np.arange(d)).reshape(d, 1))

        kernel = np.exp(-1. * ((rows_identifier - d/2)**2 +
            (cols_identifier - d/2)**2) / (2. * sigma**2))
        kernel *= 1. / (2. * math.pi * sigma**2)  # normalize
        return kernel

    field_x = np.random.uniform(low=-1, high=1, size=image.shape) * alpha
    field_y = np.random.uniform(low=-1, high=1, size=image.shape) * alpha

    kernel = gaussian_kernel(kernel_dim, sigma)

    # Distortion fields convolved with the gaussian kernel
    # This smoothes the field out.
    field_x = convolve2d(field_x, kernel, mode="same")
    field_y = convolve2d(field_y, kernel, mode="same")

    d = image.shape[0]
    cols_identifier = np.int32(np.ones((d, d))*np.array(np.arange(d)))
    rows_identifier = np.int32(
        np.ones((d, d))*np.array(np.arange(d)).reshape(d, 1))

    down_row = np.int32(np.floor(field_x)) + rows_identifier
    top_row = np.int32(np.ceil(field_x)) + rows_identifier
    down_col = np.int32(np.floor(field_y)) + cols_identifier
    top_col = np.int32(np.ceil(field_y)) + cols_identifier
#    plt.imshow(field_x, cmap=pylab.cm.gray, interpolation="none")
#    plt.show()

    padded_image = np.pad(
        image, pad_width=d, mode="constant", constant_values=0)

    x1 = down_row.flatten()
    y1 = down_col.flatten()
    x2 = top_row.flatten()
    y2 = top_col.flatten()

    Q11 = padded_image[d+x1, d+y1]
    Q12 = padded_image[d+x1, d+y2]
    Q21 = padded_image[d+x2, d+y1]
    Q22 = padded_image[d+x2, d+y2]
    x = (rows_identifier + field_x).flatten()
    y = (cols_identifier + field_y).flatten()

    # Bilinear interpolation algorithm is as described here:
    # https://en.wikipedia.org/wiki/Bilinear_interpolation#Algorithm
    distorted_image = (1. / ((x2 - x1) * (y2 - y1)))*(
        Q11 * (x2 - x) * (y2 - y) +
        Q21 * (x - x1) * (y2 - y) +
        Q12 * (x2 - x) * (y - y1) +
        Q22 * (x - x1) * (y - y1))

    distorted_image = distorted_image.reshape((d, d))
    return distorted_image

distorted_train_set = []
for k in range(9):
    # uncomment the code below to view the original and transformed images
    for i, img in enumerate(X_train):
        original_img = (1. - img).reshape((32, 32))
    #  plt.imshow(original_img, cmap=pylab.cm.gray, interpolation="none")
    #  plt.show()
        distorted_image = elastic_distortion(
                original_img, kernel_dim=31, sigma=6, alpha=47)
        distorted_train_set.append((1. - distorted_image).reshape((1, 32, 32)))
    #   plt.imshow(distorted_image, cmap=pylab.cm.gray, interpolation="none")
    #   plt.show()
        if i % 1000 == 0:
            print i


f = open("characters_dataset_elastic", "wb")
np.save(f, X_train)
np.save(f, y_train)
np.save(f, X_val)
np.save(f, y_val)
np.save(f, X_test)
np.save(f, y_test)
np.save(f, label_names)  # label names of each class (same as folder names)
np.save(f, np.array(distorted_train_set))
f.close()
