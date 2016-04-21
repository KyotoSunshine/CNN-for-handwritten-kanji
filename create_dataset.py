# This script takes as an argument the path to the folder which
# contains folders of images.
# It is assumed that name of each folder with images is
# the label for the images, that is all the images in each folder belong
# to the the same class, and the name of that class is the name of the folder.
# Images are assumed to be in the .png format. It is also assumed that
# each folder has the same number of images. It is NOT assumed that all images
# have the same dimensionality, but all the images will be rescaled to 32x32
# before being saved into the dataset file.
# The total number of images is assumed to be divisible by 10.

# The script will produce a file named "characters_dataset" which will
# contain the train/validation/test datasets and labels in numpy arrays.
# The file will also contain the names of all the
# image folders in alphabetic order.

import os
import sys
from scipy import misc
import numpy as np

# The path that you have your image folders in
path = sys.argv[1]

# We rescale each image to be of size "SHAPE"
SHAPE = (32, 32)

# folder_names is a sorted list containing names of all the folders with images
folder_names = []
for name in sorted(os.listdir(path)):
    if os.path.isdir(os.path.join(path, name)):
        folder_names.append(name)

# Each element of folder_files is a sorted list of file names
# that are contained within a folder from folder_names
folder_files = []
for folder_name in folder_names:
    folder_files.append(sorted(os.listdir(os.path.join(path, folder_name))))


number_of_classes = len(folder_names)

# we assume that all classes have the same number of elements
number_of_examples_per_class = len(folder_files[0])

# the data samples X and the labels y
X = []
y = []

# Load the images and labels into numpy arrays
for i in range(number_of_classes):
    for j in range(number_of_examples_per_class):
        image_location = os.path.join(
            path, folder_names[i], folder_files[i][j])
        image = misc.imread(image_location)
        image = misc.imresize(image, size=SHAPE, interp='bilinear', mode=None)
        X.append(image)
        y.append(i)

# Turn the samples into proper numpy array of type
# float32 (for use with GPU) rescaled in [0,1] interval.
X = np.float32(np.array(X)/255.0)
y = np.int32(np.array(y))
hex_codes = np.array(folder_names)

# Make so that each batch of size "number_of_classes" samples is
# balanced with respect to classes.
# That is, each batch of size "number_of_classes" samples
# will contain exactly one sample of each class.
# In this way, when we split the data into train, validation, and test
# datasets, all of them will be balanced with respect to classes
# as long as the sizes of all of them are divisible by "number_of_classes".
X = np.concatenate(
    [X[i::number_of_examples_per_class]
        for i in range(number_of_examples_per_class)])
y = np.concatenate(
    [y[i::number_of_examples_per_class]
        for i in range(number_of_examples_per_class)])


dataset_size = number_of_classes * number_of_examples_per_class

# train - validation - test split is 80% - 10% - 10%
# We also assume that the dataset_size is divisible by 10.
X_train = X[:(dataset_size*8)//10]
y_train = y[:(dataset_size*8)//10]

X_val = X[(dataset_size*8)//10:(dataset_size*9)//10]
y_val = y[(dataset_size*8)//10:(dataset_size*9)//10]

X_test = X[(dataset_size*9)//10:]
y_test = y[(dataset_size*9)//10:]

f = open("characters_dataset", "wb")
np.save(f, X_train)
np.save(f, y_train)
np.save(f, X_val)
np.save(f, y_val)
np.save(f, X_test)
np.save(f, y_test)
np.save(f, hex_codes)  # hex codes of each class (same as folder names)
f.close()
