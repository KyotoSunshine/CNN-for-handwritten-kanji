# This script creates a keras model for classifying handwritten kanji
# in augmented dataset created with elastic distortions using the
# "create_augmented_dataset.py" script.
# It saves the model into files "augmented_model_architecture.json"
# and "augmented_model_weights.h5".
#
# This script is essentially the same as "model.py", except is has
# larger number of neurons in the fully connected layers, larger number of
# filters in convolutional layers, and it does not retrain the data on
# the whole dataset after training it on training set and validating it on
# the validation set.
# It also does the "early stopping"->"decrease learning rate" cycle three
# times instead of two in the "model.py".

import numpy as np
import sys
from keras.utils import np_utils
from keras.callbacks import Callback, EarlyStopping
from keras.models import Sequential
from keras import backend
from keras.constraints import maxnorm
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam
# Just to make sure that it is installed, because it is needed
# for weights saving.
import h5py

# Flush the stdout and stderr after each epoch
class Flush(Callback):
    def on_epoch_end(self, epoch, logs={}):
        sys.stdout.flush()
        sys.stderr.flush()


f = open("characters_dataset_elastic", "rb")
X_train = np.load(f)
y_train = np.load(f)
X_val = np.load(f)
y_val = np.load(f)
X_test = np.load(f)
y_test = np.load(f)
label_names = np.load(f)
X_train_elastic = np.load(f)
f.close()

# Reshape the samples array into the
# form (number_of_samples, depth, height, width).
# Since our input is grayscale we only use one channel (i.e. depth=1).
X_train = np.reshape(
    X_train, (X_train.shape[0], 1, X_train.shape[1], X_train.shape[2]))
X_val = np.reshape(
    X_val, (X_val.shape[0], 1, X_val.shape[1], X_val.shape[2]))
X_test = np.reshape(
    X_test, (X_test.shape[0], 1, X_test.shape[1], X_test.shape[2]))

y_train_elastic = np.concatenate((y_train,)*(len(X_train_elastic)//len(X_train)))
X_train = np.concatenate((X_train, X_train_elastic))
y_train = np.concatenate((y_train, y_train_elastic))
del X_train_elastic
del y_train_elastic



# Some parameters that will not change anymore
MINIBATCH_SIZE = 100
NUMBER_OF_CLASSES = len(label_names)
MAX_NORM = 4  # Max-norm constraint on weights
SAMPLE_SHAPE = X_train[0].shape
INITIAL_ADAM_LEARNING_RATE = 0.01
# If you don't mind long training times, make the below two values larger
MAXIMUM_NUMBER_OF_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 4

# Convert labels to one-hot representation
y_train = np_utils.to_categorical(y_train, NUMBER_OF_CLASSES)
y_val = np_utils.to_categorical(y_val, NUMBER_OF_CLASSES)
y_test = np_utils.to_categorical(y_test, NUMBER_OF_CLASSES)

depth = 40
model = Sequential()
model.add(Convolution2D(
    depth, 5, 5, border_mode='same',
    W_constraint=maxnorm(MAX_NORM),
    init='he_normal',
    input_shape=(SAMPLE_SHAPE[0], SAMPLE_SHAPE[1], SAMPLE_SHAPE[2])))
model.add(BatchNormalization())
model.add(PReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

depth *= 2
model.add(Convolution2D(
    depth, 5, 5, init='he_normal',
    border_mode='same', W_constraint=maxnorm(MAX_NORM)))
model.add(BatchNormalization())
model.add(PReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

depth *= 2
model.add(Convolution2D(
    depth, 5, 5, init='he_normal',
    border_mode='same', W_constraint=maxnorm(MAX_NORM)))
model.add(BatchNormalization())
model.add(PReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(3000, init='he_normal', W_constraint=maxnorm(MAX_NORM)))
model.add(BatchNormalization())
model.add(PReLU())
model.add(Dropout(0.5))

model.add(Dense(3000, init='he_normal', W_constraint=maxnorm(MAX_NORM)))
model.add(BatchNormalization())
model.add(PReLU())
model.add(Dropout(0.5))

model.add(Dense(NUMBER_OF_CLASSES, W_constraint=maxnorm(MAX_NORM)))
model.add(Activation('softmax'))

adam = Adam(lr=INITIAL_ADAM_LEARNING_RATE)
model.compile(
    loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

for i in range(3):
    results = model.fit(
        X_train, y_train,
        batch_size=MINIBATCH_SIZE,
        nb_epoch=MAXIMUM_NUMBER_OF_EPOCHS,
        # We have already selected which model architecture and parameters
        # to use, so we don't discriminate between validation and test
        # sets, and combine them both into a validation set
        validation_data=(
            np.concatenate((X_val, X_test)), np.concatenate((y_val, y_test))),
        shuffle=True,
        verbose=2,
        callbacks=[
            EarlyStopping(
                monitor='val_loss', patience=EARLY_STOPPING_PATIENCE,
                verbose=2, mode='auto'),
            Flush()]
        )

    # Divide the learning rate by 10
    backend.set_value(adam.lr, 0.1 * backend.get_value(adam.lr))

# Save the model representation and weights to files
model_in_json = model.to_json()
f = open('augmented_model_architecture.json', 'w')
f.write(model_in_json)
f.close()
model.save_weights('augmented_model_weights.h5', overwrite=True)
