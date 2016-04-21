# This script takes a list of image paths as arguments and
# outputs the class label for each image.
# The script loads a model from files "model_architecture.json"
# and "model_weights.h5", so these files must be present in the
# same folder as the script before running it.

import sys
import numpy as np
from scipy import misc
from keras.models import model_from_json
from keras.optimizers import Adam

# this shortcut will generate all the class names as they were given
class_names = ["B0"+hex(a)[-2:].upper() for a in range(161, 255)]
class_names += ["B1"+hex(a)[-2:].upper() for a in range(161, 167)]

f = open('model_architecture.json', 'r')
model = model_from_json(f.read())
f.close()

model.load_weights('model_weights.h5')

# SHAPE can also potentially be inferred automatically from the loaded model
SHAPE = (32, 32)

X = []

for image_location in sys.argv[1:]:
    image = misc.imread(image_location)
    image = misc.imresize(image, size=SHAPE, interp='bilinear', mode=None)
    X.append(image)

X = np.float32(np.array(X)/255.0)
X = np.reshape(X, (len(X), 1, SHAPE[0], SHAPE[1]))

adam = Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

predicted_classes = model.predict_classes(X, batch_size=1, verbose=1)

for i, predicted_class in enumerate(predicted_classes):
    print sys.argv[i+1], class_names[predicted_class]
