#
# predict.py
#
# checks the captured picture to calculate probability of wet floor
#

from keras import applications, optimizers
from keras.preprocessing import image
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
import numpy as np

filename = 'toilet.jpg'
img_height, img_width = 150, 150
channels = 3

# init models
model = None

def init_models():
  global model

  # build the VGG16 network
  base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))

  # build a classifier model to put on top of the convolutional model
  top_model = Sequential()
  top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
  top_model.add(Dense(256, activation='relu'))
  top_model.add(Dropout(0.5))
  top_model.add(Dense(1, activation='sigmoid'))

  # full model
  model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
  model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
  model.load_weights('full_model.h5')

# predict
def predict():
  img = image.load_img(filename, target_size=(img_height, img_width))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = x / 255.0
  return model.predict(x)[0]

