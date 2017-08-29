#
# predict.py
#

import os
import sys
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing import image
import numpy as np

filename = 'toilet.jpg'
img_height, img_width = 150, 150
channels = 3

# init models
model = None

def init_models():

  # VGG16
  input_tensor = Input(shape=(img_height, img_width, channels))
  vgg16_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

  # top model 
  top_model = Sequential()
  top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
  top_model.add(Dense(256, activation='relu'))
  top_model.add(Dropout(0.5))
  top_model.add(Dense(1, activation='sigmoid'))
  top_model.load_weights('bottleneck_fc_model.h5')

  # final model 
  global model
  model = Model(input=vgg16_model.input, output=top_model(vgg16_model.output))
  model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
  #model.summary()

# predict
def predict():
  img = image.load_img(filename, target_size=(img_height, img_width))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = x / 255.0
  return model.predict(x)[0] > 0.5

