#
# classify.py
#

import time
import numpy as np
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.preprocessing import image

top_model_weights_path = 'bottleneck_fc_model.h5'
filename = 'toilet.jpg'
vgg_model = None
top_model = None

# init the models
def init_models():

  # init VGG model
  global vgg_model
  vgg_model = applications.VGG16(include_top=False, weights='imagenet')

  # init top model
  global top_model
  top_model = Sequential()
  top_model.add(Flatten(input_shape=(4, 4, 512)))
  top_model.add(Dense(256, activation='relu'))
  top_model.add(Dropout(0.5))
  top_model.add(Dense(1, activation='sigmoid'))
  top_model.compile(optimizer='rmsprop',
                loss='binary_crossentropy', metrics=['accuracy'])
  top_model.load_weights(top_model_weights_path)

# classify with the model
def classify():

  # apply vgg model
  img = image.load_img(filename, target_size=(150, 150))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  preds = vgg_model.predict(applications.vgg16.preprocess_input(x))

  # apply top model
  final_preds = top_model.predict(preds)
  return final_preds[0][0] > 0.5


