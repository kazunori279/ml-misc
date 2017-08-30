#
# monitor.py
# 
# when the toilet isn't dark, takes a picture and check if the floor is wet. 
#

import time
import predict
from picamera import PiCamera
from picamera.exc import PiCameraRuntimeError
from grovepi import *

# I/O ports
LIGHT = 0
LED = 5
BUZZ = 6

# buzzer
def buzz():
  for i in range (0, 5):
    time.sleep(0.2)
    digitalWrite(BUZZ, 1)
    time.sleep(0.01)
    digitalWrite(BUZZ, 0)

# init models
print('Init models...')
predict.init_models()

# init I/O and camera
pinMode(LIGHT, 'INPUT')
pinMode(LED, 'OUTPUT')
pinMode(BUZZ, 'OUTPUT')
camera = PiCamera()

# monitoring loop
print('Started monitoring.')
while(True):

  # reset LED and buzzer
  digitalWrite(LED, 0)
  digitalWrite(BUZZ, 0)

  # do nothing when the toilet is dark
  if (analogRead(LIGHT) < 100):
    time.sleep(0.1)
    continue

  # capture 
  try:
    camera.capture('toilet.jpg')
  except PiCameraRuntimeError:
    print('Camera error')

  # get probability of wet floor 
  wet_prob = predict.predict()
  print('wet prob: ' + str(wet_prob))
  if(wet_prob > 0.2):
    digitalWrite(LED, 1)
    buzz()

