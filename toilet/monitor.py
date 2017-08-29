#
# monitor.py
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

  # wait until someone get's in the toilet
  if (analogRead(LIGHT) < 100):
    time.sleep(0.1)
    continue

  # capture 
  try:
    camera.capture('toilet.jpg')
  except PiCameraRuntimeError:
    print('Camera error')

  # classify 
  is_wet = predict.predict()
  print('is_wet: ' + str(is_wet))
  if(is_wet):
    digitalWrite(LED, 1)
    buzz()

