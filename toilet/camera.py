import time
from picamera import PiCamera

camera = PiCamera()
while(True):
  time.sleep(0.1)
  filename = str(time.time()) + '.jpg'
  camera.capture(filename)
  print(filename)
