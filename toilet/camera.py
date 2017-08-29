import time
from picamera import PiCamera

camera = PiCamera()
c = 5000
while(True):
  time.sleep(0.2)
  filename = str(c) + '.jpg'
  camera.capture(filename)
  print(filename)
  c = c + 1
