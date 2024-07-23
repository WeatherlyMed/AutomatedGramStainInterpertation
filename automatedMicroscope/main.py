import pygame
import RPi.GPIO as RPIO
from DRV8825 import DRV8825
import time
import os
import pygame.camera

slide_number = "1" 
dimension = 5 

if not isinstance(dimension, int):
    print("Improper dimension")
    exit(1)

dr = "/media/pi/PortableSSD/" 
dirname = "Slide" + slide_number
try:
    path = os.path.join(dr, dirname)
    os.mkdir(path)
except FileExistsError:
    print("Slide has been processed")
    exit(1)

steplang = 2500

def take(camera, a, b):
    pic = camera.get_image()
    pygame.image.save(pic, fileName(a, b))

def fileName(a, b):
    num = str((a * 100) + b)
    name = os.path.join(dr, dirname, 'ISR' + num + '.png')
    return name

def moveRight():
    Motor2.TurnStep(Dir='forward', steps=steplang, stepdelay=0.00)
    Motor2.Stop()

def moveLeft():
    Motor2.TurnStep(Dir='backward', steps=steplang, stepdelay=0.00)
    Motor2.Stop()

def moveDown():
    Motor1.TurnStep(Dir='forward', steps=steplang, stepdelay=0.00)
    Motor1.Stop()

def moveUp():
    Motor1.TurnStep(Dir='backward', steps=steplang, stepdelay=0.0)
    Motor1.Stop()

pygame.camera.init()
camlist = pygame.camera.list_cameras()
if camlist:
    cam = pygame.camera.Camera(camlist[0], (1920, 1080))
    cam.start()
else:
    print("\error cam")
    exit(1)

try:
    Motor1 = DRV8825(dir_pin=13, step_pin=19, enable_pin=12, mode_pins=(16, 17, 20))
    Motor2 = DRV8825(dir_pin=24, step_pin=18, enable_pin=4, mode_pins=(21, 22, 27))
    Motor1.SetMicroStep('software', '1/32step')
    Motor2.SetMicroStep('software', '1/32step')
except Exception as e:
    RPIO.cleanup()
    print("\nFailure Mounting:", e)
    cam.stop()
    exit(1)

for i in range(dimension):
    for k in range(dimension):
        moveDown()
        time.sleep(0.5)
        take(cam, i, k)
    moveUp()
    moveRight()
moveDown()

## end protocol
Motor1.Stop()
Motor2.Stop()
cam.stop()
RPIO.cleanup()
