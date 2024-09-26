import pygame
import RPi.GPIO as RPIO
from DRV8825 import DRV8825
import time
import os
import pygame.camera
import sys
steplang = 2500

if len(sys.argv) != 3:
    print("Usage: slidename dimesion")
    exit(1)
if isinstance(int(sys.argv[2]), int) == False:
    print("Improper dimesion")
    exit(1)
dim = int(sys.argv[2])

dr = "/media/pi/PortableSSD/" 
dirname = "Slide"+sys.argv[1]
print("Starting slide:")
print(sys.argv[1])
try:
    path = os.path.join(dr, dirname)
    os.mkdir(path)
except:
    print("Slide with this number has been processed")
    exit(1)

def take(camera, a, b):
    # Use a breakpoint in the code line below to debug your script.
    pic = cam.get_image()
    pygame.image.save(pic, fileName(a,b))
def fileName(a,b):
    num = str((a*100)+b)
    name = (dr+'/'+dirname+'/' + 'ISR' + num + '.png')
    return name
def moveRight():
    Motor2.TurnStep(Dir='forward', steps=steplang, stepdelay=0.00)
    Motor2.Stop()
def moveLeft():
    Motor2.TurnStep(Dir='backward', steps=(steplang), stepdelay=0.00)
    Motor2.Stop()
def moveDown():
    Motor1.TurnStep(Dir='forward', steps=steplang, stepdelay=0.00)
    Motor1.Stop()
def moveUp():
    Motor1.TurnStep(Dir='backward', steps=(steplang), stepdelay=0.0)
    Motor1.Stop()
pygame.camera.init()
camlist = pygame.camera.list_cameras()
if  camlist:
	cam = pygame.camera.Camera(camlist[0], (1920,1080))
	cam.start()
else:
	print("\n Error: Camera Refused to Open")
	exit(1)
try:
    Motor1 = DRV8825(dir_pin=13, step_pin=19, enable_pin=12, mode_pins=(16, 17, 20))
    Motor2 = DRV8825(dir_pin=24, step_pin=18, enable_pin=4, mode_pins=(21, 22, 27))
    Motor1.SetMicroStep('softward','1/32step')
    Motor2.SetMicroStep('softward','1/32step')
    
except:
    RPIO.cleanup()
    print("\n Error: Failure Mounting Motors")
    cam.stop()
    exit(1)
print("\n Startup Successful: Beginning Scanning")

for i in range(dim):
    for k in range(dim):
        moveDown()
        time.sleep(0.5)
        take(cam,i,k)
    moveUp()
    moveRight()
moveDown()

print("Slide Scanned Successfully")
Motor1.Stop()
Motor2.Stop()
cam.stop()
RPIO.cleanup()
