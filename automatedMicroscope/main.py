import pygame
import RPi.GPIO as RPIO
from DRV8825 import DRV8825
import time
import os
import pygame.camera
import sys
steplang = 2500
dr = "/media/pi/PortableSSD/" 

if len(sys.argv) != 3:
    print("Usage: slidename dimesion")
    exit(1)
if isinstance(int(sys.argv[2]), int) == False:
    print("Improper dimesion")
    exit(1)
dim = int(sys.argv[2])

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
def moveRight(steplang):
    Motor2.TurnStep(Dir='forward', steps=steplang, stepdelay=0.00)
    Motor2.Stop()
def moveLeft(steplang):
    Motor2.TurnStep(Dir='backward', steps=(steplang), stepdelay=0.00)
    Motor2.Stop()
def moveDown(steplang):
    Motor1.TurnStep(Dir='forward', steps=steplang, stepdelay=0.00)
    Motor1.Stop()
def moveUp(steplang):
    Motor1.TurnStep(Dir='backward', steps=(steplang), stepdelay=0.0)
    Motor1.Stop()
def intialSet():
    print("Adjust to Upper left corner of slide")
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption('Slide Adjustment')
    steplang = 10  # Define a default step length
    total_length_adjustedx = 0
    total_length_adjustedy = 0
    adjusting = True
    camlist = pygame.camera.list_cameras()
    if camlist:
        cam = pygame.camera.Camera(camlist[0])
	cam.start()
    else:
	print("\n Error: Camera Failed to Open")
	return
    while adjusting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    moveLeft(steplang)
                    total_length_adjustedx -= steplang
                elif event.key == pygame.K_RIGHT:
                    moveRight(steplang)
                    total_length_adjustedx += steplang
                elif event.key == pygame.K_UP:
                    moveUp(steplang)
                    total_length_adjustedy -= steplang
                elif event.key == pygame.K_DOWN:
                    moveDown(steplang)
                    total_length_adjustedy += steplang
                elif event.key == pygame.K_RETURN:
                    adjusting = False
		if cam.query_image():
	            frame = cam.get_image()
	            screen.blit(frame, (0, 0))
	            pygame.display.update()
    cam.stop()
    pygame.quit()
    return total_length_adjustedx, total_length_adjustedy


pygame.camera.init()
camlist = pygame.camera.list_cameras()
if  camlist:
	cam = pygame.camera.Camera(camlist[0], (1920,1080))
	cam.start()
	print("camera initalised")
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

total_length_adjusted = intialSet()
stepx = total_length_adjusted[0] / dim
stepy = total_length_adjusted[1] / dim
for i in range(dim):
    for k in range(dim):
        moveDown(stepy)
        time.sleep(0.5)
        take(cam,i,k)
    moveUp(stepx)
    moveRight(total_length_adjusted[0])
moveDown(total_length_adjusted[1])

print("Slide Scanned Successfully")
Motor1.Stop()
Motor2.Stop()
cam.stop()
RPIO.cleanup()
