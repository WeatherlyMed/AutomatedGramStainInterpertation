import pygame
import RPi.GPIO as RPIO
from DRV8825 import DRV8825
import time
import pygame.camera

steplang = 3

def take_and_display(camera):
    # Capture image
    pic = camera.get_image()
    # Display image on screen
    screen.blit(pic, (0, 0))
    pygame.display.update()

pygame.init()
pygame.camera.init()

# Set up display
screen = pygame.display.set_mode((640, 480))

# Get camera list and start camera
camlist = pygame.camera.list_cameras()
if camlist:
  cam = pygame.camera.Camera(camlist[0], (640, 480))
else:
  print("Camera Failed to Open")
  exit(1)
cam.start()
Motor1 = DRV8825(dir_pin=13, step_pin=19, enable_pin=12, modepins=(16, 17, 20))
Motor2 = DRV8825(dir_pin=24, step_pin=18, enable_pin=4, modepins=(21, 22, 27))
Motor1.SetMicroStep('softward', '1/32step')
Motor2.SetMicroStep('softward', '1/32step')
i, j = 32, 32  # Set the dimensions of the photo grid
for x in range(i):
  for y in range(j):
    take_and_display(cam)
    time.sleep(0.05)
    Motor2.TurnStep(Dir='forward', steps=steplang, stepdelay=0.00)
    Motor2.Stop()
  Motor1.TurnStep(Dir='forward', steps=steplang, stepdelay=0.00)
  Motor1.Stop()
##End Process
RPIO.cleanup()
Motor1.Stop()
Motor2.Stop()
cam.stop()
pygame.quit()
