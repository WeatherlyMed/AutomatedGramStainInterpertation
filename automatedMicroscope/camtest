import pygame
import pygame.camera
pygame.init()
pygame.camera.init()
screen = pygame.display.set_mode((640, 480))
camlist = pygame.camera.list_cameras()
if camlist:
    camera = pygame.camera.Camera(camlist[0], (640, 480))
else:
    print("Camera Failed to Open")
    exit(1)
camera.start()
pic = camera.get_image()
screen.blit(pic, (0, 0))
pygame.display.update()
camera.stop()
pygame.quit()
