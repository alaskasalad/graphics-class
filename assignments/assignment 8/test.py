import pygame
pygame.init()
pygame.display.set_mode((400, 300))
pygame.display.set_caption("Test Pygame Window")

import moderngl
ctx = moderngl.create_standalone_context()
print("GL_VERSION:", ctx.info["GL_VERSION"])


running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
pygame.quit()
